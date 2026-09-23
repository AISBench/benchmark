from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

from . import __version__
from .artifacts import (
    artifact_paths,
    find_latest_execution_manifest,
    sha256_file,
    validate_artifacts,
    write_json,
)
from .errors import PrefixCacheError
from .multimodal import (
    prepare_multimodal_scenario,
    report_performance,
    run_multimodal_benchmark,
    validate_multimodal_manifest,
)
from .pipeline import inspect_scenario, prepare_scenario
from .runtime import analyze_snapshots, run_scenario
from .scenario import (
    Scenario,
    load_scenario,
    new_execution_timestamp,
    validate_scenario_mode,
    with_execution_timestamp,
)

# Parent logger name shared by all module loggers (ais_bench_prefix_cache.*).
PLUGIN_LOG_NAME = "ais_bench_prefix_cache"

LOG_NORMAL_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

# 显式挂在 PLUGIN_LOG_NAME 之下（不用 __name__）：python -m 运行时 __name__ 会变成
# "__main__"，导致日志绕过插件 logger 直接传播到 root。
logger = logging.getLogger(f"{PLUGIN_LOG_NAME}.cli")


def _format_hit_rate(value: object) -> str:
    """Format a 0..1 hit-rate value for the final CLI summary."""
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _format_hit_rate_difference(value: object) -> str:
    """Format the difference between two percentage hit-rate values."""
    if value is None:
        return "N/A"
    return f"{float(value):.2f}%"


def _format_run_summary_heading(analysis: dict) -> str:
    """Render an AISBench-style heading immediately before the result table."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    return (
        f"[{timestamp}] [{PLUGIN_LOG_NAME}] [INFO] "
        f"Prefix Cache Results of task [{analysis.get('run_id', 'unknown')}]:"
    )


def _format_run_summary_table(analysis: dict) -> str:
    """Render the five overall Prefix Cache metrics as a two-column table."""
    actual = analysis.get("actual")
    actual_hit_rate = actual.get("global_hit_rate") if isinstance(actual, dict) else None
    rows = [
        ("Overall Target Hit Rate", _format_hit_rate(analysis.get("requested_target_hit_rate"))),
        ("Overall Theoretical Hit Rate", _format_hit_rate(analysis.get("theoretical_hit_rate"))),
        ("Overall Actual Hit Rate", _format_hit_rate(actual_hit_rate)),
        (
            "Theory vs Actual Difference",
            _format_hit_rate_difference(analysis.get("theory_actual_absolute_difference_pp")),
        ),
        (
            "Theory vs Target Difference",
            _format_hit_rate_difference(analysis.get("target_absolute_difference_pp")),
        ),
    ]
    headers = ("Prefix Cache Metric", "Value")
    first_width = max(len(headers[0]), *(len(name) for name, _ in rows))
    second_width = max(len(headers[1]), *(len(value) for _, value in rows))

    def border(left: str, middle: str, right: str, fill: str) -> str:
        return left + fill * (first_width + 2) + middle + fill * (second_width + 2) + right

    def row(first: str, second: str) -> str:
        return f"│ {first:<{first_width}} │ {second:<{second_width}} │"

    lines = [
        border("╒", "╤", "╕", "═"),
        row(*headers),
        border("╞", "╪", "╡", "═"),
    ]
    for index, values in enumerate(rows):
        lines.append(row(*values))
        if index != len(rows) - 1:
            lines.append(border("├", "┼", "┤", "─"))
    lines.append(border("╘", "╧", "╛", "═"))
    return "\n".join(lines)


class PromptProgress:
    """Render prompt-generation progress to a text stream without touching stdout."""

    def __init__(self, stream: TextIO | None = None, width: int = 30):
        self.stream = stream if stream is not None else sys.stderr
        self.width = max(1, width)
        self._active = False
        self._completed = False

    def update(self, completed: int, total: int) -> None:
        if total < 1:
            return
        completed = min(max(0, completed), total)
        filled = self.width * completed // total
        percent = 100 * completed // total
        bar = "#" * filled + "-" * (self.width - filled)
        end = "\n" if completed == total else "\r"
        self.stream.write(f"\rGenerate prompts [{bar}] {completed}/{total} {percent:3d}%{end}")
        self.stream.flush()
        self._active = completed < total
        self._completed = completed == total

    def close(self) -> None:
        """Terminate an unfinished progress line before another stderr message."""
        if self._active and not self._completed:
            self.stream.write("\n")
            self.stream.flush()
        self._active = False


def build_parser() -> argparse.ArgumentParser:
    """构建 Prefix Cache 数据准备、运行与分析子命令。"""
    parser = argparse.ArgumentParser(prog="ais-bench-prefix-cache")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "inspect", "run"):
        # prepare 生成请求工件；inspect 只预览不发请求（共用 --scenario）。
        item = sub.add_parser(name)
        item.add_argument("--scenario", required=True, type=Path)
    prepare = sub.choices["prepare"]
    prepare.add_argument("--mode", required=True, choices=("text", "mm"))
    prepare.add_argument("--overwrite", action="store_true")
    run = sub.choices["run"]
    run.add_argument("--config", type=Path)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--manifest", required=True, type=Path)
    analyze.add_argument("--baseline", required=True, type=Path)
    analyze.add_argument("--after", required=True, type=Path)
    report = sub.add_parser("report")
    report.add_argument("--manifest", required=True, type=Path)
    return parser


def _resolve_log_file(
    command: str,
    scenario_path: Path | None = None,
    manifest_path: Path | None = None,
    execution_timestamp: str | None = None,
    scenario_mode: str | None = None,
) -> Path | None:
    """Resolve a per-command log under the run output directory's log/ layer.

    prepare / inspect 从 scenario 解析 output_dir 与 run_id（prepare 优先复用
    最近一次成功 inspect Manifest 的时间戳目录）；
    validate 从 manifest 的 run_id 与 effective_config.run.output_dir 解析。

    Falls back to console-only logging when the config cannot be loaded
    or the output directory is not writable (the real error surfaces in
    the normal command flow).
    """
    if command in {"validate", "analyze", "report"}:
        if manifest_path is None:
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_id = manifest["run_id"]
            output_dir = Path(manifest["effective_config"]["run"]["output_dir"])
            log_file = output_dir / "log" / f"{run_id}.validate.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            return log_file
        except (KeyError, TypeError, OSError, json.JSONDecodeError):
            return None
    if scenario_path is None:
        return None
    try:
        load_mode = scenario_mode or ("mm" if command == "run" else "text")
        scenario = load_scenario(scenario_path, mode=load_mode)
        if execution_timestamp is not None:
            scenario = with_execution_timestamp(scenario, execution_timestamp)
        log_file = scenario.output_dir / "log" / f"{scenario.run_id}.{command}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        return log_file
    except (PrefixCacheError, OSError):
        return None


def _reusable_execution_timestamp(scenario: Scenario, *, inspected_only: bool) -> str | None:
    """Return the newest reusable timestamp discovered from a matching Manifest."""
    statuses = {"inspected"} if inspected_only else {"prepared"}
    found = find_latest_execution_manifest(scenario, statuses)
    return found[0] if found is not None else None


def _read_manifest(path: Path | str) -> dict:
    manifest_path = Path(path).resolve()
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrefixCacheError(f"cannot read Manifest {manifest_path}: {exc}") from exc


def _manifest_mode(manifest: dict) -> str:
    mode = manifest.get("benchmark_mode", "text")
    if mode not in {"text", "mm"}:
        raise PrefixCacheError(f"unknown Manifest benchmark_mode: {mode!r}")
    return mode


def _scenario_work_dir(scenario: Scenario) -> Path:
    configured = Path(scenario.section("aisbench")["work_dir"])
    if configured.is_absolute():
        return configured.resolve()
    return (scenario.source_path.parent / configured).resolve()


def _persist_inspect_manifest(
    scenario_path: Path,
    result: dict,
    log_file: Path | None,
    timestamp: str,
) -> Path:
    """Persist inspect output as the run's lightweight Manifest."""
    base_scenario = load_scenario(scenario_path)
    scenario = with_execution_timestamp(base_scenario, timestamp)
    effective = copy.deepcopy(scenario.to_effective_dict())
    configured_api_key = bool(effective["service"].pop("api_key", ""))
    effective["service"]["api_key_configured"] = configured_api_key
    summary = copy.deepcopy(result)
    if log_file is not None:
        summary["log"] = str(log_file)
    manifest = {
        "schema_version": "1.0",
        "plugin_version": __version__,
        "benchmark_mode": "text",
        "status": "inspected",
        "run_id": scenario.run_id,
        "scenario_path": str(base_scenario.source_path),
        "scenario_sha256": sha256_file(base_scenario.source_path),
        "effective_config": effective,
        "inspect": {
            "timestamp": timestamp,
            "base_run_id": base_scenario.run_id,
            "base_output_dir": str(base_scenario.output_dir),
            "sends_requests": False,
            "summary": summary,
        },
    }
    path = artifact_paths(scenario.output_dir, scenario.run_id).manifest
    write_json(path, manifest, overwrite=False)
    logger.info("[cli] inspect persisted manifest=%s", path)
    return path


def _install_logger(log_file: Path | None) -> None:
    """安装插件自身的 logger handler，不依赖 ais_bench 的 AISLogger。

    解析到 .log 文件时只写入文件；无法解析日志路径时回退为仅控制台输出，
    真实错误仍由正常命令流程抛出。
    """
    plugin_logger = logging.getLogger(PLUGIN_LOG_NAME)
    for existing in plugin_logger.handlers:
        existing.close()
    plugin_logger.handlers.clear()
    plugin_logger.propagate = False
    plugin_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_NORMAL_FORMAT)
    if log_file is not None:
        # 先清空上一轮同名插件日志，再以 append 模式打开。AISBench 正式任务的
        # stdout/stderr 由其 LocalRunner 写入 work_dir/logs/infer，不与本文件混写。
        log_file.write_text("", encoding="utf-8")
        handler: logging.Handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        handler.setFormatter(formatter)
        plugin_logger.addHandler(handler)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        plugin_logger.addHandler(handler)


def _close_logger() -> None:
    """Close command-scoped handlers so repeated CLI calls do not leak files."""
    plugin_logger = logging.getLogger(PLUGIN_LOG_NAME)
    for handler in plugin_logger.handlers:
        handler.close()
    plugin_logger.handlers.clear()


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口：分发到对应子命令并统一处理错误码。"""
    args = build_parser().parse_args(argv)
    # inspect 每次生成新时间戳目录；prepare/run 从 Manifest 发现可复用目录。
    execution_timestamp: str | None = None
    reused_execution_timestamp = False
    if args.command == "inspect":
        execution_timestamp = new_execution_timestamp()
    elif args.command == "prepare":
        try:
            scenario = load_scenario(args.scenario, mode=args.mode)
            validate_scenario_mode(scenario, args.mode)
            reusable = (
                _reusable_execution_timestamp(scenario, inspected_only=True)
                if args.mode == "text"
                else None
            )
        except PrefixCacheError:
            reusable = None
        if reusable is not None:
            execution_timestamp = reusable
            reused_execution_timestamp = True
        else:
            execution_timestamp = new_execution_timestamp()
    elif args.command == "run":
        try:
            reusable = _reusable_execution_timestamp(
                load_scenario(args.scenario, mode="mm"),
                inspected_only=False,
            )
        except PrefixCacheError:
            reusable = None
        if reusable is not None:
            execution_timestamp = reusable
            reused_execution_timestamp = True
    log_file = _resolve_log_file(
        args.command,
        scenario_path=getattr(args, "scenario", None),
        manifest_path=getattr(args, "manifest", None),
        execution_timestamp=execution_timestamp,
        scenario_mode=getattr(args, "mode", None),
    )
    # 安装插件自身的 logger；日志路径可用时只写文件，不回显到 CLI 终端。
    _install_logger(log_file)
    logger.info("[cli] command=%s args=%s log_file=%s reused_execution_timestamp=%s", args.command, vars(args), log_file, reused_execution_timestamp)
    progress = PromptProgress() if args.command in {"prepare", "run"} else None
    try:
        if args.command == "prepare":
            scenario = load_scenario(args.scenario, mode=args.mode)
            validate_scenario_mode(scenario, args.mode)
            logger.info(
                "[cli] prepare mode=%s scenario=%s overwrite=%s",
                args.mode,
                args.scenario,
                args.overwrite,
            )
            if args.mode == "text":
                paths = prepare_scenario(
                    args.scenario,
                    overwrite=args.overwrite,
                    progress=progress.update,
                    execution_timestamp=execution_timestamp,
                )
                result = {key: str(value) for key, value in paths.__dict__.items()}
            else:
                manifest_path = prepare_multimodal_scenario(
                    args.scenario,
                    overwrite=args.overwrite,
                    execution_timestamp=execution_timestamp,
                )
                result = validate_multimodal_manifest(manifest_path) | {
                    "manifest": str(manifest_path)
                }
            if log_file is not None:
                result["log"] = str(log_file)
            logger.info("[cli] prepare returned result=%s", result)
            print(json.dumps(result, ensure_ascii=False))
        elif args.command == "validate":
            logger.info("[cli] validate manifest=%s", args.manifest)
            manifest = _read_manifest(args.manifest)
            mode = _manifest_mode(manifest)
            result = (
                validate_multimodal_manifest(args.manifest)
                if mode == "mm"
                else validate_artifacts(args.manifest)
            )
            logger.info("[cli] validate mode=%s returned result=%s", mode, result)
            print(json.dumps(result, ensure_ascii=False))
        elif args.command == "inspect":
            logger.info("[cli] inspect scenario=%s", args.scenario)
            result = inspect_scenario(args.scenario)
            if log_file is not None:
                result["log"] = str(log_file)
            manifest_path = _persist_inspect_manifest(
                args.scenario,
                result,
                log_file,
                execution_timestamp,
            )
            result["manifest"] = str(manifest_path)
            logger.info("[cli] inspect_scenario returned result=%s", json.dumps(result, ensure_ascii=False))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "run":
            logger.info("[cli] run scenario=%s config=%s", args.scenario, args.config)
            if execution_timestamp is None:
                raise PrefixCacheError(
                    "no prepared Manifest found; run prepare --mode text or --mode mm first"
                )
            base_scenario = load_scenario(args.scenario, mode="mm")
            scenario = with_execution_timestamp(base_scenario, execution_timestamp)
            manifest_path = artifact_paths(scenario.output_dir, scenario.run_id).manifest
            manifest = _read_manifest(manifest_path)
            mode = _manifest_mode(manifest)
            if mode == "text":
                result = run_scenario(
                    args.scenario,
                    args.config,
                    execution_timestamp=execution_timestamp,
                    progress=progress.update,
                )
                logger.info("[cli] run mode=%s returned status=%s", mode, result.get("status"))
                print(_format_run_summary_heading(result))
                print(_format_run_summary_table(result))
                print(f"[INFO] Detailed analysis is available at: {result.get('analysis', 'N/A')}")
            else:
                if args.config is not None:
                    raise PrefixCacheError("run --config is only supported for text mode")
                service = scenario.section("service")
                tokenizer = scenario.section("tokenizer")
                dataset_cfg = scenario.section("aisbench")["dataset"]
                model_cfg = scenario.section("aisbench")["model"]
                result = run_multimodal_benchmark(
                    manifest_path=manifest_path,
                    scenarios=manifest["datasets"],
                    inference_url=service["inference_url"],
                    model=service["model"],
                    tokenizer_path=tokenizer["path"],
                    work_dir=_scenario_work_dir(scenario),
                    batch_size=model_cfg["batch_size"],
                    extra_args=scenario.section("aisbench")["extra_args"],
                    api_key=service.get("api_key", ""),
                    stream=model_cfg["stream"],
                    retry=model_cfg["retry"],
                    generation_kwargs=model_cfg["generation_kwargs"],
                    pred_role=dataset_cfg["pred_role"],
                    dataset_abbr=dataset_cfg["abbr"],
                    model_abbr=model_cfg["abbr"],
                    model_attr=model_cfg["attr"],
                    model_max_out_len=model_cfg["max_out_len"],
                )
                logger.info("[cli] run mode=%s returned result=%s", mode, result)
                print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "analyze":
            logger.info(
                "[cli] analyze manifest=%s baseline=%s after=%s",
                args.manifest,
                args.baseline,
                args.after,
            )
            result = analyze_snapshots(args.manifest, args.baseline, args.after)
            logger.info("[cli] analyze_snapshots returned status=%s", result.get("status"))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "report":
            manifest = _read_manifest(args.manifest)
            mode = _manifest_mode(manifest)
            effective = manifest.get("effective_config", {})
            aisbench = effective.get("aisbench", {})
            configured_work_dir = Path(aisbench.get("work_dir", "outputs/default"))
            if not configured_work_dir.is_absolute():
                scenario_source = Path(manifest.get("scenario_path", args.manifest)).resolve()
                configured_work_dir = (scenario_source.parent / configured_work_dir).resolve()
            if mode == "mm":
                configured_abbr = aisbench.get("dataset", {}).get("abbr")
                selected = list(manifest["datasets"])
                result = {}
                for name in selected:
                    scenario_abbr = (
                        f"{configured_abbr}-{name}"
                        if configured_abbr and len(selected) > 1
                        else configured_abbr or name
                    )
                    result[name] = report_performance(
                        configured_work_dir,
                        scenario_abbr,
                    )
            else:
                dataset_cfg = aisbench.get("dataset", {})
                dataset_abbr = dataset_cfg.get("abbr") or manifest["run_id"]
                result = report_performance(configured_work_dir, dataset_abbr)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except PrefixCacheError as exc:
        # 业务错误统一以 ERROR 输出并返回退出码 2，便于脚本判断。
        if progress is not None:
            progress.close()
        logger.warning("[cli] PrefixCacheError: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        _close_logger()


def console_main() -> None:
    """控制台入口：把 main 的返回码作为进程退出码。"""
    raise SystemExit(main())


if __name__ == "__main__":
    console_main()
