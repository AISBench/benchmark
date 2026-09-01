from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
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
from .pipeline import inspect_scenario, prepare_scenario
from .runtime import analyze_snapshots, run_scenario
from .scenario import Scenario, load_scenario, new_execution_timestamp, with_execution_timestamp

# Parent logger name shared by all module loggers (ais_bench_prefix_cache.*).
PLUGIN_LOG_NAME = "ais_bench_prefix_cache"

LOG_NORMAL_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

# 显式挂在 PLUGIN_LOG_NAME 之下（不用 __name__）：python -m 运行时 __name__ 会变成
# "__main__"，导致日志绕过插件 logger 直接传播到 root。
logger = logging.getLogger(f"{PLUGIN_LOG_NAME}.cli")


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
    prepare.add_argument("--overwrite", action="store_true")
    run = sub.choices["run"]
    run.add_argument("--config", type=Path)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--manifest", required=True, type=Path)
    analyze.add_argument("--baseline", required=True, type=Path)
    analyze.add_argument("--after", required=True, type=Path)
    return parser


def _resolve_log_file(
    command: str,
    scenario_path: Path | None = None,
    manifest_path: Path | None = None,
    execution_timestamp: str | None = None,
) -> Path | None:
    """Resolve a per-command log under the run output directory's log/ layer.

    prepare / inspect 从 scenario 解析 output_dir 与 run_id（prepare 优先复用
    最近一次成功 inspect Manifest 的时间戳目录）；
    validate 从 manifest 的 run_id 与 effective_config.run.output_dir 解析。

    Falls back to console-only logging when the config cannot be loaded
    or the output directory is not writable (the real error surfaces in
    the normal command flow).
    """
    if command in {"validate", "analyze"}:
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
        scenario = load_scenario(scenario_path)
        if execution_timestamp is not None:
            scenario = with_execution_timestamp(scenario, execution_timestamp)
        log_file = scenario.output_dir / "log" / f"{scenario.run_id}.{command}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        return log_file
    except (PrefixCacheError, OSError):
        return None


def _reusable_execution_timestamp(scenario: Scenario, *, inspected_only: bool) -> str | None:
    """Return the newest reusable timestamp discovered from a matching Manifest."""
    statuses = {"inspected"} if inspected_only else {"inspected", "prepared"}
    found = find_latest_execution_manifest(scenario, statuses)
    return found[0] if found is not None else None


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
        # 先清空上一轮同名日志，再以 append 模式打开。run 的 AISBench 子进程
        # 也会追加写入同一文件，O_APPEND 可避免父子进程使用独立文件偏移时覆盖。
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
    elif args.command in {"prepare", "run"}:
        try:
            reusable = _reusable_execution_timestamp(
                load_scenario(args.scenario),
                inspected_only=args.command == "prepare",
            )
        except PrefixCacheError:
            reusable = None
        if reusable is not None:
            execution_timestamp = reusable
            reused_execution_timestamp = True
        else:
            execution_timestamp = new_execution_timestamp()
    log_file = _resolve_log_file(
        args.command,
        scenario_path=getattr(args, "scenario", None),
        manifest_path=getattr(args, "manifest", None),
        execution_timestamp=execution_timestamp,
    )
    # 安装插件自身的 logger；日志路径可用时只写文件，不回显到 CLI 终端。
    _install_logger(log_file)
    logger.info("[cli] command=%s args=%s log_file=%s reused_execution_timestamp=%s", args.command, vars(args), log_file, reused_execution_timestamp)
    progress = PromptProgress() if args.command in {"prepare", "run"} else None
    try:
        if args.command == "prepare":
            logger.info("[cli] prepare scenario=%s overwrite=%s", args.scenario, args.overwrite)
            paths = prepare_scenario(
                args.scenario,
                overwrite=args.overwrite,
                progress=progress.update,
                execution_timestamp=execution_timestamp,
            )
            result = {key: str(value) for key, value in paths.__dict__.items()}
            if log_file is not None:
                result["log"] = str(log_file)
            logger.info("[cli] prepare_scenario returned paths=%s", result)
            print(json.dumps(result, ensure_ascii=False))
        elif args.command == "validate":
            logger.info("[cli] validate manifest=%s", args.manifest)
            result = validate_artifacts(args.manifest)
            logger.info("[cli] validate_artifacts returned result=%s", result)
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
            result = run_scenario(
                args.scenario,
                args.config,
                execution_timestamp=execution_timestamp,
                progress=progress.update,
            )
            logger.info("[cli] run_scenario returned status=%s", result.get("status"))
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
