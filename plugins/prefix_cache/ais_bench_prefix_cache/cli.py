from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TextIO

from .artifacts import validate_artifacts
from .errors import PrefixCacheError
from .pipeline import inspect_scenario, prepare_scenario
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
    """构建命令行参数解析器：prepare / inspect / validate 三个离线子命令。"""
    parser = argparse.ArgumentParser(prog="ais-bench-prefix-cache")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "inspect"):
        # prepare 生成请求工件；inspect 只预览不发请求（共用 --scenario）。
        item = sub.add_parser(name)
        item.add_argument("--scenario", required=True, type=Path)
    prepare = sub.choices["prepare"]
    prepare.add_argument("--overwrite", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    return parser


def _resolve_log_file(
    command: str,
    scenario_path: Path | None = None,
    manifest_path: Path | None = None,
    execution_timestamp: str | None = None,
) -> Path | None:
    """Resolve a per-command log under the run output directory's log/ layer.

    prepare / inspect 从 scenario 解析 output_dir 与 run_id（prepare 优先复用
    最近一次成功 inspect 的时间戳目录，见 _reusable_inspect_timestamp）；
    validate 从 manifest 的 run_id 与 effective_config.run.output_dir 解析。

    Falls back to console-only logging when the config cannot be loaded
    or the output directory is not writable (the real error surfaces in
    the normal command flow).
    """
    if command == "validate":
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


def _inspect_pointer_path(output_dir: Path) -> Path:
    """每个基础 output_dir 一个指针文件，记录最近一次成功 inspect 的时间戳。"""
    return output_dir.with_name(f"{output_dir.name}.inspect.json")


def _reusable_inspect_timestamp(scenario: Scenario) -> str | None:
    """若存在与当前场景匹配的 inspect 指针且其时间戳目录还在，返回可复用的时间戳。

    指针记录的 run_id / output_dir 必须与当前场景一致，时间戳格式合法，
    且对应的时间戳目录仍然存在，否则视为不可复用（返回 None）。
    """
    pointer = _inspect_pointer_path(scenario.output_dir)
    try:
        record = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if record.get("schema_version") != "1.0":
        return None
    if record.get("run_id") != scenario.run_id or record.get("output_dir") != str(scenario.output_dir):
        return None
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str):
        return None
    try:
        stamped = with_execution_timestamp(scenario, timestamp)
    except PrefixCacheError:
        return None
    if not stamped.output_dir.is_dir():
        return None
    return timestamp


def _persist_inspect_pointer(scenario_path: Path, log_file: Path, timestamp: str) -> None:
    """写入 inspect 复用指针，供后续 prepare 复用同一时间戳目录。

    Best-effort：任何持久化失败只记日志，不影响 inspect 命令本身的结果。
    """
    try:
        base_scenario = load_scenario(scenario_path)
    except PrefixCacheError as exc:
        logger.warning("[cli] cannot load scenario for inspect pointer: %s", exc)
        return
    run_dir = log_file.parent.parent
    pointer = _inspect_pointer_path(base_scenario.output_dir)
    record = {
        "schema_version": "1.0",
        "timestamp": timestamp,
        "run_id": base_scenario.run_id,
        "output_dir": str(base_scenario.output_dir),
        "output_dir_with_timestamp": str(run_dir),
    }
    try:
        pointer.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info("[cli] inspect persisted pointer=%s record=%s", pointer, record)
    except OSError as exc:
        logger.warning("[cli] cannot persist inspect pointer: %s", exc)


def _install_logger(log_file: Path | None) -> None:
    """安装插件自身的 logger handler，不依赖 ais_bench 的 AISLogger。

    解析到 .log 文件时日志只写入文件、不在终端打印；否则回退为仅控制台输出，
    真实的错误信息在正常命令流程中抛出。
    """
    plugin_logger = logging.getLogger(PLUGIN_LOG_NAME)
    for existing in plugin_logger.handlers:
        existing.close()
    plugin_logger.handlers.clear()
    plugin_logger.propagate = False
    plugin_logger.setLevel(logging.INFO)
    if log_file is not None:
        handler: logging.Handler = logging.FileHandler(log_file, mode="w")
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_NORMAL_FORMAT))
    plugin_logger.addHandler(handler)


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口：分发到对应子命令并统一处理错误码。"""
    args = build_parser().parse_args(argv)
    # inspect 每次生成新时间戳目录；prepare 优先复用最近一次成功 inspect 的
    # 时间戳目录，让 inspect / prepare / validate 的 .log 与 .json 落在同一目录。
    execution_timestamp: str | None = None
    reused_inspect_timestamp = False
    if args.command == "inspect":
        execution_timestamp = new_execution_timestamp()
    elif args.command == "prepare":
        try:
            reusable = _reusable_inspect_timestamp(load_scenario(args.scenario))
        except PrefixCacheError:
            reusable = None
        if reusable is not None:
            execution_timestamp = reusable
            reused_inspect_timestamp = True
        else:
            execution_timestamp = new_execution_timestamp()
    log_file = _resolve_log_file(
        args.command,
        scenario_path=getattr(args, "scenario", None),
        manifest_path=getattr(args, "manifest", None),
        execution_timestamp=execution_timestamp,
    )
    # 安装插件自身的 logger（日志只缓存到 .log 文件，不在终端打印）。
    _install_logger(log_file)
    logger.info("[cli] command=%s args=%s log_file=%s reused_inspect_timestamp=%s", args.command, vars(args), log_file, reused_inspect_timestamp)
    progress = PromptProgress() if args.command == "prepare" else None
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
                # 写入复用指针，供后续 prepare/validate 复用同一时间戳目录。
                _persist_inspect_pointer(args.scenario, log_file, execution_timestamp)
            logger.info("[cli] inspect_scenario returned result=%s", json.dumps(result, ensure_ascii=False))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except PrefixCacheError as exc:
        # 业务错误统一以 ERROR 输出并返回退出码 2，便于脚本判断。
        if progress is not None:
            progress.close()
        logger.warning("[cli] PrefixCacheError: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def console_main() -> None:
    """控制台入口：把 main 的返回码作为进程退出码。"""
    raise SystemExit(main())


if __name__ == "__main__":
    console_main()
