from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .artifacts import validate_artifacts
from .errors import PrefixCacheError
from .pipeline import inspect_scenario, prepare_scenario
from .scenario import load_scenario

# Parent logger name shared by all module loggers (ais_bench_prefix_cache.*).
PLUGIN_LOG_NAME = "ais_bench_prefix_cache"

LOG_NORMAL_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

# 显式挂在 PLUGIN_LOG_NAME 之下（不用 __name__）：python -m 运行时 __name__ 会变成
# "__main__"，导致日志绕过插件 logger 直接传播到 root。
logger = logging.getLogger(f"{PLUGIN_LOG_NAME}.cli")


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


def _resolve_log_file(command: str, scenario_path: Path | None) -> Path | None:
    """Per-command log file in the same directory as the run_id artifacts.

    Falls back to console-only logging when the scenario cannot be loaded
    or the output directory is not writable (the real error surfaces in
    the normal command flow).
    """
    if scenario_path is None:
        return None
    try:
        scenario = load_scenario(scenario_path)
        log_file = scenario.output_dir / f"{scenario.run_id}.{command}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        return log_file
    except (PrefixCacheError, OSError):
        return None


def _install_logger(log_file: Path | None) -> None:
    """安装插件自身的 logger handler，不依赖 ais_bench 的 AISLogger。

    解析到 .log 文件时日志只写入文件、不在终端打印；否则回退为仅控制台输出，
    真实的错误信息在正常命令流程中抛出。
    """
    plugin_logger = logging.getLogger(PLUGIN_LOG_NAME)
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
    log_file = _resolve_log_file(args.command, getattr(args, "scenario", None))
    # 安装插件自身的 logger（日志只缓存到 .log 文件，不在终端打印）。
    _install_logger(log_file)
    logger.info("[cli] command=%s args=%s log_file=%s", args.command, vars(args), log_file)
    try:
        if args.command == "prepare":
            logger.info("[cli] prepare scenario=%s overwrite=%s", args.scenario, args.overwrite)
            paths = prepare_scenario(args.scenario, overwrite=args.overwrite)
            result = {key: str(value) for key, value in paths.__dict__.items()}
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
            logger.info("[cli] inspect_scenario returned result=%s", json.dumps(result, ensure_ascii=False))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except PrefixCacheError as exc:
        # 业务错误统一以 ERROR 输出并返回退出码 2，便于脚本判断。
        logger.warning("[cli] PrefixCacheError: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def console_main() -> None:
    """控制台入口：把 main 的返回码作为进程退出码。"""
    raise SystemExit(main())


if __name__ == "__main__":
    console_main()
