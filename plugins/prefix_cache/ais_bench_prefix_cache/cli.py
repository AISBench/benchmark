from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ais_bench.benchmark.utils.logging.logger import AISLogger

from .artifacts import validate_artifacts
from .errors import PrefixCacheError
from .pipeline import inspect_scenario, prepare_scenario
from .runtime import analyze_snapshots, run_scenario
from .scenario import load_scenario

logger = logging.getLogger(__name__)

# Parent logger name shared by all module loggers (ais_bench_prefix_cache.*);
# AISLogger installs the console + file handlers on it, following the same
# style as ais_bench/benchmark/datasets/hle.py.
PLUGIN_LOG_NAME = "ais_bench_prefix_cache"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ais-bench-prefix-cache")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "inspect"):
        item = sub.add_parser(name)
        item.add_argument("--scenario", required=True, type=Path)
    prepare = sub.choices["prepare"]
    prepare.add_argument("--overwrite", action="store_true")
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("--scenario", required=True, type=Path)
    run.add_argument("--config", type=Path)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--manifest", required=True, type=Path)
    analyze.add_argument("--baseline", required=True, type=Path)
    analyze.add_argument("--after", required=True, type=Path)
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_file = _resolve_log_file(args.command, getattr(args, "scenario", None))
    AISLogger(name=PLUGIN_LOG_NAME, log_file=str(log_file) if log_file else None, file_mode="w")
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
        elif args.command == "run":
            logger.info("[cli] run scenario=%s config=%s", args.scenario, args.config)
            result = run_scenario(args.scenario, args.config)
            logger.info("[cli] run_scenario returned result=%s", json.dumps(result, ensure_ascii=False))
            for warning in result.get("warnings", []):
                print(f"WARNING: {warning}", file=sys.stderr)
        else:
            logger.info("[cli] analyze manifest=%s baseline=%s after=%s", args.manifest, args.baseline, args.after)
            result = analyze_snapshots(args.manifest, args.baseline, args.after)
            logger.info("[cli] analyze_snapshots returned result=%s", json.dumps(result, ensure_ascii=False))
            for warning in result.get("warnings", []):
                print(f"WARNING: {warning}", file=sys.stderr)
        return 0
    except PrefixCacheError as exc:
        logger.warning("[cli] PrefixCacheError: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def console_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    console_main()
