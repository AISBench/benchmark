from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import validate_artifacts
from .errors import PrefixCacheError
from .pipeline import inspect_scenario, prepare_scenario
from .runtime import analyze_snapshots, run_scenario


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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            paths = prepare_scenario(args.scenario, overwrite=args.overwrite)
            print(json.dumps({key: str(value) for key, value in paths.__dict__.items()}, ensure_ascii=False))
        elif args.command == "validate":
            print(json.dumps(validate_artifacts(args.manifest), ensure_ascii=False))
        elif args.command == "inspect":
            print(json.dumps(inspect_scenario(args.scenario), ensure_ascii=False, indent=2))
        elif args.command == "run":
            result = run_scenario(args.scenario, args.config)
            for warning in result.get("warnings", []):
                print(f"WARNING: {warning}", file=sys.stderr)
        else:
            result = analyze_snapshots(args.manifest, args.baseline, args.after)
            for warning in result.get("warnings", []):
                print(f"WARNING: {warning}", file=sys.stderr)
        return 0
    except PrefixCacheError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def console_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    console_main()
