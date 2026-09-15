from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import PrefixCacheError
from .multimodal import (
    SCENARIO_NAMES,
    prepare_multimodal_datasets,
    report_performance,
    run_multimodal_benchmark,
    validate_multimodal_manifest,
)


def _default_mmmu_parquet_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "MMMU"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ais-bench-prefix-cache-mm")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="build the two required 1319-row datasets")
    prepare.add_argument("--tokenizer", required=True)
    prepare.add_argument(
        "--gsm8k",
        type=Path,
        default=Path(r"C:\cjs\datasets\grade-school-math\grade_school_math\data\test.jsonl"),
    )
    prepare.add_argument(
        "--mmmu-parquet-dir",
        type=Path,
        default=_default_mmmu_parquet_dir(),
        help="MMMU directory containing Parquet shards; exact-size images are selected from bytes fields",
    )
    prepare.add_argument("--output-dir", type=Path, default=Path("outputs/prefix_cache_multimodal"))
    prepare.add_argument("--request-count", type=int, default=1319)
    prepare.add_argument("--text-tokens", type=int, default=30)
    prepare.add_argument("--output-tokens", type=int, default=256)
    prepare.add_argument("--trust-remote-code", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")

    validate = sub.add_parser("validate", help="validate row counts, tokens, images, and checksums")
    validate.add_argument("--manifest", required=True, type=Path)

    run = sub.add_parser("run", help="run AISBench perf and print TTFT/TPOT/ITL")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--scenario", choices=["all", *SCENARIO_NAMES], default="all")
    run.add_argument("--inference-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--tokenizer", required=True)
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--work-dir", type=Path, default=Path("outputs/prefix_cache_multimodal_perf"))
    run.add_argument("--extra-arg", action="append", default=[])

    report = sub.add_parser("report", help="extract acceptance metrics from an AISBench work directory")
    report.add_argument("--work-dir", required=True, type=Path)
    report.add_argument("--dataset-abbr", choices=list(SCENARIO_NAMES))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare_multimodal_datasets(
                tokenizer_path=args.tokenizer,
                gsm8k_path=args.gsm8k,
                mmmu_parquet_dir=args.mmmu_parquet_dir,
                output_dir=args.output_dir,
                request_count=args.request_count,
                text_tokens=args.text_tokens,
                output_tokens=args.output_tokens,
                overwrite=args.overwrite,
                trust_remote_code=args.trust_remote_code,
            )
            result = validate_multimodal_manifest(manifest)
            print(json.dumps(result | {"manifest": str(manifest)}, ensure_ascii=False, indent=2))
        elif args.command == "validate":
            print(json.dumps(validate_multimodal_manifest(args.manifest), ensure_ascii=False, indent=2))
        elif args.command == "run":
            scenarios = SCENARIO_NAMES if args.scenario == "all" else (args.scenario,)
            result = run_multimodal_benchmark(
                manifest_path=args.manifest,
                scenarios=scenarios,
                inference_url=args.inference_url,
                model=args.model,
                tokenizer_path=args.tokenizer,
                work_dir=args.work_dir,
                batch_size=args.batch_size,
                extra_args=args.extra_arg,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(report_performance(args.work_dir, args.dataset_abbr), ensure_ascii=False, indent=2))
        return 0
    except (PrefixCacheError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def console_main() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    console_main()
