#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Map AISBench source paths to unit-test paths and run scoped coverage.

The mapping mirrors ``tests/run_tests.py`` (``tests/UT/<sub>`` covers
``ais_bench/benchmark/<sub>``).  A scoped run never proves the repository-wide
coverage gate; every coverage result therefore carries ``scope="subset"``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = 1
SOURCE_ROOT = Path("ais_bench/benchmark")
TEST_ROOT = Path("tests/UT")
DEFAULT_PACKAGE = Path("ais_bench")


class MapError(RuntimeError):
    pass


def test_dir_for_subdir(sub: str) -> Path:
    return TEST_ROOT / sub


def posix(path: Path | str) -> str:
    """Repository-relative paths use forward slashes regardless of host OS."""
    return Path(path).as_posix()


def source_dir_for_subdir(sub: str, repo_root: Path) -> Path:
    candidate = SOURCE_ROOT / sub
    if (repo_root / candidate).exists():
        return candidate
    return SOURCE_ROOT


def suggest_for_source(path: str, repo_root: Path) -> dict:
    """Return suggested test path(s) and the coverage source dir for a source file."""
    pure = Path(path)
    try:
        rel = pure.relative_to(SOURCE_ROOT)
    except ValueError:
        return {
            "source": path,
            "supported": False,
            "reason": f"不在 {SOURCE_ROOT} 下；映射规则未覆盖该路径，请人工指定测试与覆盖率范围",
        }
    parts = rel.parts
    sub = parts[0] if len(parts) > 1 else ""
    if not sub:
        return {
            "source": path,
            "supported": False,
            "reason": "位于包根目录，需人工指定测试目录",
        }
    test_dir = test_dir_for_subdir(sub)
    source_dir = source_dir_for_subdir(sub, repo_root)
    if pure.name == "__init__.py" or len(parts) == 1:
        test_file = test_dir / f"test_{sub}.py"
    else:
        stem = pure.stem
        prefix = "test_" if not stem.startswith("test_") else ""
        test_file = test_dir / f"{prefix}{stem}.py"
    return {
        "source": path,
        "supported": True,
        "subdir": sub,
        "test_dir": posix(test_dir),
        "suggested_test_file": posix(test_file),
        "suggested_test_file_exists": (repo_root / test_file).exists(),
        "coverage_source_dir": posix(source_dir),
        "test_dir_exists": (repo_root / test_dir).is_dir(),
    }


def suggest(changed: list[str], repo_root: Path) -> dict:
    rows = [suggest_for_source(path, repo_root) for path in changed]
    test_dirs = sorted({row["test_dir"] for row in rows if row.get("supported")})
    source_dirs = sorted({row["coverage_source_dir"] for row in rows if row.get("supported")})
    missing_tests = [row["source"] for row in rows if row.get("supported") and not row["suggested_test_file_exists"]]
    return {
        "schema": SCHEMA,
        "changed": changed,
        "rows": rows,
        "test_dirs": test_dirs,
        "coverage_source_dirs": source_dirs,
        "sources_without_suggested_test": missing_tests,
        "unsupported": [row["source"] for row in rows if not row.get("supported")],
    }


def map_test_dir(test_dir: str, repo_root: Path) -> dict:
    pure = Path(test_dir)
    try:
        rel = pure.relative_to(TEST_ROOT)
    except ValueError:
        raise MapError(f"{test_dir} 不在 {TEST_ROOT} 下")
    if not rel.parts:
        return {
            "schema": SCHEMA,
            "test_dir": test_dir,
            "coverage_source_dirs": [posix(SOURCE_ROOT)],
            "note": "仅 tests/UT，按 run_tests.py 规则计量整个 ais_bench/benchmark",
        }
    sub = rel.parts[0]
    source_dir = source_dir_for_subdir(sub, repo_root)
    return {
        "schema": SCHEMA,
        "test_dir": test_dir,
        "subdir": sub,
        "coverage_source_dirs": [posix(source_dir)],
        "resolved": posix(source_dir),
        "test_dir_exists": (repo_root / pure).is_dir(),
    }


def _run_coverage(test_dir: str, source_dirs: list[str], output: str, repo_root: Path, timeout: int) -> dict:
    runner = repo_root / "tests" / "run_tests.py"
    if not runner.is_file():
        raise MapError("未找到 tests/run_tests.py；请在仓库根目录运行，或先确认覆盖率入口存在")
    cmd = [
        sys.executable,
        str(runner),
        test_dir,
        "--output",
        output,
    ]
    for source_dir in source_dirs:
        cmd.extend(["--source-dirs", source_dir])
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    missing_dep = re.search(r"ModuleNotFoundError|No module named '([^']+)'", combined)
    if missing_dep:
        return {
            "status": "UNKNOWN",
            "scope": "subset",
            "reason": "缺少运行依赖，未自动安装：" + missing_dep.group(0),
            "command": cmd,
            "exit": proc.returncode,
        }
    coverage_json = Path(output) / "coverage.json"
    payload: dict = {
        "scope": "subset",
        "command": cmd,
        "exit": proc.returncode,
    }
    if coverage_json.is_file():
        data = json.loads(coverage_json.read_text(encoding="utf-8"))
        totals = data.get("totals", {})
        payload["totals"] = totals
        payload["coverage_status"] = (
            "PASS" if proc.returncode == 0 and not data.get("files") is None else "UNKNOWN"
        )
    else:
        payload["coverage_status"] = "UNKNOWN"
        payload["reason"] = "未生成 coverage.json"
    payload["status"] = "UNKNOWN"
    payload["reason"] = payload.get("reason") or "子集覆盖率不能证明全量门禁，全量由 pr_test 在 CI 给出"
    payload["tail"] = combined.strip().splitlines()[-20:]
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ut_map.py", description="AISBench 源码↔UT 映射与子集覆盖率")
    sub = parser.add_subparsers(dest="command", required=True)

    sug = sub.add_parser("suggest", help="按变更文件建议测试路径与覆盖率范围")
    sug.add_argument("--changed", nargs="+", required=True)
    sug.add_argument("--repo-root", default=".")

    mapping = sub.add_parser("map", help="按测试目录反查覆盖率源码目录")
    mapping.add_argument("--test-dir", required=True)
    mapping.add_argument("--repo-root", default=".")

    cov = sub.add_parser("coverage", help="运行子集测试并读取覆盖率（不安装依赖）")
    cov.add_argument("--test-dir", required=True)
    cov.add_argument("--source-dirs", nargs="*", default=None)
    cov.add_argument("--output", default="test_reports/devflow-subset")
    cov.add_argument("--repo-root", default=".")
    cov.add_argument("--timeout", type=int, default=1800)
    cov.add_argument("--dry-run", action="store_true")

    for sub_parser in (sug, mapping, cov):
        sub_parser.add_argument("--format", default="json", choices=["json", "text"])
    return parser


def emit(payload: dict, fmt: str, exit_code: int = 0) -> int:
    if fmt == "text":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return exit_code


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    try:
        if args.command == "suggest":
            return emit(suggest(args.changed, repo_root), args.format)
        if args.command == "map":
            return emit(map_test_dir(args.test_dir, repo_root), args.format)
        if args.command == "coverage":
            source_dirs = args.source_dirs
            if not source_dirs:
                source_dirs = map_test_dir(args.test_dir, repo_root)["coverage_source_dirs"]
            if args.dry_run:
                runner = repo_root / "tests" / "run_tests.py"
                cmd = [sys.executable, str(runner), args.test_dir, "--output", args.output]
                for source_dir in source_dirs:
                    cmd.extend(["--source-dirs", source_dir])
                return emit(
                    {
                        "schema": SCHEMA,
                        "dry_run": True,
                        "command": cmd,
                        "scope": "subset",
                        "note": "未执行；本地不安装依赖",
                    },
                    args.format,
                )
            return emit(
                _run_coverage(args.test_dir, source_dirs, args.output, repo_root, args.timeout),
                args.format,
            )
    except MapError as exc:
        return emit({"schema": SCHEMA, "error": str(exc), "exit": 2}, args.format, 2)
    return 2


if __name__ == "__main__":
    sys.exit(main())
