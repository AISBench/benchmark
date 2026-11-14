"""Run the aisbench-devflow unit tests and write a JSON report.

Usage:
    python tests/run_unit_tests.py [--output report.json]

Exit code is 0 only when every test passed and none was skipped unexpectedly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.passed = 0

    def addSuccess(self, test) -> None:  # noqa: N802 - unittest API
        super().addSuccess(test)
        self.passed += 1


def source_identity() -> dict:
    digest = hashlib.sha256()
    for folder in ("scripts", "tests"):
        for path in sorted((ROOT / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in (".py", ".json"):
                continue
            digest.update(path.relative_to(ROOT).as_posix().encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).digest())
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain", "--", "."],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git optional
        commit, dirty = None, None
    return {"sha256": digest.hexdigest(), "commit": commit, "dirty": dirty}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run aisbench-devflow unit tests")
    parser.add_argument("--output", default=None, help="JSON report path")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover
            pass

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult)
    result = runner.run(suite)

    report = {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "counts": {
            "run": result.testsRun,
            "passed": getattr(result, "passed", 0),
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
        },
        "failed_tests": [str(case) for case, _ in result.failures],
        "errored_tests": [str(case) for case, _ in result.errors],
        "skipped_tests": [str(case) for case, _ in result.skipped],
        "source": source_identity(),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == "__main__":
    sys.exit(main())
