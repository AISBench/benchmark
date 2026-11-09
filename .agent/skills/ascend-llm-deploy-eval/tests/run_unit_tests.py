"""Run skill unit tests and write counts, skipped test IDs and source identity."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = 0

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed += 1


def source_identity():
    digest = hashlib.sha256()
    files = [p for folder in ('scripts', 'tests', 'schemas', 'examples')
             for p in (ROOT/folder).rglob('*') if p.is_file()
             and p.suffix in ('.py', '.json') and '__pycache__' not in p.parts]
    files.append(ROOT/'requirements.txt')
    for path in sorted(files):
        digest.update(path.relative_to(ROOT).as_posix().encode()+b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                         text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', '.'],
                                             text=True, stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {'git_commit': commit, 'working_tree_dirty': dirty,
            'tested_source_sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-bash', action='store_true', help='Fail CI instead of silently skipping shell coverage')
    parser.add_argument('--start-directory', type=Path, default=ROOT/'tests/unit')
    args = parser.parse_args()
    if args.require_bash:
        sys.path.insert(0, str(ROOT/'tests/unit'))
        from bash_support import require_bash
        try:
            require_bash()
        except unittest.SkipTest as error:
            parser.error(str(error))
    identity = source_identity()
    sys.path.insert(0, str(ROOT/'scripts'))
    suite = unittest.defaultTestLoader.discover(str(args.start_directory.resolve()))
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    report = {'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': platform.python_version(), 'system': platform.system(),
              'platform': platform.platform(), 'source': identity,
              'tests_run': result.testsRun, 'passed': result.passed,
              'failures': len(result.failures), 'errors': len(result.errors),
              'expected_failures': len(result.expectedFailures),
              'unexpected_successes': len(result.unexpectedSuccesses),
              'skipped': [{'test': test.id(), 'reason': reason} for test, reason in result.skipped],
              'successful': result.wasSuccessful()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
