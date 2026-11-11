"""Check unavailable Bash handling and failure-preserving test reports."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import bash_support


class TestInfrastructureTests(unittest.TestCase):
    def test_bash_missing_stub_timeout_and_permission_denied(self):
        replies = [Mock(returncode=1, stdout=''), Mock(returncode=0, stdout=''),
                   subprocess.TimeoutExpired('bash', 5), PermissionError('denied')]
        with patch.object(bash_support.os, 'name', 'posix'), patch.object(bash_support.shutil, 'which', return_value='/stub/bash'):
            for reply in replies:
                with self.subTest(reply=reply), patch.object(bash_support.subprocess, 'run') as run:
                    if isinstance(reply, Exception):
                        run.side_effect = reply
                    else:
                        run.return_value = reply
                    with self.assertRaisesRegex(unittest.SkipTest, 'Bash unavailable'):
                        bash_support.require_bash()
            with patch.object(bash_support.shutil, 'which', return_value=None):
                with self.assertRaisesRegex(unittest.SkipTest, 'no executable'):
                    bash_support.require_bash()

    def test_real_bash_version_is_accepted(self):
        with patch.object(bash_support.os, 'name', 'posix'), patch.object(bash_support.shutil, 'which', return_value='/bin/bash'), patch.object(bash_support.subprocess, 'run', return_value=Mock(returncode=0, stdout='5.2.15(1)-release\n')) as run:
            self.assertEqual(bash_support.require_bash(), '/bin/bash')
            self.assertEqual(run.call_args.kwargs['timeout'], 5)

    def test_test_report_preserves_failure_exit_and_skip_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'test_sample.py').write_text('import unittest\nclass Sample(unittest.TestCase):\n def test_ok(self): pass\n def test_bad(self): self.fail("expected")\n @unittest.skip("platform reason")\n def test_skip(self): pass\n')
            report = root/'report.json'
            runner = Path(__file__).resolve().parents[1]/'run_unit_tests.py'
            result = subprocess.run([sys.executable, str(runner), '--start-directory', str(root), '--output', str(report)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 1)
            data = json.loads(report.read_text())
            self.assertEqual((data['tests_run'], data['passed'], data['failures']), (3, 1, 1))
            self.assertEqual(data['skipped'][0]['reason'], 'platform reason')
            self.assertFalse(data['successful'])
            self.assertTrue(data['source']['tested_source_sha256'])
