import sys
import unittest
from unittest.mock import Mock
from deploy_eval_v2.process_io import bounded_command
from deploy_eval_v2.protocol import ProtocolError


class SystemdIOTests(unittest.TestCase):
    def test_actual_process_bounded_stderr_and_stdout(self):
        code, output, diagnostic = bounded_command([sys.executable, '-c',
            "import sys; sys.stderr.write('x'*1000000); print('LoadState=loaded')"])
        self.assertEqual(code, 0)
        self.assertIn(b'LoadState=loaded', output)
        self.assertEqual(len(diagnostic), 2048)
        with self.assertRaisesRegex(ProtocolError, 'exceeds budget'):
            bounded_command([sys.executable, '-c', "print('x'*1000000)"], limit=1024)

    def test_actual_process_timeout(self):
        with self.assertRaisesRegex(ProtocolError, 'UNKNOWN_OUTCOME'):
            bounded_command([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=.1)
