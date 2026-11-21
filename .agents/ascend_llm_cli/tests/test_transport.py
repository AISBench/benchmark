import sys
import unittest
import threading
import time
from deploy_eval_v2.ssh_transport import exchange, ssh_argv
from deploy_eval_v2.protocol import ProtocolError


class TransportTests(unittest.TestCase):
    def run_child(self, program):
        return exchange([sys.executable, '-c', program], {'task_id': 't'}, timeout=3)

    def test_valid_and_stderr_flood(self):
        result = self.run_child("import sys; sys.stderr.write('x'*1000000); print('{\"protocol_version\":\"1.0\",\"task_id\":\"t\",\"status\":\"WAIT_TIMEOUT\"}')")
        self.assertEqual(result['status'], 'WAIT_TIMEOUT')

    def test_banner_and_oversize(self):
        for program in ("print('banner')", "print('x'*1000000)"):
            with self.assertRaisesRegex(ProtocolError, 'TRANSPORT_PROTOCOL_ERROR'):
                self.run_child(program)

    def test_unknown_outcome(self):
        with self.assertRaisesRegex(ProtocolError, 'UNKNOWN_OUTCOME'):
            self.run_child('raise SystemExit(255)')

    def test_no_pty_and_host_validation(self):
        self.assertIn('-T', ssh_argv('user@host', ['python3', '/a b/client.py']))
        with self.assertRaises(ProtocolError):
            ssh_argv('-oProxyCommand=bad', [])

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        for data in ('{"protocol_version":"1.0","task_id":"other","task_id":"t","status":"EVENTS"}',
                     '{"protocol_version":"1.0","task_id":"t","status":"EVENTS","value":NaN}'):
            with self.assertRaisesRegex(ProtocolError,'TRANSPORT_PROTOCOL_ERROR'):
                self.run_child('print('+repr(data)+')')

    def test_cancel_waiting_transport_has_bounded_duration(self):
        flag=threading.Event()
        timer=threading.Timer(.15,flag.set)
        timer.start()
        start=time.monotonic()
        try:
            with self.assertRaisesRegex(ProtocolError,'UNKNOWN_OUTCOME'):
                exchange([sys.executable,'-c','import time;time.sleep(30)'],{'task_id':'t'},timeout=5,cancelled=flag)
            self.assertLess(time.monotonic()-start,2)
        finally:
            timer.join()
