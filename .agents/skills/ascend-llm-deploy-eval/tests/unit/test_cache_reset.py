"""No live service required: verify reset gating and per-instance evidence."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from cache_reset import reset_instances, require_idle

IDLE = 'vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting{engine="0"} 0\n'


class CacheResetTests(unittest.TestCase):
    def exercise(self, replies, metrics=IDLE):
        self.calls = []
        def http(port, path, method='GET'):
            self.calls.append((port, path, method))
            if path == '/metrics':
                return metrics
            reply = replies[port]
            if isinstance(reply, Exception):
                raise reply
            return reply
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            try:
                reset_instances(folder, [8100, 8200], http)
                error = None
            except Exception as exc:
                error = exc
            report = json.loads((folder/'cache-reset.json').read_text())
            self.evidence = {p.name:p.read_text() for p in folder.iterdir()}
        return report, error

    def test_all_instances_checked_before_reset(self):
        report, error = self.exercise({8100:'{"success":true}',8200:'{"success":true}'})
        self.assertIsNone(error)
        self.assertEqual(report['status'], 'PASSED')
        self.assertEqual([c[2] for c in self.calls], ['GET','GET','POST','POST'])

    def test_busy_blocks_all_resets(self):
        report, error = self.exercise({}, IDLE.replace('} 0', '} 1', 1))
        self.assertIsNotNone(error)
        self.assertEqual(report['status'], 'FAILED')
        self.assertFalse(any(c[2]=='POST' for c in self.calls))

    def test_missing_or_invalid_metrics_rejected(self):
        for metrics in ['', IDLE.replace('} 0', '} NaN', 1), IDLE.replace('} 0', '} -1', 1)]:
            with self.subTest(metrics=metrics), self.assertRaises(RuntimeError):
                require_idle(metrics)

    def test_one_false_still_records_other_instance(self):
        report, error = self.exercise({8100:'{"success":false}',8200:'{"success":true}'})
        self.assertIsNotNone(error)
        self.assertEqual([i['status'] for i in report['instances']], ['FAILED','PASSED'])
        self.assertEqual(len([c for c in self.calls if c[2]=='POST']), 2)

    def test_non_boolean_or_invalid_response_rejected(self):
        for body in ['{"success":"true"}', '{"success":1}', '', '[]']:
            with self.subTest(body=body):
                report, error = self.exercise({8100:body,8200:'{"success":true}'})
                self.assertIsNotNone(error)
                self.assertEqual(report['status'], 'FAILED')

    def test_http_error_body_preserved(self):
        response = HTTPError('http://localhost/reset_prefix_cache',404,'Not Found',{},io.BytesIO(b'no route'))
        report, error = self.exercise({8100:response,8200:'{"success":true}'})
        self.assertIsNotNone(error)
        self.assertEqual(report['instances'][0]['http_status'],404)
        self.assertEqual(self.evidence['reset-8100-http-error.txt'],'no route')

    def test_timeout_blocks_and_is_recorded(self):
        report, error = self.exercise({8100:TimeoutError('test'),8200:'{"success":true}'})
        self.assertIsNotNone(error)
        self.assertIn('TimeoutError', report['instances'][0]['error'])


if __name__ == '__main__':
    unittest.main()
