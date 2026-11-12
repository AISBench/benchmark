import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from deploy_eval_v2.cli import main
from deploy_eval_v2.protocol import ProtocolError, envelope


class Output:
    def __init__(self):
        self.buffer = io.BytesIO()


class CLITests(unittest.TestCase):
    def invoke(self, args, effect=None, response=None):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder)/'control.json'
            config.write_text(json.dumps({'tasks': {'t': {'host': 'example.invalid', 'root': '/work/t', 'allowed_root': '/work'}}}), encoding='utf-8')
            output = Output()
            with patch('deploy_eval_v2.cli.sys.stdout', output), patch('deploy_eval_v2.cli.exchange', side_effect=effect, return_value=response) as transport:
                code = main(['--config', str(config)]+args)
            raw = output.buffer.getvalue()
            self.assertLessEqual(len(raw), 8193)
            return code, json.loads(raw), transport

    def test_query_validation_precedes_ssh(self):
        for args in (['wait-event', '--task-ref', 't', '--after-seq', '-1'],
                     ['wait-event', '--task-ref', 't', '--after-seq', '0', '--wait-ms', '30001'],
                     ['read-result', '--task-ref', 't', '--artifact-ref', '../secret']):
            code, result, transport = self.invoke(args)
            self.assertEqual(code, 2)
            self.assertEqual(result['error']['code'], 'INVALID_INPUT')
            transport.assert_not_called()

    def test_disconnect_preserves_cursor(self):
        code, result, _ = self.invoke(['wait-event', '--task-ref', 't', '--after-seq', '42'], effect=ProtocolError('UNKNOWN_OUTCOME'))
        self.assertEqual(code, 3)
        self.assertEqual(result['task_id'], 't')
        self.assertEqual(result['next_cursor'], 42)

    def test_remote_error_exit_code(self):
        result = envelope('t', 0, 'ERROR')
        result['error'] = {'code': 'UNKNOWN_OUTCOME', 'message': 'unknown', 'retryable': False}
        code, _, _ = self.invoke(['get-status', '--task-ref', 't'], response=result)
        self.assertEqual(code, 3)

    def test_wait_timeout_is_successful_observation(self):
        code, result, _ = self.invoke(['wait-event', '--task-ref', 't', '--after-seq', '3'], response=envelope('t', 3, 'WAIT_TIMEOUT'))
        self.assertEqual(code, 0)
        self.assertEqual(result['next_cursor'], 3)

    def test_task_handoff_and_unified_log_are_readable_through_cli(self):
        for artifact in ('task.jsonl','service.json'):
            code,result,transport=self.invoke(['read-result','--task-ref','t','--artifact-ref',artifact],response=envelope('t',0))
            self.assertEqual(code,0)
            self.assertEqual(transport.call_args.args[1]['artifact_ref'],artifact)
