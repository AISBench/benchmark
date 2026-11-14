import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from deploy_eval_v2.event_journal import Journal
from deploy_eval_v2.task_runtime import Runtime
from deploy_eval_v2.fake_runtime import FakeExecutor
from deploy_eval_v2.protocol import ProtocolError
from test_runtime import command, finish


class RecoveryTests(unittest.TestCase):
    def test_partial_tail_repair_preserves_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            journal = Journal(folder, 't')
            journal.append('future.ready', 'op', 1, 'READY', {})
            with journal.path.open('ab') as stream:
                stream.write(b'{"seq":2')
            recovered = Journal(folder, 't', repair_tail=True)
            self.assertEqual(len(recovered.events), 1)
            diagnostics=[json.loads(line) for line in recovered.path.read_text().splitlines() if json.loads(line).get('record_type')=='diagnostic']
            self.assertEqual(bytes.fromhex(diagnostics[0]['message']),b'{"seq":2')
            self.assertEqual(len(Journal(folder, 't').events), 1)

    def test_complete_corrupt_line_is_never_repaired(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'task.jsonl'
            original = b'{bad}\n{partial'
            path.write_bytes(original)
            with self.assertRaises(ProtocolError):
                Journal(folder, 't', repair_tail=True)
            self.assertEqual(path.read_bytes(), original)

    def test_missing_index_and_wrong_snapshot_rebuilt(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = Runtime(folder, 't', FakeExecutor())
            req = command(plan_ref='p')
            accepted = runtime.submit(req)
            finish(runtime)
            root = Path(folder)
            self.assertFalse((root/'commands').exists())
            (root/'state.json').write_text('{"state":"NEW"}')
            recovered = Runtime(folder, 't', FakeExecutor())
            self.assertEqual(recovered.submit(req), accepted)
            self.assertEqual(json.loads((root/'state.json').read_text())['state'], 'READY')
            self.assertFalse((root/'commands').exists())
            self.assertIn('r',recovered.commands)

    def test_snapshot_failure_never_starts_side_effect(self):
        with tempfile.TemporaryDirectory() as folder:
            executor = FakeExecutor()
            runtime = Runtime(folder, 't', executor)
            with patch('deploy_eval_v2.task_runtime.atomic_json', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    runtime.submit(command(plan_ref='p'))
            self.assertEqual(executor.calls, [])
            recovered = Runtime(folder, 't', executor)
            self.assertEqual(recovered.state, 'RECOVERY_REQUIRED')
            self.assertEqual(len(recovered.commands), 1)
            boundary=recovered.journal.events[-1]
            self.assertEqual(boundary['type'],'runtime.recovery_required')
            response=recovered.journal.watch(1,0)
            self.assertEqual(response['events'][-1]['type'],'runtime.recovery_required')
            again=Runtime(folder,'t',executor)
            self.assertEqual(len(again.journal.events),len(recovered.journal.events))
            self.assertEqual(executor.calls,[])
