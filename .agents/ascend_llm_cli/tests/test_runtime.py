import tempfile
import time
import unittest
import threading
from deploy_eval_v2.task_runtime import Runtime
from deploy_eval_v2.fake_runtime import FakeExecutor
from deploy_eval_v2.protocol import ProtocolError
from pathlib import Path
from deploy_eval_v2.json_io import load_json


def command(kind='deploy', version=0, request='r', **fields):
    return dict(protocol_version='1.0', task_id='t', kind=kind, request_id=request,
                expected_version=version, **fields)


def finish(runtime):
    deadline = time.monotonic()+2
    while time.monotonic() < deadline:
        with runtime.lock:
            if not runtime.active:
                return
        time.sleep(.001)
    raise AssertionError('executor did not finish')


class RuntimeTests(unittest.TestCase):
    def test_cancel_accepted_before_success_boundary_runs_cleanup(self):
        entered=threading.Event();complete=threading.Event();cleaning=threading.Event();cleaned=threading.Event()
        class Executor:
            def __call__(self,*args):
                entered.set()
                if not complete.wait(2):
                    raise RuntimeError('test synchronization')
                return 'deployment.ready','READY',{'service_id':'owned'}
            def cancel_idle(self,operation):
                cleaning.set()
                if not cleaned.wait(2):
                    raise RuntimeError('test cleanup synchronization')
                return 'operation.cancelled','CANCELLED',{'cleanup_status':'VERIFIED'}
        with tempfile.TemporaryDirectory() as root:
            runtime=Runtime(root,'t',Executor())
            accepted=runtime.submit(command(plan_ref='frozen'))
            self.assertTrue(entered.wait(1))
            runtime.submit(command('cancel',runtime.version,'cancel',scope='task',operation_id=accepted['operation_id']))
            complete.set()
            self.assertTrue(cleaning.wait(1))
            # Cleanup is outside runtime lock; still active until evidence returns.
            with runtime.lock:
                self.assertEqual(runtime.active,accepted['operation_id'])
                self.assertFalse(any(e['type']=='deployment.ready' for e in runtime.journal.events))
            cleaned.set();finish(runtime)
            self.assertEqual(runtime.state,'CANCELLED')
            self.assertEqual(runtime.journal.events[-1]['type'],'operation.cancelled')

    def test_unhandled_executor_error_preserves_evidence_and_requires_recovery(self):
        def fail(*args):
            raise RuntimeError('stage adapter broke')
        with tempfile.TemporaryDirectory() as root:
            runtime=Runtime(root,'t',fail)
            runtime.submit(command(plan_ref='frozen'));finish(runtime)
            self.assertEqual(runtime.state,'RECOVERY_REQUIRED')
            reference=runtime.journal.events[-1]['data']['evidence_ref']
            self.assertIn('stage adapter broke',(Path(root)/reference).read_text())
    def test_unknown_event_does_not_end_operation_or_replace_service(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=Runtime(root,'t',FakeExecutor(.1))
            accepted=runtime.submit(command(plan_ref='frozen'))
            with runtime.lock:
                runtime._emit('future.informational',accepted['operation_id'],'RELEASED',{'service_id':'foreign'})
                self.assertEqual(runtime.active,accepted['operation_id'])
                self.assertIsNone(runtime.service_id)
            finish(runtime)
            self.assertEqual(runtime.state,'READY')

    def test_operation_index_is_rebuilt_from_journal(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=Runtime(root,'t',FakeExecutor())
            accepted=runtime.submit(command(plan_ref='frozen'))
            finish(runtime)
            path=Path(root)/'results'/'operation-index.json'
            operation=accepted['operation_id']
            self.assertEqual(load_json(path)['operations'][operation]['event_type'],'deployment.ready')
            path.unlink()
            restored=Runtime(root,'t',FakeExecutor())
            self.assertEqual(load_json(path)['operations'][operation]['status'],'READY')
            self.assertEqual(restored.operations[accepted['operation_id']]['terminal_seq'],2)
    def test_replay_and_lifecycle(self):
        with tempfile.TemporaryDirectory() as root:
            fake = FakeExecutor()
            runtime = Runtime(root, 't', fake)
            request = command(plan_ref='frozen')
            first = runtime.submit(request)
            finish(runtime)
            self.assertEqual(runtime.submit(request), first)
            restored = Runtime(root, 't', fake)
            self.assertEqual(restored.submit(request), first)
            restored.submit(command('eval', restored.version, 'e', service_id='fake-service', case_id='c'))
            finish(restored)
            restored.submit(command('release', restored.version, 'x', service_id='fake-service'))
            finish(restored)
            with self.assertRaises(ProtocolError):
                restored.submit(command('eval', restored.version, 'late', service_id='fake-service', case_id='c'))
            self.assertEqual(fake.calls, ['deploy', 'eval', 'release'])

    def test_conflict_and_unknown_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = Runtime(root, 't', FakeExecutor(.1))
            request = command(plan_ref='frozen')
            runtime.submit(request)
            with self.assertRaisesRegex(ProtocolError, 'REQUEST_CONFLICT'):
                runtime.submit(dict(request, plan_ref='changed'))
            restored = Runtime(root, 't', FakeExecutor())
            self.assertEqual(restored.state, 'RECOVERY_REQUIRED')
            finish(runtime)

    def test_task_cancel_is_terminal(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = Runtime(root, 't', FakeExecutor(1))
            accepted = runtime.submit(command(plan_ref='frozen'))
            runtime.submit(command('cancel', runtime.version, 'cancel', scope='task', operation_id=accepted['operation_id']))
            finish(runtime)
            self.assertTrue(runtime.task_cancelled)
            self.assertEqual(runtime.state, 'CANCELLED')
