import tempfile
import unittest
from deploy_eval_v2.api import API
from deploy_eval_v2.task_runtime import Runtime
from deploy_eval_v2.fake_runtime import FakeExecutor
from test_runtime import command, finish


class APITests(unittest.TestCase):
    def test_short_stage_returns_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            api = API(Runtime(root, 't', FakeExecutor()))
            result = api.submit(command(plan_ref='p', wait_ms=1000))
            self.assertEqual(result['status'], 'EVENTS')
            self.assertEqual(result['events'][-1]['type'], 'deployment.ready')
            from pathlib import Path
            from deploy_eval_v2.json_io import load_json
            index=load_json(Path(root)/api.status()['result_index_ref'])
            self.assertEqual(index['operations'][result['operation_id']]['event_type'],'deployment.ready')

    def test_long_stage_resume(self):
        with tempfile.TemporaryDirectory() as root:
            api = API(Runtime(root, 't', FakeExecutor(.05)))
            result = api.submit(command(plan_ref='p', wait_ms=0))
            self.assertEqual(result['status'], 'RUNNING')
            resumed = api.runtime.journal.watch(result['next_cursor'], 1000)
            self.assertEqual(resumed['events'][-1]['type'], 'deployment.ready')
            finish(api.runtime)
