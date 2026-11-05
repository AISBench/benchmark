import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from group_run import run, validate, eligible
from group_worker import terminal
from task_events import emit, read_events


def plan():
    return {'group_id': '20261101-901', 'tasks': [
        {'id': name, 'host': 'user@node.example', 'run_dir': '/work/'+name,
         'plan_sha256': 'a'*64, 'depends_on': ['a', 'b'] if name == 'c' else []}
        for name in ['a', 'b', 'c']]}


class FakeWorkers:
    def __init__(self, fail=False, block=False, disconnect=False):
        self.started = []; self.running = set(); self.rounds = 0
        self.fail, self.block, self.disconnect = fail, block, disconnect

    def __call__(self, task, action, **kwargs):
        name = task['id']
        resources = ['node-a', 'node-b'] if name == 'c' else ['node-'+name]
        base = {'resources': resources, 'state': 'PENDING', 'events': {'offset': 0, 'events': []}}
        if action == 'start':
            self.started.append(name); self.running.add(name)
            return {'state': 'RUNNING'}
        if name in self.running:
            if self.disconnect and name == 'a' and self.rounds == 1:
                raise TimeoutError('SSH unavailable')
            if self.rounds > 0:
                return {**base, 'state': 'BLOCKED' if self.block and name == 'a' else 'DONE',
                        'cleanup_verified': not (self.block and name == 'a'),
                        'result': 'FAILED' if self.fail and name == 'a' else 'PASSED'}
            return {**base, 'state': 'RUNNING'}
        return base

    def sleep(self, seconds):
        self.rounds += 1
        if self.rounds == 1:
            assert self.started == ['a', 'b'], self.started
        if self.rounds > 10:
            raise AssertionError('Scheduler did not finish')


class GroupTests(unittest.TestCase):
    def execute(self, workers):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'group-plan.json').write_text(json.dumps(plan()))
            result = run(root, workers, workers.sleep)
            events = read_events(root)['events']
            self.assertTrue(events)
            before = list(workers.started)
            self.assertEqual(run(root, workers, workers.sleep), result)
            self.assertEqual(workers.started, before)
            return result

    def test_parallel_single_then_dual(self):
        workers = FakeWorkers()
        self.assertEqual(self.execute(workers)['status'], 'PASSED')
        self.assertEqual(workers.started, ['a', 'b', 'c'])

    def test_failed_measurement_with_verified_cleanup_allows_successor(self):
        workers = FakeWorkers(fail=True)
        self.assertEqual(self.execute(workers)['status'], 'FAILED')
        self.assertEqual(workers.started, ['a', 'b', 'c'])

    def test_failed_cleanup_blocks_successor(self):
        workers = FakeWorkers(block=True)
        result = self.execute(workers)
        self.assertEqual(result['tasks']['c']['state'], 'BLOCKED')
        self.assertEqual(workers.started, ['a', 'b'])

    def test_disconnect_does_not_replay(self):
        workers = FakeWorkers(disconnect=True)
        self.assertEqual(self.execute(workers)['status'], 'PASSED')
        self.assertEqual(workers.started, ['a', 'b', 'c'])

    def test_same_host_never_parallel(self):
        states = {'a': {'state':'RUNNING','resources':['host']}, 'b':{'state':'PENDING','resources':['host']}}
        self.assertFalse(eligible({'id':'b'}, states))

    def test_cycle_rejected(self):
        value=plan();value['tasks'][0]['depends_on']=['c']
        with self.assertRaisesRegex(ValueError, 'cycle'): validate(value)

    def test_terminal_requires_all_node_release_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            spec={'sub_tasks':[{'id':'case1'}]}
            resolved={'deployments':{'dep':{'groups':{'a':{},'b':{}}}}}
            (root/'sub_tasks/case1').mkdir(parents=True)
            (root/'sub_tasks/case1/result.json').write_text(json.dumps({'status':'PASSED'}))
            (root/'summary.json').write_text(json.dumps({'status':'PASSED','sub_tasks':{'case1':{'status':'PASSED'}},'cleanup':{'status':'PASSED'}}))
            (root/'release-results.json').write_text(json.dumps({'dep':{'a':{'status':'PASSED'}}}))
            self.assertEqual(terminal(root,spec,resolved)['state'],'BLOCKED')
            (root/'release-results.json').write_text(json.dumps({'dep':{'a':{'status':'PASSED'},'b':{'status':'PASSED'}}}))
            self.assertTrue(terminal(root,spec,resolved)['cleanup_verified'])

    def test_event_cursor_waits_for_complete_line(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);emit(root,'testing','doing',subcase_id='s1')
            first=read_events(root)
            with (root/'logs/events.jsonl').open('ab') as stream: stream.write(b'{"incomplete":')
            second=read_events(root,first['offset'])
            self.assertEqual(second,{'offset':first['offset'],'events':[]})

    def test_event_io_failure_does_not_interrupt_cleanup_path(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch('task_events.os.open', side_effect=OSError('disk unavailable')), patch('sys.stderr'):
                emit(Path(temp),'client_cleanup','doing',subcase_id='s1')


if __name__ == '__main__': unittest.main()
