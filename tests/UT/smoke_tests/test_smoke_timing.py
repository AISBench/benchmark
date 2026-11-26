"""Model-free regression tests for complete and cancelled CI evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


FRAMEWORK = Path(__file__).resolve().parents[3] / 'smoke_tests/framework'
spec = importlib.util.spec_from_file_location('smoke_timing_under_test', FRAMEWORK / 'smoke_timing.py')
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


class SmokeTimingTest(unittest.TestCase):
    def test_complete_suite_is_streamed_and_ranked_without_sensitive_result_details(self):
        with tempfile.TemporaryDirectory() as root:
            records = timing.CaseTimings(root, 2, fast=True)
            for name, elapsed in [('fast', 1.0), ('slow', 5.0)]:
                records.record({'case_name': name, 'rank_size': 3},
                               {'success': True, 'execution_time': elapsed, 'message': 'SECRET_EXAMPLE'})
            self.assertEqual(len(records.path.read_text(encoding='utf-8').splitlines()), 3)
            self.assertEqual(records.slowest(1)[0]['case_name'], 'slow')
            records.finish(6)
            self.assertTrue(timing.completed_successfully(records.path))
            self.assertNotIn('SECRET_EXAMPLE', records.path.read_text(encoding='utf-8'))

    def test_missing_cancelled_failed_incomplete_and_processing_errors_do_not_pass(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'missing.jsonl'
            self.assertFalse(timing.completed_successfully(path))
            for reason in ('cancelled', 'failed', 'incomplete', 'processing', 'missing-end'):
                records = timing.CaseTimings(root, 1)
                if reason != 'incomplete':
                    records.record({'case_name': reason}, {'success': reason != 'failed'})
                if reason == 'processing':
                    records.error(reason)
                if reason != 'missing-end':
                    records.finish(1, interrupted=reason == 'cancelled')
                self.assertFalse(timing.completed_successfully(records.path), reason)

    def test_new_suite_start_invalidates_stale_success(self):
        with tempfile.TemporaryDirectory() as root:
            first = timing.CaseTimings(root, 1)
            first.record({'case_name': 'one'}, {'success': True})
            first.finish(1)
            self.assertTrue(timing.completed_successfully(first.path))
            second = timing.CaseTimings(root, 1)
            self.assertFalse(timing.completed_successfully(second.path))
            second.record({'case_name': 'one'}, {'success': True})
            second.finish(1)
            self.assertTrue(timing.completed_successfully(second.path))
            with second.path.open('a', encoding='utf-8') as stream:
                stream.write('{partial')
            self.assertFalse(timing.completed_successfully(second.path))

    def test_unknown_nonfinite_duration_is_not_sorted_and_duplicate_cases_fail(self):
        with tempfile.TemporaryDirectory() as root:
            records = timing.CaseTimings(root, 2)
            for value in (float('nan'), -1):
                self.assertIsNone(records.record({'case_name': 'duplicate'},
                                                {'success': True, 'execution_time': value}))
            self.assertEqual(records.slowest(), [])
            records.finish(1)
            self.assertFalse(timing.completed_successfully(records.path))

    def test_out_of_order_or_boolean_counts_do_not_pass(self):
        start = dict(event='suite_start', suite_id='x', case_count=1)
        case = dict(event='case_end', suite_id='x', case_name='one', success=True)
        end = dict(event='suite_end', suite_id='x', success=True,
                   interrupted=False, expected=1, completed=1)
        for rows in ([start, end, case],
                     [start, case, dict(end, expected=True, completed=True)],
                     [start, case, end, end]):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as root:
                path = Path(root) / 'timing.jsonl'
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
                self.assertFalse(timing.completed_successfully(path))

    def test_unfinished_list_includes_queue_but_not_completed_futures(self):
        futures = {Mock(done=Mock(return_value=False)): {'case_name': str(i)} for i in range(12)}
        futures[Mock(done=Mock(return_value=True))] = {'case_name': 'done'}
        names = timing.unfinished_names(futures)
        self.assertIn('(+2)', names)
        self.assertNotIn('done', names)

    def test_history_creates_external_parent_and_is_reloadable_atomically(self):
        spec = importlib.util.spec_from_file_location('smoke_scheduler_under_test', FRAMEWORK / 'schedule_task.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'run_log': SimpleNamespace(log=Mock())}):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'persistent' / 'history.json'
            scheduler = module.TaskScheduler(str(path))
            scheduler.update_execution_time('long', 10)
            scheduler.update_execution_time('short', 1)
            scheduler.save_data()
            loaded = module.TaskScheduler(str(path))
            self.assertEqual([row['case_name'] for row in loaded.schedule_LPT(
                [{'case_name': 'short'}, {'case_name': 'long'}])], ['long', 'short'])
            self.assertEqual(json.loads(path.read_text())['long'], [10])
            self.assertEqual(list(path.parent.glob('smoke-history-*')), [])
            # Destruct while the owned temporary directory still exists.
            del loaded, scheduler


if __name__ == '__main__':
    unittest.main()
