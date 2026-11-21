import json
from pathlib import Path
import tempfile
import threading
import unittest
from deploy_eval_v2.event_journal import Journal
from deploy_eval_v2.protocol import encode, summary_bytes


class EventBudgetTests(unittest.TestCase):
    def test_large_detail_externalized_without_losing_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            journal = Journal(folder, 't')
            detail = '中文\\"'*1000
            journal.append('future.failure', 'op', 1, 'FAILED',
                           {'error_code': 'FAILED', 'cleanup_status': 'UNVERIFIED', 'detail': detail})
            result = journal.watch(0, 0)
            event = result['events'][0]
            self.assertTrue(event['data']['details_externalized'])
            stored = json.loads((Path(folder)/event['data']['artifact_ref']).read_text(encoding='utf-8'))
            self.assertEqual(stored['data']['detail'], detail)
            self.assertEqual(result['next_cursor'], 1)
            self.assertLessEqual(len(encode(result)), 8192)

    def test_progress_does_not_end_wait(self):
        with tempfile.TemporaryDirectory() as folder:
            journal = Journal(folder, 't')
            journal.append('phase.changed', 'op', 1, 'RUNNING', {'phase':'accepted'})
            journal.append('phase.changed', 'op', 1, 'RUNNING', {'phase': 'health'})
            timeout = journal.watch(0, 0)
            self.assertEqual(timeout['status'], 'WAIT_TIMEOUT')
            self.assertEqual(timeout['next_cursor'], 2)
            self.assertEqual(timeout['events'], [])
            timer = threading.Timer(.05, lambda: journal.append('future.ready', 'op', 2, 'READY', {}))
            timer.start()
            try:
                self.assertEqual(journal.watch(0, 1000)['events'][0]['seq'], 3)
            finally:
                timer.join()

    def test_summary_pagination_never_skips_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            journal = Journal(folder, 't')
            for i in range(20):
                journal.append('future.failure', str(i), i, 'FAILED', {'summary': '中'*70})
                journal.append('phase.changed', str(i), i, 'RUNNING', {'phase':'progress'})
            seen, cursor = [], 0
            while cursor < len(journal.events):
                response = journal.watch(cursor, 0, metadata={'operation_id': 'op', 'task_version': 20})
                self.assertLessEqual(len(encode(response)), 8192)
                self.assertLessEqual(summary_bytes(response), 1024)
                seen.extend(e['seq'] for e in response['events'])
                self.assertGreater(response['next_cursor'], cursor)
                cursor = response['next_cursor']
            self.assertEqual(seen, list(range(1, 40, 2)))
