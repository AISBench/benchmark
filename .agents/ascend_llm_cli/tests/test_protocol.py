import tempfile
import threading
import unittest
from deploy_eval_v2.event_journal import Journal
from deploy_eval_v2.protocol import encode, ProtocolError


class JournalTests(unittest.TestCase):
    def test_pagination_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            journal = Journal(root, 'task')
            for i in range(23):
                journal.append('future.completed', str(i), i, 'SUCCEEDED', {'artifact_ref': 'results/x'})
            recovered = Journal(root, 'task')
            cursor, seen = 0, []
            while cursor < 23:
                response = recovered.watch(cursor, 0)
                self.assertLessEqual(len(encode(response)), 8192)
                self.assertLessEqual(len(response['events']), 10)
                seen.extend(e['seq'] for e in response['events'])
                cursor = response['next_cursor']
            self.assertEqual(seen, list(range(1, 24)))

    def test_wait_notification_and_timeout(self):
        with tempfile.TemporaryDirectory() as root:
            journal = Journal(root, 'task')
            self.assertEqual(journal.watch(0, 0)['status'], 'WAIT_TIMEOUT')
            worker = threading.Thread(target=lambda: journal.append('future.ready', 'op', 1, 'READY', {}))
            worker.start()
            self.assertEqual(journal.watch(0, 1000)['next_cursor'], 1)
            worker.join()
            with self.assertRaisesRegex(ProtocolError, 'CURSOR_INVALID'):
                journal.watch(2, 0)

    def test_unicode_budget_and_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            journal = Journal(root, 'task')
            for _ in range(4):
                journal.append('future.failure', 'op', 1, 'FAILED', {'detail': '中文\\"'*300})
            response = journal.watch(0, 0)
            self.assertLessEqual(len(encode(response)), 8192)
            self.assertTrue(response['has_more'])
            with journal.path.open('ab') as stream:
                stream.write(b'{')
            with self.assertRaisesRegex(ProtocolError, 'RECOVERY_REQUIRED'):
                Journal(root, 'task')
