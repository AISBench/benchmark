import json
from pathlib import Path
import tempfile
import threading
import unittest
from deploy_eval_v2.native_results import validate_native_result,LINE_LIMIT
from deploy_eval_v2.protocol import ProtocolError
from deploy_eval_v2.process_guard import Cancelled


class NativeResultTests(unittest.TestCase):
    def test_boolean_exit_code_and_duplicate_expectations_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,_=self.fixture(folder,[dict(id=1,success=True)])
            (root/'execution.json').write_text('{"returncode":false}')
            with self.assertRaises(ProtocolError):
                validate_native_result(root,{'formal_requests':1})
            (root/'execution.json').write_text('{"returncode":0}')
            (root/'native-expectations.json').write_text('[["service","dataset"],["service","dataset"]]')
            with self.assertRaisesRegex(ProtocolError,'duplicate native'):
                validate_native_result(root,{'formal_requests':1})

    def fixture(self,folder,rows):
        root=Path(folder)
        (root/'execution.json').write_text('{"returncode":0}')
        (root/'native-expectations.json').write_text('[["service","dataset"]]')
        details=root/'outputs/run/performances/service/dataset_details.jsonl'
        details.parent.mkdir(parents=True)
        details.write_text('\n'.join(json.dumps(row) for row in rows),encoding='utf-8')
        log=root/'outputs/run/logs/worker.log';log.parent.mkdir(parents=True);log.write_text('worker evidence')
        return root,details

    def test_complete_native_groups_preserve_performance_uncertainty(self):
        with tempfile.TemporaryDirectory() as folder:
            root,_=self.fixture(folder,[dict(id=i,success=True) for i in range(1000)])
            result=validate_native_result(root,{'formal_requests':1000})
            self.assertEqual(result['request_groups'][0]['requests'],1000)
            self.assertEqual(result['performance_validity'],'UNVERIFIED')

    def test_duplicate_failed_and_wrong_request_count_rejected(self):
        for rows,count in [([dict(id=1,success=True)]*2,2),([dict(id=1,success=False)],1),
                           ([dict(id=1,success=True)],2),([dict(id=1,success=True),dict(id=2,success=True)],1)]:
            with tempfile.TemporaryDirectory() as folder:
                root,_=self.fixture(folder,rows)
                with self.assertRaises(ProtocolError):
                    validate_native_result(root,{'formal_requests':count})
                self.assertFalse((root/'validated.json').exists())

    def test_oversized_row_and_cancel_do_not_report_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            root,details=self.fixture(folder,[])
            with details.open('wb') as stream:
                for _ in range(17):
                    stream.write(b'x'*65536)
            with self.assertRaisesRegex(ProtocolError,'1 MiB'):
                validate_native_result(root,{'formal_requests':1})
            flag=threading.Event();flag.set()
            with self.assertRaises(Cancelled):
                validate_native_result(root,{'formal_requests':1},cancelled=flag)
