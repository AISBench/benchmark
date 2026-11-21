import json
from pathlib import Path
import tempfile
import unittest
from deploy_eval_v2.native_postflight import audit


class Tokenizer:
    def encode(self,text,**kwargs): return list(text)


class NativePostflightTests(unittest.TestCase):
    def test_actual_prefix_and_length_deviation_are_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'outputs/run/performances/service/dataset_details.jsonl';path.parent.mkdir(parents=True)
            path.write_text('\n'.join(json.dumps({'input':text}) for text in ('abc1','abc22')))
            report=audit(folder,dict(input_tokens=5,formal_requests=2,shared_prefix_ratio=.5),'unused',tokenizer=Tokenizer())
            self.assertEqual(report['status'],'MEASURED')
            self.assertEqual(report['common_prefix_tokens'],3)
            self.assertFalse(report['exact_input_lengths'])
            self.assertEqual(report['input_tokens']['mean'],4.5)

    def test_missing_text_and_excessive_row_are_bounded_unverified(self):
        for content in ('{}','x'*(1024*1024+1)):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'outputs/run/performances/service/dataset_details.jsonl';path.parent.mkdir(parents=True);path.write_text(content)
                report=audit(folder,dict(input_tokens=5,formal_requests=1),'unused',tokenizer=Tokenizer())
                self.assertEqual(report['status'],'UNVERIFIED')
                self.assertNotIn('common_prefix_tokens',report)
