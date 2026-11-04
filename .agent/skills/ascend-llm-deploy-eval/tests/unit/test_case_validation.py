import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from validate_case import validate

class ValidationTests(unittest.TestCase):
    def test_standard_perf_records_and_logs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            output=root/'outputs/run/performances/model/dataset_details.jsonl';output.parent.mkdir(parents=True)
            log=root/'outputs/run/logs/infer/model/dataset.out';log.parent.mkdir(parents=True);log.write_text('native worker log')
            (root/'execution.json').write_text('{"returncode":0}')
            rows=[dict(id=i,success=True,error_info='',input_tokens=100+i,output_tokens=32-i) for i in range(2)]
            def save():output.write_text('\n'.join(json.dumps(row) for row in rows))
            save();case=dict(id='c1',formal_requests=2,output_tokens=32)
            result=validate(root,case)
            self.assertEqual(result['formal']['input_max'],101)
            self.assertIn(str(log),result['logs'])
            case['input_tokens']=100
            with self.assertRaisesRegex(ValueError,'expectation'):validate(root,case)
            case.pop('input_tokens');rows[0]['output_tokens']=33;save()
            with self.assertRaisesRegex(ValueError,'token'):validate(root,case)
            rows[0]['output_tokens']=32;rows[0]['id']=1;save()
            with self.assertRaisesRegex(ValueError,'Duplicate'):validate(root,case)
            rows[0]['id']=0;rows[0]['success']=False;save()
            with self.assertRaisesRegex(ValueError,'failed'):validate(root,case)
            rows[0]['success']=True;save();log.unlink()
            with self.assertRaisesRegex(ValueError,'logs'):validate(root,case)
            (root/'execution.json').write_text('{"returncode":2}')
            with self.assertRaisesRegex(ValueError,'exit'):validate(root,case)
