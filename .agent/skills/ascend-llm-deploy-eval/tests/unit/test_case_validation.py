import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from validate_case import validate

class ValidationTests(unittest.TestCase):
    def test_native_results_and_reject_invalid_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            output=root/'outputs/run/performances/model/details_details.jsonl'
            output.parent.mkdir(parents=True)
            rows=[dict(id=i, success=True, error_info='', input_tokens=1024, output_tokens=32) for i in range(2)]
            def save_rows(): output.write_text('\n'.join(json.dumps(r) for r in rows))
            save_rows()
            analysis=root/'prefix_20261104/result/run.analysis.json'
            analysis.parent.mkdir(parents=True)
            data={'status':'complete','runtime':{'aisbench_exit_code':0,'warmup':[{'group_id':0,'dp_rank':i,'success':True} for i in range(2)]},'actual':{'by_dp':{str(i):{'queries':100,'hits':50} for i in range(2)}}}
            def save_analysis(): analysis.write_text(json.dumps(data))
            save_analysis()
            case=dict(id='c1',prefill_dp=2,prefix_num=1,formal_requests=2,input_tokens=1024,output_tokens=32,prefix_ratio=.5)
            self.assertEqual(validate(root,case)['measured_prefix_ratio'],.5)
            rows[0]['output_tokens']=31; save_rows()
            with self.assertRaisesRegex(ValueError,'token length'): validate(root,case)
            rows[0]['output_tokens']=32; rows[0]['id']=1; save_rows()
            with self.assertRaisesRegex(ValueError,'unique ID'): validate(root,case)
            rows[0]['id']=0; save_rows()
            data['runtime']['warmup'].pop(); save_analysis()
            with self.assertRaisesRegex(ValueError,'warmup'): validate(root,case)
            case['warmup']=False
            data['actual']['by_dp'].pop('1');save_analysis()
            with self.assertRaisesRegex(ValueError,'DP metrics'): validate(root,case)
            data['runtime']['aisbench_exit_code']=2;save_analysis()
            with self.assertRaisesRegex(ValueError,'complete'): validate(root,case)
