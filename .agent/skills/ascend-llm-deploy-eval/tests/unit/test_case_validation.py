import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[2]/'scripts/validate_case.py'
spec = importlib.util.spec_from_file_location('validate_case', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ValidationTests(unittest.TestCase):
    def test_per_request_output_and_warmup_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for phase, output in [('01',1),('02',1024)]:
                folder = root/'tool/outputs'/phase/'performances/model'
                folder.mkdir(parents=True)
                (folder/'gsm8k_details.jsonl').write_text('\n'.join(json.dumps(dict(id=i,success=True,error_info='',input_tokens=32768 if phase=='01' else 65536,output_tokens=output)) for i in range(2)))
            log = ''
            for q,h in [(0,0),(10,0),(10,0),(30,10)]:
                log += f'queries: normal={{0: {q}, 1: {q}}}, external={{}}\nhits: normal={{0: {h}, 1: {h}}}, external={{}}\n'
            (root/'tool/aisbench.log').write_text(log)
            for port in [9000,9001]:
                (root/f'reset-{port}-0.json').write_text('{"success":true}')
            case = dict(id='case',formal_requests=2,input_tokens=65536,output_tokens=1024,prefix_ratio=0.5)
            self.assertEqual(module.validate(root,case)['measured_prefix_ratio'],0.5)
            path = root/'tool/outputs/02/performances/model/gsm8k_details.jsonl'
            path.write_text(path.read_text().replace('1024','1023',1))
            with self.assertRaises(ValueError):
                module.validate(root,case)
