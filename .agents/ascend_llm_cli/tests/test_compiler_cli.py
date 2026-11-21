import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from deploy_eval_v2.cli import main
from deploy_eval_v2.compiler import review_inputs
from deploy_eval_v2.json_io import load_json
from deploy_eval_v2.protocol import ProtocolError
from test_cli import Output
from test_plan_model import external_plan, eval_bindings
from deploy_eval_v2.plan_model import normalize_external_plan


class CompilerCLITests(unittest.TestCase):
    def test_local_freeze_verify_without_ssh(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, _ = normalize_external_plan(external_plan())
            bindings = eval_bindings()
            runtime = 'a'*64
            review = dict(schema_version='1.0', status='APPROVED',
                          inputs=review_inputs(plan, bindings, runtime), unresolved=[],
                          evidence_refs=['review/inventory.json'])
            def write(name, value):
                path = root/name
                path.write_text(json.dumps(value), encoding='utf-8')
                return str(path)
            request = {name: write(name+'.json', value) for name, value in
                       [('plan', plan), ('bindings', bindings), ('review', review)]}
            request.update(destination=str(root/'frozen'), runtime_sha256=runtime)
            with patch('deploy_eval_v2.cli.exchange') as ssh:
                output = Output()
                with patch('deploy_eval_v2.cli.sys.stdout', output):
                    code = main(['freeze-inputs', '--request', write('freeze.json', request)])
                self.assertEqual(code, 0)
                result = json.loads(output.buffer.getvalue())
                self.assertFalse(result['production_ready'])
                verify = dict(directory=request['destination'], sha256=result['input_bundle']['sha256'], runtime_sha256=runtime)
                with patch('deploy_eval_v2.cli.sys.stdout', Output()):
                    self.assertEqual(main(['verify-inputs', '--request', write('verify.json', verify)]), 0)
                ssh.assert_not_called()

    def test_ambiguous_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'input.json'
            for raw in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'):
                path.write_text(raw)
                with self.assertRaises(ProtocolError):
                    load_json(path)
