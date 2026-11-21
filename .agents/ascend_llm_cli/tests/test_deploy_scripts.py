import hashlib
import shlex
import unittest
import tempfile
from pathlib import Path
from deploy_eval_v2.compiler import freeze_inputs, verify_frozen, review_inputs, digest, canonical
from deploy_eval_v2.protocol import ProtocolError
from deploy_eval_v2.deploy_scripts import compile_service_scripts, render_service_script
from deploy_eval_v2.deployment_compiler import compile_bound_deployment
from test_deployment_compiler import bindings
from test_deployment_model import deploy_plan


class DeployScriptTests(unittest.TestCase):
    def test_frozen_deployment_carries_scripts_and_rejects_changed_compilation(self):
        plan, bound = deploy_plan(), bindings()
        runtime = 'a'*64
        review = dict(schema_version='1.0', status='APPROVED', unresolved=[],
                      evidence_refs=['review/deploy.json'], inputs=review_inputs(plan, bound, runtime))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/'frozen'
            ref = freeze_inputs(root, plan, bound, review, runtime)
            values = verify_frozen(root, ref['sha256'], runtime)
            self.assertIn('scripts', values['compiled.json'])
            changed = values['compiled.json']
            changed['scripts']['node_a/model'][0]['content'] += '\necho changed\n'
            (root/'compiled.json').write_bytes(canonical(changed))
            # Even a caller rehashing the manifest cannot make changed generated
            # scripts match the approved deterministic inputs.
            import json
            manifest = json.loads((root/'manifest.json').read_bytes())
            manifest['files']['compiled.json'] = digest(changed)
            (root/'manifest.json').write_bytes(canonical(manifest))
            with self.assertRaisesRegex(ProtocolError, 'compilation mismatch'):
                verify_frozen(root, digest(manifest), runtime)

    def test_reference_env_is_applied_after_bootstrap_and_logs_before_it(self):
        compiled = compile_bound_deployment(deploy_plan(), bindings())
        scripts = compile_service_scripts(compiled)['node_a/model']
        prefill = next(s for s in scripts if s['instance']=='prefill' and s['purpose']=='service')
        source = prefill['content']
        self.assertLess(source.index('exec >>'), source.index('source /root/.bashrc'))
        self.assertLess(source.index('unset ASCEND_RT_VISIBLE_DEVICES'), source.index('export ASCEND_RT_VISIBLE_DEVICES='))
        self.assertEqual(prefill['sha256'], hashlib.sha256(source.encode()).hexdigest())
        command = compiled['groups']['node_a/model']['commands'][0]
        self.assertEqual(shlex.split(source.splitlines()[-1])[1:], command['argv'])
        dependency = next(s for s in scripts if s['instance']=='prefill' and s['purpose']=='dependency')
        self.assertNotIn('source /root/.bashrc', dependency['content'])
        self.assertIn('import torch_npu', dependency['content'])
        self.assertEqual(dependency['log']['host'], 'user@benchmark-host.example')

    def test_shell_metacharacters_in_bound_arguments_are_preserved(self):
        command = dict(argv=['server', "a'$(touch /tmp/bad)", '中文'],
                       env={'TOKEN': '$HOME; echo bad'}, bootstrap='')
        source = render_service_script(command, '/task/logs/model output.log')
        self.assertEqual(shlex.split(source.splitlines()[-1])[1:], command['argv'])
        self.assertIn("export TOKEN='$HOME; echo bad'", source)


if __name__ == '__main__':
    unittest.main()
