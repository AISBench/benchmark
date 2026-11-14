from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from deploy_eval_v2.readiness import check_ready
from deploy_eval_v2.compiler import freeze_inputs,review_inputs
from deploy_eval_v2.publication import build_archive
from deploy_eval_v2.plan_model import normalize_external_plan
from test_plan_model import external_plan,eval_bindings


class ReadinessTests(unittest.TestCase):
    def test_owned_unsupported_parameters_are_reported_without_remote_actions(self):
        values={'plan.json':dict(task_id='t',intent='deploy',execution={'max_retries':1},
                    evaluation=dict(test_binding='tester',cases=[{'reset_cache':True}])),
                'bindings.json':dict(nodes={'a':{'machine_id':'a'*32}},
                    testers={'tester':{'machine_id':'b'*32,'work_root':'/test','host':'tester-host'}}),
                'compiled.json':{'topology':{'groups':{'a/model':dict(node='a',host='host-a',root='/work/t')}}}}
        request=dict(task_ref='t',runtime_archive='unused.zip',input_directory='unused',
                     runtime_sha256='c'*64,bundle_sha256='d'*64)
        config={'tasks':{'t':dict(host='host-a',root='/work/t',allowed_root='/work',
                                tester=dict(host='tester-host',root='/test/t',allowed_root='/test'))}}
        with patch('deploy_eval_v2.readiness.verify_frozen',return_value=values),\
             patch('deploy_eval_v2.readiness.install_archive'),\
             patch('deploy_eval_v2.readiness.shutil.which',return_value='/ssh'):
            result=check_ready(request,config)
            self.assertEqual(result['readiness']['blockers'],[])
            values['plan.json']['execution']['max_retries']=0
            values['plan.json']['evaluation']['cases'][0]['reset_cache']=False
            values['bindings.json']['testers']['tester']['machine_id']='a'*32
            self.assertEqual(check_ready(request,config)['readiness']['status'],'READY_FOR_HOST_PREFLIGHT')

    def test_offline_readiness_requires_attested_tester_and_reports_no_remote_validation(self):
        for attested in (False,True):
            with self.subTest(attested=attested),tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                import deploy_eval_v2
                package=build_archive(Path(deploy_eval_v2.__file__).parent,root/'runtime.zip')
                plan,_=normalize_external_plan(external_plan())
                bindings=eval_bindings()
                if attested:
                    bindings['testers']['tester']['machine_id']='a'*32
                runtime=package['runtime_sha256']
                review=dict(schema_version='1.0',status='APPROVED',unresolved=[],evidence_refs=['review.json'],
                            inputs=review_inputs(plan,bindings,runtime))
                frozen=freeze_inputs(root/'inputs',plan,bindings,review,runtime)
                request=dict(task_ref='t',runtime_archive=str(root/'runtime.zip'),input_directory=str(root/'inputs'),
                             runtime_sha256=runtime,bundle_sha256=frozen['sha256'])
                config={'tasks':{'t':dict(host='example.invalid',root='/work/t',allowed_root='/work')}}
                with patch('deploy_eval_v2.readiness.shutil.which',return_value='/ssh'):
                    result=check_ready(request,config)
                self.assertEqual(result['readiness']['status'],'READY_FOR_HOST_PREFLIGHT' if attested else 'BLOCKED')
                self.assertEqual(result['readiness']['remote_checks'],'NOT_RUN')
                self.assertFalse(result['started'])
