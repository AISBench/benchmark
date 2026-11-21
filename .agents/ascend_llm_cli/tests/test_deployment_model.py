import copy
from pathlib import Path
import unittest
from deploy_eval_v2.deployment_model import validate_deploy_plan
from deploy_eval_v2.json_io import load_json
from deploy_eval_v2.protocol import ProtocolError
from deploy_eval_v2.compiler import verify_review, review_inputs


def deploy_plan():
    return dict(schema_version='1.0', task_id='t', intent='deploy',
                deployment=load_json(Path(__file__).parent/'fixtures/deployment.json'),
                policy=dict(idle_timeout_seconds=900, on_idle='keep_service', on_failure='release_owned'),
                execution=dict(startup_timeout_seconds=1800, smoke_timeout_seconds=30,
                               dependency_timeout_seconds=120, proxy_start_after_models_healthy=True,
                               enable_cache_reset_api=False, max_retries=0), unresolved=[])


class DeploymentModelTests(unittest.TestCase):
    def test_review_cannot_bypass_deployment_validation(self):
        plan = deploy_plan()
        plan['deployment']['endpoint']['port'] = 'missing'
        review = dict(schema_version='1.0', status='APPROVED', unresolved=[],
                      evidence_refs=['review/deploy.json'], inputs=review_inputs(plan, {}, 'a'*64))
        with self.assertRaisesRegex(ProtocolError, 'endpoint reference'):
            verify_review(plan, {}, review, 'a'*64)

    def test_preserves_full_reference_payload(self):
        plan = deploy_plan()
        launch = plan['deployment']['instances'][0]['launch']
        launch['args']['new-version-parameter'] = {'nested': [1, False, '中文']}
        launch['kv_transfer_config'] = dict(kv_connector='C', kv_role='kv_producer',
            kv_port=18000, engine_id='p', future={'field': True})
        original = copy.deepcopy(plan)
        validate_deploy_plan(plan)
        self.assertEqual(plan, original)

    def test_missing_explicit_privilege_and_policy(self):
        plan = deploy_plan()
        del plan['deployment']['runtime']['container']['privileged']
        with self.assertRaises(ProtocolError):
            validate_deploy_plan(plan)
        plan = deploy_plan()
        del plan['policy']['on_idle']
        with self.assertRaises(ProtocolError):
            validate_deploy_plan(plan)

    def test_broken_references_are_rejected(self):
        for mutation in ('duplicate', 'endpoint', 'kv', 'route'):
            plan = deploy_plan()
            dep = plan['deployment']
            if mutation == 'duplicate':
                dep['instances'].append(copy.deepcopy(dep['instances'][0]))
            elif mutation == 'endpoint':
                dep['endpoint']['instance'] = 'missing'
            elif mutation == 'kv':
                dep['connections']['kv_transfer'][0]['producer'] = 'decode'
            else:
                dep['connections']['routing'][0]['prefillers'] = ['decode']
            with self.assertRaises(ProtocolError):
                validate_deploy_plan(plan)
