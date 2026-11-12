from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from deploy_eval_v2.api import API
from deploy_eval_v2.compiler import freeze_inputs,review_inputs
from deploy_eval_v2.durable import atomic_json
from deploy_eval_v2.owned_executor import OwnedExecutor
from deploy_eval_v2.remote_guard import HostGuard
from deploy_eval_v2.task_runtime import Runtime
from deploy_eval_v2.protocol import ProtocolError
from test_deployment_model import deploy_plan
from test_deployment_compiler import bindings
from test_plan_model import external_plan,eval_bindings
from deploy_eval_v2.plan_model import normalize_external_plan


class OwnedExecutorTests(unittest.TestCase):
    def test_unknown_remote_client_cleanup_retains_owned_service(self):
        from deploy_eval_v2.remote_evaluation import RemoteCleanupUnknown
        with tempfile.TemporaryDirectory() as folder:
            api,executor,ref,node=self.setup_runtime(folder,Mock(side_effect=RemoteCleanupUnknown('stop unverified')))
            api.submit(dict(protocol_version='1.0',task_id='t',kind='deploy',request_id='d',expected_version=0,
                            plan_ref=ref['sha256'],wait_ms=1000))
            response=api.submit(dict(protocol_version='1.0',task_id='t',kind='eval',request_id='e',
                expected_version=api.runtime.version,service_id=executor.service_id,case_id='c',wait_ms=1000))
            self.assertEqual(response['events'][-1]['data']['cleanup_status'],'UNVERIFIED')
            self.assertEqual(api.status()['task_state'],'RECOVERY_REQUIRED')
            node.runner.assert_not_called()

    def test_modified_service_cannot_authorize_release(self):
        from copy import deepcopy
        mutations = {
            'runtime': lambda s: s.update(runtime_sha256='f'*64),
            'endpoint': lambda s: s.update(endpoint='http://other.invalid:8000'),
            'model': lambda s: s.update(model='other-model'),
            'policy': lambda s: s['policy'].update(idle_timeout_seconds=99999),
            'groups': lambda s: s.update(containers={'other/model':next(iter(s['containers'].values()))}),
            'metrics': lambda s: s.update(metrics=[{'endpoint':'http://other.invalid'}]),
            'short_id': lambda s: next(iter(s['containers'].values())).update(container_id='abc'),
            'evidence': lambda s: s.update(health_evidence_ref='results/../outside'),
        }
        with tempfile.TemporaryDirectory() as folder:
            api,executor,ref,node=self.setup_runtime(folder)
            api.submit(dict(protocol_version='1.0',task_id='t',kind='deploy',request_id='d',expected_version=0,
                            plan_ref=ref['sha256'],wait_ms=1000))
            original=executor.service()
            version=api.runtime.version
            for name,mutate in mutations.items():
                with self.subTest(name=name):
                    changed=deepcopy(original)
                    mutate(changed)
                    atomic_json(executor.root/'service.json',changed)
                    with self.assertRaisesRegex(ProtocolError,'RECOVERY_REQUIRED'):
                        api.submit(dict(protocol_version='1.0',task_id='t',kind='release',request_id='r-'+name,
                                        expected_version=version,service_id=executor.service_id,wait_ms=1000))
                    self.assertEqual(api.runtime.version,version)
                    node.runner.assert_not_called()
            atomic_json(executor.root/'service.json',original)

    def setup_runtime(self,folder,evaluator=None,idle_release=False):
        plan,bound=deploy_plan(),bindings()
        if idle_release:
            plan['policy'].update(idle_timeout_seconds=1,on_idle='release_owned')
        if evaluator is not None:
            evaluation,_=normalize_external_plan(external_plan())
            evaluation['cases'][0]['formal_requests']=17
            plan['evaluation']={key:evaluation[key] for key in ('cases','test_binding')}
            bound['testers']=eval_bindings()['testers']
        digest='a'*64
        review=dict(schema_version='1.0',status='APPROVED',unresolved=[],
            evidence_refs=['review/deploy.json'],inputs=review_inputs(plan,bound,digest))
        frozen=Path(folder)/'frozen'
        reference=freeze_inputs(frozen,plan,bound,review,digest)
        node=Mock()
        node.guard=HostGuard(Path(folder)/'guard','t','b'*64)
        node.runner=Mock(side_effect=lambda argv, **kw: (1 if argv[0]=='fuser' else 0,b'',b''))
        def deploy(compiled,nodes,execution,target,cancelled):
            result=dict(execution_status='SUCCEEDED',endpoint=compiled['topology']['endpoint'],
                        containers={'node_a/model':{'container_id':'c'*64}},health=[],smoke={'status':'VERIFIED'})
            atomic_json(target,result)
            return result
        options={} if evaluator is None else dict(evaluator=evaluator,verified_tester_host='example.invalid')
        executor=OwnedExecutor(Path(folder)/'task',frozen,reference['sha256'],digest,{'node_a':node},deployer=deploy,**options)
        runtime=Runtime(Path(folder)/'task','t',executor)
        return API(runtime),executor,reference,node

    def test_runtime_deploy_then_release_and_old_command_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            api,executor,reference,node=self.setup_runtime(folder)
            command=dict(protocol_version='1.0',task_id='t',kind='deploy',request_id='d',
                         expected_version=0,plan_ref=reference['sha256'],wait_ms=1000)
            response=api.submit(command)
            self.assertEqual(response['events'][-1]['type'],'deployment.ready')
            self.assertEqual(api.status()['task_state'],'READY')
            release=dict(protocol_version='1.0',task_id='t',kind='release',request_id='r',
                         expected_version=api.runtime.version,service_id=executor.service_id,wait_ms=1000)
            result=api.submit(release)
            self.assertEqual(result['events'][-1]['type'],'release.completed')
            self.assertEqual(api.status()['task_state'],'RELEASED')
            calls=node.runner.call_count
            api.submit(release)
            self.assertEqual(node.runner.call_count,calls)
            with api.runtime.lock:
                self.assertIsNone(api.runtime.active)

    def test_reject_wrong_frozen_reference_before_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            api,executor,reference,node=self.setup_runtime(folder)
            with self.assertRaisesRegex(ProtocolError,'plan_ref'):
                api.submit(dict(protocol_version='1.0',task_id='t',kind='deploy',request_id='d',
                                expected_version=0,plan_ref='wrong',wait_ms=0))
            self.assertEqual(api.runtime.version,0)
            node.runner.assert_not_called()

    def test_owned_evaluation_preserves_case_and_service_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            evaluator=Mock(return_value=dict(execution_status='SUCCEEDED',evidence_status='PARTIAL',performance_validity='UNVERIFIED'))
            api,executor,ref,node=self.setup_runtime(folder,evaluator)
            api.submit(dict(protocol_version='1.0',task_id='t',kind='deploy',request_id='d',expected_version=0,
                            plan_ref=ref['sha256'],wait_ms=1000))
            response=api.submit(dict(protocol_version='1.0',task_id='t',kind='eval',request_id='e',
                                expected_version=api.runtime.version,service_id=executor.service_id,case_id='c',wait_ms=1000))
            self.assertEqual(response['events'][-1]['type'],'evaluation.completed')
            adapted=evaluator.call_args.args[0]
            self.assertEqual(adapted['cases'][0]['formal_requests'],17)
            self.assertEqual(adapted['service']['ownership'],'owned')
            self.assertEqual(adapted['service']['service_id'],executor.service_id)
            self.assertFalse(adapted['service']['permissions']['release'])
            node.runner.assert_not_called()
            with api.runtime.lock:
                self.assertEqual(api.runtime.state,'EVALUATED')

    def test_evaluation_failure_uses_owning_tasks_cleanup_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            api,executor,ref,node=self.setup_runtime(folder,Mock(side_effect=RuntimeError('benchmark failed')))
            api.submit(dict(protocol_version='1.0',task_id='t',kind='deploy',request_id='d',expected_version=0,
                            plan_ref=ref['sha256'],wait_ms=1000))
            response=api.submit(dict(protocol_version='1.0',task_id='t',kind='eval',request_id='e',
                                expected_version=api.runtime.version,service_id=executor.service_id,case_id='c',wait_ms=1000))
            event=response['events'][-1]
            self.assertEqual(event['type'],'operation.failed')
            self.assertEqual(event['data']['cleanup_status'],'VERIFIED')
            self.assertEqual(node.runner.call_count,9)  # inventory plus eight frozen NPU probes
            with api.runtime.lock:
                self.assertEqual(api.runtime.state,'FAILED')

    def test_idle_release_runs_without_another_agent_command(self):
        with tempfile.TemporaryDirectory() as folder:
            api,executor,ref,node=self.setup_runtime(folder,idle_release=True)
            response=api.submit(dict(protocol_version='1.0',task_id='t',kind='deploy',request_id='d',expected_version=0,
                            plan_ref=ref['sha256'],wait_ms=1000))
            response=api.runtime.journal.watch(response['next_cursor'],3000)
            self.assertEqual(response['events'][-1]['type'],'task.expired')
            self.assertEqual(response['events'][-1]['data']['cleanup_status'],'VERIFIED')
            with api.runtime.lock:
                self.assertEqual(api.runtime.state,'RELEASED')
                self.assertIsNone(api.runtime.idle_timer)
            self.assertEqual(node.runner.call_count,9)

    def test_task_cancel_releases_idle_service_without_operation_id(self):
        with tempfile.TemporaryDirectory() as folder:
            api,executor,ref,node=self.setup_runtime(folder)
            api.submit(dict(protocol_version='1.0',task_id='t',kind='deploy',request_id='d',expected_version=0,
                            plan_ref=ref['sha256'],wait_ms=1000))
            result=api.submit(dict(protocol_version='1.0',task_id='t',kind='cancel',request_id='cancel',
                                  expected_version=api.runtime.version,scope='task',wait_ms=1000))
            self.assertEqual(result['events'][-1]['type'],'operation.cancelled')
            self.assertEqual(result['events'][-1]['data']['cleanup_status'],'VERIFIED')
            with api.runtime.lock:
                self.assertTrue(api.runtime.task_cancelled)
                self.assertIsNone(api.runtime.active)
