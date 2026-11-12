from pathlib import Path
import unittest
from unittest.mock import patch,Mock
from deploy_eval_v2.executor_factory import build_executor
from deploy_eval_v2.protocol import ProtocolError
from test_plan_model import external_plan,eval_bindings
from test_launcher import identity


class ExecutorFactoryTests(unittest.TestCase):
    def test_remote_tester_selects_supervised_adapter(self):
        root=str(Path('/work/t'))
        tester=dict(machine_id='e'*32,host='tester',work_root='/tests',python='/python')
        values={'plan.json':dict(task_id='t',intent='deploy',execution={'max_retries':0},evaluation={'test_binding':'test'}),
                'bindings.json':dict(nodes={'a':{'machine_id':'d'*32}},testers={'test':tester}),
                'compiled.json':{'topology':{'groups':{'a/model':{'node':'a','root':root}}}}}
        with patch('deploy_eval_v2.executor_factory.validate_identity'),\
             patch('deploy_eval_v2.executor_factory.verify_runtime'),\
             patch('deploy_eval_v2.executor_factory.verify_frozen',return_value=values),\
             patch('deploy_eval_v2.executor_factory.HostGuard'),\
             patch('deploy_eval_v2.executor_factory.DockerNode'),\
             patch('deploy_eval_v2.executor_factory.OwnedExecutor') as factory:
            build_executor(root,dict(identity(),task_root=root),machine_id_reader=lambda:'d'*32)
            from deploy_eval_v2.remote_evaluation import RemoteEvaluation
            self.assertIsInstance(factory.call_args.kwargs['evaluator'],RemoteEvaluation)
            self.assertEqual(factory.call_args.kwargs['verified_tester_host'],'tester')

    def test_external_host_attestation_required_before_executor_construction(self):
        plan=external_plan()
        bound=eval_bindings()
        values={'plan.json':plan,'bindings.json':bound}
        # Use a platform-native root in the identity so constructor tests remain
        # local and do not pretend Linux /etc/machine-id was inspected on Windows.
        with patch('deploy_eval_v2.executor_factory.validate_identity'),patch('deploy_eval_v2.executor_factory.verify_runtime'),patch('deploy_eval_v2.executor_factory.verify_frozen',return_value=values),patch('deploy_eval_v2.executor_factory.ExternalExecutor') as factory:
            root=str(Path('/work/t'))
            value=dict(identity(),task_root=root)
            with self.assertRaisesRegex(ProtocolError,'machine identity'):
                build_executor(root,value,machine_id_reader=lambda:'d'*32)
            factory.assert_not_called()
            bound['testers']['tester']['machine_id']='d'*32
            build_executor(root,value,machine_id_reader=lambda:'d'*32)
            self.assertEqual(factory.call_args.kwargs['verified_host'],'example.invalid')
            factory.reset_mock()
            with self.assertRaises(ProtocolError):
                build_executor(root,value,machine_id_reader=lambda:'e'*32)
            factory.assert_not_called()
