import ast
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from deploy_eval_v2.evaluate import prepare_case
from deploy_eval_v2.plan_model import normalize_external_plan
from deploy_eval_v2.process_guard import run_logged, Cancelled
from deploy_eval_v2.protocol import ProtocolError
from test_plan_model import external_plan, eval_bindings


class EvaluateTests(unittest.TestCase):
    def test_optional_prefix_metrics_do_not_override_native_performance(self):
        from contextlib import ExitStack
        from unittest.mock import patch
        from deploy_eval_v2.evaluate import execute_external_case
        from deploy_eval_v2.durable import atomic_json
        for prefix_requested,native_status,expected in ((False,'VERIFIED','VERIFIED'),
                (True,'VERIFIED','UNVERIFIED'),(False,'UNVERIFIED','UNVERIFIED')):
            with self.subTest(prefix=prefix_requested,native=native_status), tempfile.TemporaryDirectory() as root:
                root=Path(root); tool=root/'tool';tool.touch();folder=root/'case'
                case=dict(case_id='c',timeout_seconds=10,reset_cache=False,dataset_id='d',num_warmups=1)
                if prefix_requested: case['shared_prefix_ratio']=.5
                tester=dict(host='h',python=sys.executable,aisbench=str(tool),model_path=str(root),
                            datasets={'d':dict(type='jsonl',generation_manifest='manifest')})
                plan=dict(unresolved=[],test_binding='t',cases=[case],service=dict(
                    service_id='s',endpoint='http://unused',metrics=[dict(id='p')]))
                bindings=dict(testers={'t':tester})
                def prepare(*args,**kwargs):
                    folder.mkdir();return case,tester,['tool']
                def run(*args,**kwargs):
                    atomic_json(folder/'preflight.json',dict(status='VERIFIED'));return 0
                with ExitStack() as stack:
                    for target,options in (
                        ('normalize_external_plan',dict(return_value=(plan,[]))),
                        ('validate_eval_bindings',dict(return_value=None)),
                        ('require_linux',dict(return_value=None)),
                        ('wait_health',dict(return_value=[])),
                        ('prepare_case',dict(side_effect=prepare)),
                        ('run_logged',dict(side_effect=run)),
                        ('validate_native_result',dict(return_value={})),
                        ('metric_evidence.capture',dict(return_value=None)),
                        ('metric_evidence.summarize',dict(return_value=dict(status='UNVERIFIED',
                             routing_coverage='UNVERIFIED',warmup_route_coverage='UNVERIFIED')))):
                        stack.enter_context(patch('deploy_eval_v2.evaluate.'+target,**options))
                    stack.enter_context(patch('deploy_eval_v2.performance_evidence.summarize',return_value=dict(status=native_status)))
                    result=execute_external_case(plan,bindings,'c',folder,threading.Event(),verified_local_host='h')
                self.assertEqual(result['performance_validity'],expected)
                self.assertEqual(result['prefix_metric_status'],'UNVERIFIED')

    def test_native_config_preserves_load(self):
        plan, _ = normalize_external_plan(external_plan())
        plan['cases'][0]['formal_requests'] = 17
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)/'case'
            case, tester, argv = prepare_case(plan, eval_bindings(), 'c', folder, check_files=False)
            ast.parse((folder/'aisbench-config.py').read_text())
            self.assertEqual(argv[argv.index('--num-prompts')+1], '17')
            self.assertEqual(argv[argv.index('--num-warmups')+1], '1')
            self.assertNotIn('--debug', argv)
            with self.assertRaises(FileExistsError):
                prepare_case(plan, eval_bindings(), 'c', folder, check_files=False)

    def test_cache_reset_never_silently_ignored(self):
        plan, _ = normalize_external_plan(external_plan())
        plan['service']['permissions']['reset_cache'] = True
        plan['cases'][0]['reset_cache'] = True
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ProtocolError, 'target adapter'):
                prepare_case(plan, eval_bindings(), 'c', Path(root)/'case')

    def test_explicit_external_reset_targets_survive_plan_compilation(self):
        plan,_=normalize_external_plan(external_plan())
        plan['service']['permissions']['reset_cache']=True
        plan['service']['reset_targets']=[dict(id='model',url='http://model:8000/reset_prefix_cache',
                                               metrics_url='http://model:8000/metrics')]
        plan['cases'][0]['reset_cache']=True
        with tempfile.TemporaryDirectory() as root:
            case,_,_=prepare_case(plan,eval_bindings(),'c',Path(root)/'case',check_files=False)
            self.assertTrue(case['reset_cache'])
        plan['service']['reset_targets'][0]['metrics_url']='http://different:8000/metrics'
        with self.assertRaisesRegex(ProtocolError,'same instance'):
            normalize_external_plan(plan)

    def test_real_child_output_goes_to_file(self):
        with tempfile.TemporaryDirectory() as root:
            log = Path(root)/'child.log'
            code = run_logged([sys.executable, '-c', "print('x'*1000000)"], log,
                              time.monotonic()+5, threading.Event())
            self.assertEqual(code, 0)
            self.assertGreater((log.parent/'task.jsonl').stat().st_size, 1000000)

    def test_real_child_deadline_and_cancellation(self):
        with tempfile.TemporaryDirectory() as root:
            argv = [sys.executable, '-c', 'import time; time.sleep(30)']
            with self.assertRaises(TimeoutError):
                run_logged(argv, Path(root)/'timeout.log', time.monotonic()+.2, threading.Event())
            cancelled = threading.Event()
            timer = threading.Timer(.2, cancelled.set)
            timer.start()
            try:
                with self.assertRaises(Cancelled):
                    run_logged(argv, Path(root)/'cancel.log', time.monotonic()+5, cancelled)
            finally:
                timer.join()
