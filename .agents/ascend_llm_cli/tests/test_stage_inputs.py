import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from deploy_eval_v2.stage_inputs import prepare
from deploy_eval_v2.protocol import ProtocolError
from deploy_eval_v2.task_runtime import Runtime
from deploy_eval_v2.fake_runtime import FakeExecutor
from deploy_eval_v2.task_log import diagnostic
from test_plan_model import external_plan,eval_bindings
import test_service_contract
from test_deployment_model import deploy_plan
from test_runtime import command,finish


class IndependentStageTests(unittest.TestCase):
    def eval_request(self,destination):
        return dict(stage='eval',destination=str(destination),task_id='evaluation',
                    endpoint='http://localhost:8000',model='m',cases=external_plan()['cases'],
                    tester=eval_bindings()['testers']['tester'])

    def test_endpoint_only_needs_no_deployment_and_preserves_load(self):
        with tempfile.TemporaryDirectory() as folder:
            request=self.eval_request(Path(folder)/'prepared')
            request['cases'][0]['formal_requests']=4
            original=copy.deepcopy(request)
            result=prepare(request)
            plan=json.loads(Path(result['plan']).read_text())
            self.assertEqual(plan['intent'],'external_eval')
            self.assertEqual(plan['cases'][0]['formal_requests'],4)
            self.assertFalse(plan['service']['permissions']['release'])
            self.assertEqual(request,original)
            self.assertTrue(result['review_required'])

    def test_deployment_service_is_usable_without_granting_release(self):
        with tempfile.TemporaryDirectory() as folder:
            request=self.eval_request(Path(folder)/'prepared')
            del request['endpoint'];del request['model']
            request['service']=test_service_contract.ServiceContractTests().fixture()
            result=prepare(request)
            plan=json.loads(Path(result['plan']).read_text())
            self.assertEqual(plan['service']['service_id'],'s')
            self.assertEqual(plan['service']['ownership'],'external')
            self.assertFalse(plan['service']['permissions']['restart'])

    def test_deploy_only_rejects_hidden_evaluation_or_idle_release(self):
        with tempfile.TemporaryDirectory() as folder:
            plan=deploy_plan();plan['policy']['on_idle']='release_owned'
            with self.assertRaisesRegex(ProtocolError,'retained'):
                prepare(dict(stage='deploy',destination=str(Path(folder)/'p'),deployment=plan,bindings={}))
            plan['policy']=[]
            with self.assertRaises(ProtocolError):
                prepare(dict(stage='deploy',destination=str(Path(folder)/'p'),deployment=plan,bindings={}))
            plan=deploy_plan();plan['evaluation']={}
            with self.assertRaisesRegex(ProtocolError,'cannot embed'):
                prepare(dict(stage='deploy',destination=str(Path(folder)/'p'),deployment=plan,bindings={}))

    def test_diagnostics_share_event_log_and_do_not_break_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime=Runtime(folder,'t',FakeExecutor())
            threads=[threading.Thread(target=diagnostic,args=(Path(folder)/'preflight.log','output-'+str(i))) for i in range(8)]
            for thread in threads:thread.start()
            accepted=runtime.submit(command(plan_ref='p'));finish(runtime)
            for thread in threads:thread.join()
            recovered=Runtime(folder,'t',FakeExecutor())
            self.assertEqual(recovered.submit(command(plan_ref='p')),accepted)
            self.assertEqual(recovered.state,'READY')
            self.assertEqual(list(Path(folder).rglob('*.log')),[])
            self.assertFalse((Path(folder)/'commands').exists())
            records=[json.loads(line) for line in (Path(folder)/'task.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r.get('record_type')=='diagnostic' for r in records),8)

    def test_deploy_only_prepares_valid_plan_without_tester(self):
        from test_deployment_compiler import bindings
        with tempfile.TemporaryDirectory() as folder:
            plan=deploy_plan();plan['policy']['on_idle']='keep_service'
            prepared=prepare(dict(stage='deploy',destination=str(Path(folder)/'p'),deployment=plan,bindings=bindings()))
            self.assertEqual(json.loads(Path(prepared['plan']).read_text()),plan)
            self.assertNotIn('testers',json.loads(Path(prepared['bindings']).read_text()))

    def test_public_cli_prepare_export_and_errors_without_transport(self):
        import subprocess,sys
        from deploy_eval_v2.hashes import digest
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for command,request in [('prepare-inputs',self.eval_request(root/'prepared')),
                                    ('export-contracts',dict(destination=str(root/'contracts')))]:
                source=root/'request.json';source.write_text(json.dumps(request))
                child=subprocess.run([sys.executable,'-m','deploy_eval_v2',command,'--request',str(source)],capture_output=True)
                self.assertEqual(child.returncode,0,child.stderr+child.stdout)
                self.assertEqual(json.loads(child.stdout)['status'],'EVENTS')
            from deploy_eval_v2 import stage_inputs
            schemas=Path(stage_inputs.__file__).parent/'schemas'
            self.assertEqual({p.name:p.read_bytes() for p in schemas.glob('*.json')},
                             {p.name:p.read_bytes() for p in (root/'contracts').glob('*.json')})
            bad=self.eval_request(root/'bad');bad['service']=[];del bad['endpoint'];del bad['model']
            source.write_text(json.dumps(bad))
            child=subprocess.run([sys.executable,'-m','deploy_eval_v2','prepare-inputs','--request',str(source)],capture_output=True)
            self.assertEqual(child.returncode,2)
            self.assertEqual(json.loads(child.stdout)['error']['code'],'INVALID_INPUT')
            self.assertFalse((root/'bad').exists())

    def test_malformed_diagnostic_cannot_bypass_journal_validation(self):
        from deploy_eval_v2.event_journal import Journal
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'task.jsonl'
            for record in ({'record_type':'diagnostic'},[],{'record_type':'diagnostic','message':'x','source':'test','time':'now'}):
                target.write_text(json.dumps(record))  # incomplete lines are also corruption without repair
                with self.assertRaisesRegex(ProtocolError,'RECOVERY_REQUIRED'):
                    Journal(folder,'t')

    def test_native_log_preserved_and_framework_logging_failure_propagates(self):
        import sys,time
        from unittest.mock import patch
        from deploy_eval_v2.process_guard import run_logged
        with tempfile.TemporaryDirectory() as folder:
            log=Path(folder)/'aisbench.log'
            code=run_logged([sys.executable,'-c',"print('native output')"],log,time.monotonic()+5,threading.Event())
            self.assertEqual(code,0)
            self.assertEqual(log.read_text(),'native output\n')
            with patch('deploy_eval_v2.task_log.diagnostic',side_effect=[None,OSError('disk full'),None]):
                with self.assertRaisesRegex(RuntimeError,'log write failed'):
                    run_logged([sys.executable,'-c',"print('diagnostic')"],Path(folder)/'preflight.log',time.monotonic()+5,threading.Event())
