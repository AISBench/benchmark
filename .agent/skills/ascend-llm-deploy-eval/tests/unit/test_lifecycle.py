import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import evaluate
import stage_runtime as runtime
from plan_run import normalize
from test_serial_task import fixture

class LifecycleTests(unittest.TestCase):
    def exercise(self,success_at=None,safe=True):
        task=normalize(fixture())
        case=task['cases'][0]
        spec={'execution':task['execution'],'benchmark':{'tool_dir':'unused'}}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['scripts','logs','results']:(root/name).mkdir()
            runtime.save(root/'deployment-artifacts.json', {'id':'test-01','path':str(root/'deployments/test-01')})
            runtime.update(root,service='READY',container_id='test',smoke_identity={'identity':1})
            calls=[]
            def attempt(argv,**kwargs):
                folder=Path(argv[-1]); n=len(calls)+1; calls.append(n)
                passed=n==success_at
                runtime.save(folder/'cleanup.json',{'released':safe})
                runtime.save(folder/'summary.json',{'status':'PASSED' if passed else 'FAILED','error':None if passed else 'test failure'})
                return SimpleNamespace(returncode=0 if passed else 2)
            with patch.object(evaluate,'read_plan',return_value=spec),patch.object(evaluate,'ready',return_value=True),patch.object(evaluate,'service_identity',return_value={'identity':1}),patch.object(evaluate.subprocess,'run',side_effect=attempt),patch.object(evaluate.time,'sleep'):
                if success_at:
                    self.assertEqual(evaluate.evaluate(root,case)['status'],'PASSED')
                else:
                    with self.assertRaises((ValueError,RuntimeError)):evaluate.evaluate(root,case)
            return calls,runtime.case_state(root,case['id'])
    def test_four_attempts_total_and_evidence_retained(self):
        calls,state=self.exercise()
        self.assertEqual(calls,[1,2,3,4]); self.assertEqual(state['status'],'FAILED')
        self.assertEqual(len({a['folder'] for a in state['attempts']}),4)
    def test_retry_stops_on_success(self):
        calls,state=self.exercise(success_at=2)
        self.assertEqual(calls,[1,2]); self.assertEqual(state['status'],'PASSED')
    def test_unsafe_cleanup_not_retried(self):
        calls,state=self.exercise(safe=False)
        self.assertEqual(calls,[1]); self.assertEqual(state['status'],'FAILED')
    def test_stale_smoke_invalidates_ready(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'logs').mkdir()
            runtime.update(root,service='READY',smoke_identity={'pid':1})
            with patch.object(runtime,'service_identity',return_value={'pid':2}):
                self.assertFalse(runtime.ready(root,{}))
            self.assertEqual(runtime.state(root)['service'],'UNKNOWN')

class ReleaseTests(unittest.TestCase):
    def test_stop_clears_process_registry_and_repeat_is_safe(self):
        import finalize
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'logs').mkdir()
            runtime.update(root,service='READY',container_id='cid',image_id='image')
            runtime.save(root/'service-processes.json',[{'pid':42,'start_ticks':'123'}])
            c={'Id':'cid','Image':'image','State':{'Running':False,'Pid':0}}
            with patch.object(finalize,'read_plan',return_value={}),patch.object(finalize,'inspect_owned',return_value=c),patch.object(finalize,'command',return_value='No running processes found in NPU 0'):
                finalize.release(root,'stop_container','job')
                finalize.release(root,'stop_container','job')
            self.assertEqual(json.loads((root/'service-processes.json').read_text()),[])
            self.assertEqual(runtime.state(root)['service'],'RELEASED')
            self.assertEqual(len(runtime.state(root)['releases']),2)

class DeploymentRetryTests(unittest.TestCase):
    def exercise(self,error):
        import cluster_runtime as cluster
        from unittest.mock import MagicMock
        client=MagicMock();client.call.side_effect=error
        with tempfile.TemporaryDirectory() as d,patch.object(cluster,'clients',return_value={'node':client}),patch.object(cluster,'release_all',return_value={'node':{'status':'PASSED'}}) as release:
            with self.assertRaises(type(error)):
                cluster.deploy(Path(d),{'execution':{'max_retries':3}}, {})
            return client.call.call_count,release.call_count,len(list(Path(d).glob('deploy-*/failure.json')))
    def test_bounded_deploy_retry(self):
        self.assertEqual(self.exercise(RuntimeError('startup failure')),(4,4,4))
    def test_ownership_not_retried(self):
        self.assertEqual(self.exercise(ValueError('ownership')),(1,1,1))

class LockTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform=='linux','Linux flock integration')
    def test_container_lock_excludes_other_process_and_releases(self):
        import os
        import subprocess
        spec={'target':{'container':'unit-lock-'+str(os.getpid())}}
        env=dict(os.environ, PYTHONPATH=str(Path(runtime.__file__).parent))
        code='import json; from stage_runtime import locked\nspec=json.loads('+repr(json.dumps(spec))+')\nwith locked(spec): print("acquired")'
        with runtime.locked(spec):
            child=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=env)
            self.assertNotEqual(child.returncode,0)
            self.assertIn('Container is locked',child.stderr)
        child=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env=env)
        self.assertEqual(child.returncode,0,child.stderr)

class SharedContainerTests(unittest.TestCase):
    def test_service_fingerprint_does_not_own_shared_container(self):
        self.assertEqual(runtime.ownership_digest({'container_fingerprint':'pool','service_fingerprint':'one'}),'pool')
        self.assertEqual(runtime.ownership_digest({'container_fingerprint':'pool','service_fingerprint':'two'}),'pool')

    def test_job_end_removes_shared_container_once(self):
        import finalize
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir()
            spec={'container_registry':str(root/'pool.json'),'container_fingerprint':'pool','service_fingerprint':'service'}
            runtime.save_pool(spec,container_id='cid',image_id='image',active_run_dir=str(root),removed=False)
            runtime.update(root,service='READY',container_id='cid',image_id='image')
            c={'Id':'cid','Image':'image','State':{'Running':False,'Pid':0}}
            with patch.object(finalize,'read_plan',return_value=spec),patch.object(finalize,'inspect_owned',return_value=c),patch.object(finalize,'command',return_value='') as command:
                result=finalize.release(root,'stop_container','job')
                repeat=finalize.release(root,'stop_container','job')
            self.assertTrue(result['container_removed'])
            self.assertTrue(runtime.pool_state(spec)['removed'])
            self.assertEqual(sum(c.args[0][:2]==['docker','rm'] for c in command.call_args_list),1)
            self.assertEqual(repeat['status'],'released')

    def test_switch_reuses_registered_container_and_rejects_active_owner(self):
        import deploy
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir();(root/'scripts').mkdir()
            runtime.save(root/'preflight-evidence.json',{'image_id':'image'})
            spec={'container_registry':str(root/'pool.json'),'container_fingerprint':'pool','service_fingerprint':'second',
                  'target':{'container':'job'}}
            runtime.save_pool(spec,container_id='cid',image_id='image',active_run_dir='previous',removed=False)
            with patch.object(deploy,'read_plan',return_value=spec),patch.object(deploy,'ready',return_value=False),patch.object(deploy,'check'):
                with self.assertRaises(ValueError):deploy.start_node(root)
                runtime.save_pool(spec,active_run_dir='')
                with patch.object(deploy.subprocess,'run',return_value=SimpleNamespace(returncode=0)),patch.object(deploy,'inspect_owned',return_value={'Id':'cid','Image':'image','State':{'Running':True}}),patch.object(deploy,'command',return_value='') as command:
                    deploy.start_node(root)
                self.assertFalse(any(c.args[0][:2]==['docker','run'] for c in command.call_args_list))
                self.assertEqual(runtime.pool_state(spec)['active_run_dir'],str(root))
