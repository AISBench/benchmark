import copy,json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import run as entry
from task_model import validate_stages


def fixture():
    subs=[{'id':x,'deployment_id':x,'case_id':'case','after_case':'stop_container','on_failure':'record_only'} for x in 'abc']
    spec={'sub_tasks':subs,'cases':[{'id':'case','concurrency':1}], 'execution':{'max_retries':0,'startup_timeout_seconds':1,'smoke':'skip'},
          'stages':[{'id':'parallel','branches':[{'id':x,'sub_tasks':[x]} for x in 'ab']},{'id':'next','branches':[{'id':'c','sub_tasks':['c']}]}]}
    plan={'deployments':{},'tests':{}}
    for x in 'abc':
        plan['deployments'][x]={'groups':{x:{'host':x,'root':'/unused','log_names':[]}},'health_urls':[]}
        plan['tests'][x]={'node':x,'timeout':1}
    return spec,plan

class StageTests(unittest.TestCase):
    def exercise(self, failure=None):
        spec,plan=fixture(); calls=[]; barrier=threading.Barrier(2); writers=set(); owner=threading.get_ident()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'.runtime').mkdir(); (root/'.runtime/manifest.json').write_text('{"files":{}}')
            original=entry.save
            def save(path,value):
                writers.add(threading.get_ident());original(path,value)
            def invoke(group,path,*args,**kwargs):
                calls.append(path)
                if path in ('scripts/tests/a.sh','scripts/tests/b.sh'):
                    barrier.wait(timeout=3)
                if failure=='branch' and path=='scripts/tests/a.sh': raise RuntimeError('case failed')
                if failure=='interrupt' and path=='scripts/tests/a.sh': raise KeyboardInterrupt()
                if failure=='release' and path=='scripts/a/a-stop.sh': raise RuntimeError('release unknown')
            def collect(*args):
                calls.append('collect')
                if failure=='artifacts': raise RuntimeError('Missing required artifacts')
            with patch.object(entry,'save',side_effect=save),patch.object(entry,'distribute'),patch.object(entry,'claim_execution'),patch.object(entry,'invoke',side_effect=invoke),patch.object(entry,'collect',side_effect=collect):
                if failure:
                    with self.assertRaises(BaseException):entry.execute(root,spec,plan)
                else: entry.execute(root,spec,plan)
            status=json.loads((root/'status.json').read_text())
        self.assertEqual(writers,{owner})
        return calls,status
    def test_parallel_release_and_artifact_barrier(self):
        calls,status=self.exercise(); self.assertEqual(status['status'],'PASSED')
        launch=calls.index('scripts/c/c-container.sh')
        for p in ('scripts/a/a-stop.sh','scripts/b/b-stop.sh','collect'): self.assertLess(calls.index(p),launch)
        self.assertEqual([e['status'] for e in status['timeline'] if e['kind']=='stage'],['RUNNING','PASSED','RUNNING','PASSED'])
    def test_failed_branch_blocks_successor(self):
        calls,status=self.exercise('branch');self.assertNotIn('scripts/c/c-container.sh',calls);self.assertEqual(status['cases']['c'],'NOT_RUN')
    def test_release_failure_blocks_successor(self):
        calls,status=self.exercise('release');self.assertNotIn('scripts/c/c-container.sh',calls);self.assertEqual(status['status'],'FAILED')
    def test_missing_artifacts_blocks_successor(self):
        calls,status=self.exercise('artifacts');self.assertNotIn('scripts/c/c-container.sh',calls);self.assertEqual(status['status'],'FAILED')
    def test_interrupt_releases_and_blocks_successor(self):
        calls,status=self.exercise('interrupt');self.assertNotIn('scripts/c/c-container.sh',calls);self.assertNotIn('RUNNING',status['cases'].values())
        self.assertIn('scripts/a/a-stop.sh',calls);self.assertIn('scripts/b/b-stop.sh',calls)
    def test_duplicate_coordinator_rejected(self):
        spec,plan=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.runtime').mkdir();(root/'.runtime/task.lock').write_text('123')
            with patch.object(entry,'invoke') as invoke:
                with self.assertRaises(FileExistsError):entry.execute(root,spec,plan)
                invoke.assert_not_called()
    def test_serial_cases_reuse_then_redeploy(self):
        spec,plan=fixture()
        spec['sub_tasks']=[{'id':x,'deployment_id':'a','case_id':'case','after_case':'keep_service' if x=='a' else 'stop_container','on_failure':'record_only'} for x in 'abc']
        spec['stages']=[{'id':'first','branches':[{'id':'serial','sub_tasks':['a','b']}]},{'id':'second','branches':[{'id':'serial','sub_tasks':['c']}]}]
        plan['deployments']={'a':plan['deployments']['a']}
        for x in 'abc': plan['tests'][x]['node']='a'
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.runtime').mkdir();(root/'.runtime/manifest.json').write_text('{"files":{}}')
            def invoke(group,path,*args,**kwargs): calls.append((path,args))
            with patch.object(entry,'distribute'),patch.object(entry,'claim_execution'),patch.object(entry,'collect'),patch.object(entry,'invoke',side_effect=invoke):entry.execute(root,spec,plan)
        starts=[args for path,args in calls if path.endswith('-start.sh')]
        self.assertEqual(starts,[('run01',),('run02',)])
        self.assertEqual([path for path,args in calls if path.startswith('scripts/tests')],['scripts/tests/a.sh','scripts/tests/b.sh','scripts/tests/c.sh'])

    def test_resource_and_task_validation(self):
        spec,_=fixture();spec['deployments']=[{'id':x,'instances':[{'placement':{'node':x}}]} for x in 'abc']
        self.assertEqual(validate_stages(spec),[])
        spec['deployments'][1]['instances'][0]['placement']['node']='a'
        self.assertIn('Parallel branches share physical nodes',validate_stages(spec))
        spec['stages'][1]['branches'][0]['sub_tasks']=['a']
        self.assertIn('Stages must reference every sub_task exactly once',validate_stages(spec))

if __name__=='__main__':unittest.main()
