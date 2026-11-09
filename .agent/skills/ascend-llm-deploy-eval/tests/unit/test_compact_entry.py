import copy
import json
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch, Mock

PACKAGE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PACKAGE/'scripts'))
import run as entry
from plan_run import normalize, validate
from aisbench_cli import commands


def inputs():
    spec=json.loads((PACKAGE/'examples/job_spec.serial.json').read_text())
    spec['state']='ready';spec['unresolved']=[]
    spec['execution']={'max_retries':0,'startup_timeout_seconds':1,'smoke':'default','smoke_timeout_seconds':1}
    spec['cases']=spec['cases'][:1]
    spec['sub_tasks']=spec['sub_tasks'][:1]
    spec['sub_tasks'][0]['case_id']=spec['cases'][0]['id']
    spec['deployments']=spec['deployments'][:1]
    dep=spec['deployments'][0];spec['sub_tasks'][0]['deployment_id']=dep['id']
    ports={};number=18000
    for instance in dep['instances']:
        instance['launch']['bootstrap']=''
        if instance['role']!='proxy':instance['launch']['engine_id']='0'
        ports[instance['id']]={}
        for name,width in instance['ports'].items():
            ports[instance['id']][name]=number;number+=width+1
    bindings={'nodes':{'node_a':{'host':'root@192.0.2.1','work_dir':'/tasks',
        'model_paths':{'dsv4':'/models/dsv4'},'paths':{'proxy_program':'/proxy.py','comm':'/etc/hixlep'},
        'deployments':{dep['id']:{'network':{'ip':'192.0.2.1','interface':'eth0'},'ports':ports}},
        'benchmark':{'datasets':{spec['cases'][0].get('dataset_id','default'):{'path':'/data/test.jsonl'}}},'mounts':[]}}}
    return spec,bindings


class CompactTests(unittest.TestCase):
    def test_cache_reset_api_default_and_opt_out(self):
        spec,bindings=inputs()
        for instance in spec['deployments'][0]['instances']:
            instance['launch'].setdefault('env',{}).pop('VLLM_SERVER_DEV_MODE',None)
        for enabled in (True,False):
            candidate=copy.deepcopy(spec)
            if not enabled:candidate['execution']['enable_cache_reset_api']=False
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)/'job';normalized,_=entry.generate(candidate,bindings,root)
                self.assertEqual(normalized['execution']['enable_cache_reset_api'],enabled)
                for instance in candidate['deployments'][0]['instances']:
                    text=(root/'scripts'/candidate['deployments'][0]['id']/('node_a-'+instance['id']+'.sh')).read_text()
                    self.assertEqual('export VLLM_SERVER_DEV_MODE=1' in text,enabled and instance['role']!='proxy')
                self.assertFalse(normalized['cases'][0]['reset_cache'])
        candidate=copy.deepcopy(spec);candidate['execution']['enable_cache_reset_api']=False
        candidate['deployments'][0]['instances'][0]['launch']['env']['VLLM_SERVER_DEV_MODE']='1'
        self.assertEqual(normalize(candidate)['deployments'][0]['instances'][0]['launch']['env']['VLLM_SERVER_DEV_MODE'],'1')

    def test_distribution_shared_path_reuses_and_changed_bundle_rejected(self):
        spec,bindings=inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job';_,plan=entry.generate(spec,bindings,root)
            remote=Path(tmp)/'remote'
            group=next(iter(plan['deployments'].values()))['groups']['node_a']
            group['root']=str(remote)
            groups=next(iter(plan['deployments'].values()))['groups']
            groups['alias']={**group,'host':'root@192.0.2.2'}
            def transport(host,argv):return [sys.executable,*argv[1:]]
            with patch.object(entry,'transport_argv',side_effect=transport):
                entry.distribute(root,plan)
                entry.distribute(root,plan)
                self.assertEqual((remote/'job_spec.json').read_bytes(),(root/'job_spec.json').read_bytes())
                (remote/'job_spec.json').write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'distribution failed'):entry.distribute(root,plan)

    def test_native_results_multiple_groups_and_request_failure(self):
        from aisbench_cli import validate_native_result
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'execution.json').write_text('{"returncode":0}')
            for name in ('model-a','model-b'):
                file=root/'outputs/time/performances'/name/'data_details.jsonl';file.parent.mkdir(parents=True)
                file.write_text('{"id":0,"success":true}\n')
            log=root/'outputs/time/logs/worker.log';log.parent.mkdir(parents=True);log.write_text('done')
            validate_native_result(root,{'formal_requests':1})
            self.assertEqual(json.loads((root/'validated.json').read_text())['request_groups'],2)
            file.write_text('{"id":0,"success":false}\n')
            with self.assertRaisesRegex(ValueError,'Failed'):validate_native_result(root,{'formal_requests':1})

    def test_shell_syntax(self):
        bash=Path('C:/Program Files/Git/bin/bash.exe')
        if not bash.exists():
            import shutil
            bash=shutil.which('bash')
        if not bash:self.skipTest('Bash unavailable')
        spec,bindings=inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job';entry.generate(spec,bindings,root)
            for file in root.rglob('*.sh'):
                result=subprocess.run([str(bash),'-n',str(file)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)

    def test_native_config_retains_adapter_and_task_load(self):
        spec,bindings=inputs();spec['cases'][0]['aisbench_config']='custom'
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'custom.py';config.write_text('models = [dict(type="native")]\ndatasets = []\n')
            bindings['nodes']['node_a']['benchmark']['configs']={'custom':str(config)}
            root=Path(tmp)/'job';entry.generate(spec,bindings,root)
            result=next((root/'scripts/tests').glob('*.config.py')).read_text()
            self.assertIn('type="native"',result)
            self.assertNotIn('_task_model.update',result)
            self.assertEqual(result,config.read_text())

    def test_full_success_reuses_deployment(self):
        spec,bindings=inputs();second=copy.deepcopy(spec['sub_tasks'][0]);second['id']='second'
        spec['sub_tasks'][0]['after_case']='keep_service';spec['sub_tasks'].append(second)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job';spec,plan=entry.generate(spec,bindings,root)
            manifest=json.loads((root/'.runtime/manifest.json').read_text())['files']
            def remote(argv,**kwargs):
                name=next(n for n in manifest if n in str(argv));return Mock(returncode=0,stdout=manifest[name]+' file')
            response=Mock();response.__enter__=Mock(return_value=Mock(status=200));response.__exit__=Mock(return_value=False)
            opener=Mock();opener.open.return_value=response
            with patch.object(entry,'transport_argv',side_effect=lambda host,argv:argv),patch.object(entry,'distribute'),patch.object(entry,'claim_execution'),patch.object(entry,'collect'),patch.object(entry,'invoke') as invoke,patch.object(entry.subprocess,'run',side_effect=remote),patch.object(entry.urllib.request,'build_opener',return_value=opener):
                entry.execute(root,spec,plan)
            state=json.loads((root/'status.json').read_text())
            self.assertEqual(state['status'],'PASSED')
            self.assertEqual(sum(c.args[1].endswith('-container.sh') for c in invoke.call_args_list),1)
            self.assertEqual(sum(c.args[1].startswith('scripts/tests/') for c in invoke.call_args_list),2)

    def test_generation_no_process_and_deterministic(self):
        spec,bindings=inputs()
        with tempfile.TemporaryDirectory() as tmp,patch.object(entry.subprocess,'run',side_effect=AssertionError('side effect')):
            a=Path(tmp)/'a';b=Path(tmp)/'b'
            entry.generate(spec,bindings,a);entry.generate(spec,bindings,b)
            for path in (a/'scripts').rglob('*'):
                if path.is_file():self.assertEqual(path.read_bytes(),(b/path.relative_to(a)).read_bytes())
            self.assertEqual(json.loads((a/'status.json').read_text())['status'],'SCRIPTS_READY')
            self.assertFalse((a/'common').exists())
            scripts='\n'.join(p.read_text() for p in (a/'scripts').rglob('*.sh'))
            self.assertIn('ais_bench',scripts);self.assertIn('--mode perf',scripts)
            self.assertIn('ASCEND_RT_VISIBLE_DEVICES',scripts)
            self.assertNotIn('--device-ids',scripts)
            with self.assertRaisesRegex(ValueError,'overwrite'):entry.generate(spec,bindings,a)

    def test_accuracy_reserved_without_perf_fields(self):
        spec,bindings=inputs();spec['cases']=[{'id':spec['cases'][0]['id'],'mode':'accuracy'}]
        self.assertEqual(validate(normalize(spec)),[])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job'
            with self.assertRaisesRegex(ValueError,'reserved'):entry.generate(spec,bindings,root)
            self.assertFalse(root.exists())
        with self.assertRaisesRegex(ValueError,'reserved'):commands('ais_bench','config',{'mode':'accuracy'},Path('.'))

    def test_preserve_logging_env(self):
        spec,bindings=inputs()
        spec['deployments'][0]['instances'][0]['launch'].setdefault('env',{})['ASCEND_GLOBAL_LOG_LEVEL']='1'
        with tempfile.TemporaryDirectory() as tmp:
            entry.generate(spec,bindings,Path(tmp)/'job')
            text='\n'.join(p.read_text() for p in (Path(tmp)/'job/scripts').rglob('*.sh'))
            self.assertIn('export ASCEND_GLOBAL_LOG_LEVEL=1',text)

    def test_full_failure_releases_owned_resources_and_marks_case(self):
        spec,bindings=inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job';spec,plan=entry.generate(spec,bindings,root)
            digest=json.loads((root/'.runtime/manifest.json').read_text())['files']
            def remote(argv,**kwargs):
                # SHA verification only; deployment operations use mocked invoke.
                relative=next(name for name in digest if name in str(argv))
                return Mock(returncode=0,stdout=digest[relative]+'  file',stderr='')
            calls=[]
            def invoke(group,relative,*args,**kwargs):
                calls.append(relative)
                if relative.endswith('-start.sh'):raise RuntimeError('start failed')
            with patch.object(entry,'transport_argv',side_effect=lambda host,argv:argv),patch.object(entry,'distribute'),patch.object(entry,'claim_execution'),patch.object(entry,'collect'),patch.object(entry,'invoke',side_effect=invoke),patch.object(entry.subprocess,'run',side_effect=remote):
                entry.execute(root,spec,plan)
            state=json.loads((root/'status.json').read_text())
            self.assertEqual(state['status'],'FAILED')
            self.assertIn('FAILED',state['cases'].values())
            self.assertTrue(any(p.endswith('-stop.sh') for p in calls))
            self.assertFalse((root/'.runtime/task.lock').exists())

    def test_interrupt_does_not_leave_running_case(self):
        spec,bindings=inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job';spec,plan=entry.generate(spec,bindings,root)
            manifest=json.loads((root/'.runtime/manifest.json').read_text())['files']
            def remote(argv,**kwargs):
                name=next(n for n in manifest if n in str(argv))
                return Mock(returncode=0,stdout=manifest[name]+' file')
            def invoke(group,relative,*args,**kwargs):
                if relative.endswith('-start.sh'):raise KeyboardInterrupt()
            with patch.object(entry,'transport_argv',side_effect=lambda host,argv:argv),patch.object(entry,'distribute'),patch.object(entry,'claim_execution'),patch.object(entry,'collect'),patch.object(entry,'invoke',side_effect=invoke),patch.object(entry.subprocess,'run',side_effect=remote):
                with self.assertRaises(KeyboardInterrupt):entry.execute(root,spec,plan)
            state=json.loads((root/'status.json').read_text())
            self.assertEqual(state['status'],'INTERRUPTED');self.assertNotIn('RUNNING',state['cases'].values())


if __name__=='__main__':unittest.main()
