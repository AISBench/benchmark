import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from test_compact_entry import inputs
import run as entry
import container_dependencies as deps

class DependencyTests(unittest.TestCase):
    def test_selected_proxy_is_readable_without_execution(self):
        spec,binding=inputs()
        dep=spec['deployments'][0]
        proxy=next(i for i in dep['instances'] if i['role']=='proxy')
        proxy['launch']['proxy_program_key']='selected_proxy'
        proxy['launch']['bootstrap']='# unrelated exec must not be mistaken for the generated command\nif false; then\nexec python3 /not-selected.py\nfi'
        with tempfile.TemporaryDirectory() as d:
            program=Path(d)/'selected proxy.py'
            program.write_text('raise RuntimeError("must not execute")')
            binding['nodes']['node_a']['paths']['selected_proxy']='/selected proxy.py'
            root=Path(d)/'job';entry.generate(spec,binding,root)
            script=root/'scripts'/dep['id']/('node_a-'+proxy['id']+'.sh')
            script.write_text(script.read_text().replace("'/selected proxy.py'",__import__('shlex').quote(str(program))))
            with patch.object(deps.subprocess,'run') as run:
                deps.proxy_check(root,dep,'node_a')
                run.assert_not_called()
            program.unlink()
            with self.assertRaisesRegex(ValueError,'Proxy program unavailable'):
                deps.proxy_check(root,dep,'node_a')
            # Another node must not check a proxy it does not own.
            deps.proxy_check(root,dep,'other_node')

    def test_proxy_directory_and_read_permission_failure(self):
        spec,binding=inputs();dep=spec['deployments'][0]
        with tempfile.TemporaryDirectory() as d:
            program=Path(d)/'proxy.py';program.mkdir()
            binding['nodes']['node_a']['paths']['proxy_program']='/selected-proxy.py'
            root=Path(d)/'job';entry.generate(spec,binding,root)
            proxy=next(i for i in dep['instances'] if i['role']=='proxy')
            script=root/'scripts'/dep['id']/('node_a-'+proxy['id']+'.sh')
            script.write_text(script.read_text().replace('/selected-proxy.py',__import__('shlex').quote(str(program))))
            with self.assertRaisesRegex(ValueError,'Proxy program unavailable'):
                deps.proxy_check(root,dep,'node_a')
            program.rmdir();program.write_text('# proxy')
            original=Path.open
            def denied(path,*args,**kwargs):
                if path==program:raise PermissionError('denied')
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',denied):
                with self.assertRaisesRegex(ValueError,'Proxy program unavailable'):
                    deps.proxy_check(root,dep,'node_a')

    def test_failure_blocks_all_start_and_releases(self):
        spec,binding=inputs()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'job';spec,plan=entry.generate(spec,binding,root)
            self.assertEqual(spec['execution']['dependency_timeout_seconds'],120)
            dep=next(iter(plan['deployments'].values()))
            dep['groups']['second']=dict(next(iter(dep['groups'].values())),host='second')
            manifest=json.loads((root/'.runtime/manifest.json').read_text())['files'];calls=[]
            def remote(argv,**kw):
                name=next(n for n in manifest if n in str(argv));return Mock(returncode=0,stdout=manifest[name]+' file')
            def invoke(group,name,*args,**kw):
                calls.append(name)
                if name.endswith('second-dependencies.sh'):raise RuntimeError('driver unavailable')
            with patch.object(entry,'transport_argv',side_effect=lambda h,a:a),patch.object(entry,'distribute'),patch.object(entry,'claim_execution'),patch.object(entry,'collect'),patch.object(entry,'invoke',side_effect=invoke),patch.object(entry.subprocess,'run',side_effect=remote):
                entry.execute(root,spec,plan)
            self.assertFalse(any(n.endswith('-start.sh') for n in calls))
            self.assertEqual(sum(n.endswith('-container.sh') for n in calls),2)
            self.assertEqual(sum(n.endswith('-stop.sh') for n in calls),2)
            status=json.loads((root/'status.json').read_text());self.assertEqual(status['status'],'FAILED')
            self.assertTrue(any(n.endswith('second-dependencies.log') for n in status['expected_artifacts']))
    @unittest.skipIf(__import__('os').name=='nt','Linux process groups')
    def test_timeout_terminates_only_worker_process_group(self):
        import subprocess
        import signal
        process=Mock(pid=12345);process.wait.side_effect=[subprocess.TimeoutExpired('worker',1),-9]
        with patch.object(deps.sys,'argv',['helper','/task','dep','node','1']),patch.object(deps.subprocess,'Popen',return_value=process),patch.object(deps.os,'killpg',create=True) as kill:
            with self.assertRaises(TimeoutError):deps.main()
            kill.assert_called_once_with(12345,signal.SIGKILL if hasattr(signal,'SIGKILL') else 9)

    def test_no_bootstrap_and_environment_is_explicit(self):
        spec,binding=inputs();item=spec['deployments'][0]['instances'][0]
        item['launch']['bootstrap']='echo NON_IDEMPOTENT'
        item['launch']['env']={'LD_LIBRARY_PATH':'/confirmed/lib'}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'job';spec,plan=entry.generate(spec,binding,root)
            self.assertNotIn('NON_IDEMPOTENT',''.join(p.read_text() for p in root.glob('scripts/*/*-dependencies.sh')))
            data=json.loads((root/'.runtime/dependency-env.json').read_text())
            self.assertEqual(data[spec['deployments'][0]['id']]['node_a'][item['id']]['LD_LIBRARY_PATH'],'/confirmed/lib')
    def test_missing_executable_and_bad_dataset(self):
        spec={'cases':[{'id':'c','formal_requests':1}]};sub={'id':'s','case_id':'c'}
        binding={'benchmark':{'datasets':{}}}
        with patch.object(deps.shutil,'which',return_value=None):
            with self.assertRaisesRegex(ValueError,'executable unavailable'):deps.benchmark_check(Path('.'),spec,binding,sub)
        dist=Mock();dist.entry_points=[type('Entry',(),{'group':'console_scripts','name':'ais_bench'})()];dist.metadata={'Name':'ais_bench_benchmark'};dist.version='test'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.jsonl';p.write_text('{"question":"x"}\n')
            binding['benchmark']['datasets']={'default':{'path':str(p)}}
            with patch.object(deps.shutil,'which',return_value='/bin/ais_bench'),patch.object(deps.importlib.metadata,'distributions',return_value=[dist]):
                with self.assertRaisesRegex(ValueError,'columns'):deps.benchmark_check(Path(d),spec,binding,sub)
if __name__=='__main__':unittest.main()
