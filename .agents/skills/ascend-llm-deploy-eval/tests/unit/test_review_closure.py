import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from test_staged_entry import fixture
from test_compact_entry import inputs
import run as entry
import aisbench_cli
import task_preflight
from metric_evidence import capture, summarize

PACKAGE=Path(__file__).resolve().parents[2]


class ReviewClosureTests(unittest.TestCase):
    def test_retired_lifecycle_entrypoints(self):
        names=('preflight','deploy','evaluate','finalize','stage_runtime','cluster_runtime','plan_run','submit_job','run_task','node_rpc','node_prepare','group_run','group_worker','submission','service_process','case_runner')
        for name in names:
            with self.subTest(name=name):
                result=subprocess.run([sys.executable,str(PACKAGE/'scripts'/(name+'.py')),'--help'],capture_output=True,text=True,timeout=10)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('Retired',result.stderr)
        import stage_runtime
        with self.assertRaisesRegex(SystemExit,'Retired'):stage_runtime.entry('preflight')

    def test_process_check_is_non_destructive(self):
        entry.check_process(os.getpid())
        with patch.object(entry.os, 'name', 'nt'), patch.object(entry.os, 'kill') as kill:
            with patch.object(entry.subprocess, 'run', return_value=Mock(returncode=0, stdout='"python.exe","123","Console","1","1 K"')):
                entry.check_process(123)
                with self.assertRaises(ProcessLookupError): entry.check_process(124)
            kill.assert_not_called()

    def test_lock_diagnostics_preserve_existing_lock(self):
        for content, effect, expected in ((str(os.getpid()),None,'exists'),('987654321',ProcessLookupError(),'stale'),('bad',None,'invalid PID')):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'.runtime').mkdir();lock=root/'.runtime/task.lock';lock.write_text(content)
                spec,plan=fixture()
                with patch.object(entry,'check_process',side_effect=effect),patch.object(entry,'distribute') as distribute:
                    with self.assertRaisesRegex(FileExistsError,expected):entry.execute(root,spec,plan)
                    distribute.assert_not_called()
                self.assertEqual(lock.read_text(),content)

    def test_retry_classification_in_coordinator(self):
        for failure_at,error,expected_tests,expected_creates in (
            ('test',entry.ScriptFailure('tests/a.sh',1),1,1),
            ('container',entry.ScriptFailure('container.sh',1),0,1),
            ('container',entry.ScriptFailure('container.sh',255),1,2),
            ('container',subprocess.TimeoutExpired('ssh',1),1,2)):
            with self.subTest(failure_at=failure_at,error=error),tempfile.TemporaryDirectory() as tmp:
                spec,plan=fixture();spec.pop('stages');spec['sub_tasks']=spec['sub_tasks'][:1]
                spec['execution']['max_retries']=2;plan['deployments']={'a':plan['deployments']['a']}
                root=Path(tmp);(root/'.runtime').mkdir();(root/'.runtime/manifest.json').write_text('{"files":{}}')
                calls=[];failed=False
                def invoke(group,path,*args,**kwargs):
                    nonlocal failed
                    calls.append(path)
                    target=path=='scripts/tests/a.sh' if failure_at=='test' else path.endswith('-container.sh')
                    if target and not failed:failed=True;raise error
                with patch.object(entry,'invoke',side_effect=invoke),patch.object(entry,'distribute'),patch.object(entry,'claim_execution'),patch.object(entry,'collect'):
                    entry.execute(root,spec,plan)
                status=json.loads((root/'status.json').read_text())
                self.assertEqual(calls.count('scripts/tests/a.sh'),expected_tests)
                self.assertEqual(calls.count('scripts/a/a-container.sh'),expected_creates)
                self.assertEqual(status['attempt_failures'][0]['retrying'],expected_creates==2)
                self.assertEqual(status['status'],'PASSED' if expected_creates==2 else 'FAILED')
        self.assertFalse(entry.retry_decision(subprocess.TimeoutExpired('test',1),'test')[0])
        self.assertFalse(entry.retry_decision(ValueError('bad binding'),'deployment')[0])

    def test_mixed_metrics_and_reset(self):
        instances=[{'id':'mixed','role':'mixed','url':'local'}]
        def raw(q,h):return f'vllm:prefix_cache_queries_total {q}\nvllm:prefix_cache_hits_total {h}\n'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);capture(instances,root,'before',lambda _:raw(10,5));capture(instances,root,'after',lambda _:raw(30,15))
            self.assertEqual(summarize(root)['hit_ratio'],0.5)
            capture(instances,root,'after',lambda _:raw(1,0))
            self.assertEqual(summarize(root)['status'],'UNVERIFIED')

    def test_digest_uses_bounded_reads(self):
        data=b'abc'*(1024*1024)
        class Bounded(io.BytesIO):
            def read(self,size=-1):
                self_outer.assertGreater(size,0);self_outer.assertLessEqual(size,1024*1024)
                return super().read(size)
        self_outer=self;path=Mock();path.open.return_value=Bounded(data)
        self.assertEqual(aisbench_cli.dataset_digest(path),hashlib.sha256(data).hexdigest())
        path.read_bytes.assert_not_called()

    def test_result_source_matches_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);detail=root/'outputs/time/performances/model/data_details.jsonl';detail.parent.mkdir(parents=True)
            detail.write_text(json.dumps({'id':0,'success':True})+'\n')
            log=root/'outputs/time/logs/worker.log';log.parent.mkdir();log.write_text('done')
            (root/'execution.json').write_text('{"returncode":0}')
            for native in (False,True):
                case={'formal_requests':1}
                if native:case['aisbench_config']='config.py'
                aisbench_cli.validate_native_result(root,case)
                self.assertEqual(json.loads((root/'validated.json').read_text())['config'],'native' if native else 'generated')

    def test_probe_pins_image_id_and_checks_cleanup_identity(self):
        for foreign in (False,True):
            container={'Id':'container-id','Image':'other' if foreign else 'sha256:verified','Config':{'Labels':{'benchmark.job_id':'job','benchmark.role':'preflight-probe'}}}
            replies=[Mock(returncode=1),Mock(returncode=0,stdout=json.dumps([container])),Mock(returncode=1)]
            def command(argv,**kwargs):
                if argv[:3]==['docker','image','inspect']:return '[{"Id":"sha256:verified"}]'
                if argv[:2]==['docker','run']:
                    self.assertIn('sha256:verified',argv);raise RuntimeError('probe failed')
                return ''
            with patch.object(task_preflight.subprocess,'run',side_effect=replies),patch.object(task_preflight,'command',side_effect=command) as cmd:
                with self.assertRaisesRegex(ValueError if foreign else RuntimeError,'ownership' if foreign else 'probe failed'):
                    task_preflight.inspect_image_tools('registry/model:tag','job')
                removals=[c for c in cmd.call_args_list if c.args[0][:2]==['docker','rm']]
                self.assertEqual(len(removals),0 if foreign else 1)

    def test_instance_and_node_scripts_coexist(self):
        spec,bindings=inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job';entry.generate(spec,bindings,root)
            folder=root/'scripts'/spec['deployments'][0]['id']
            for kind in ('container','dependencies','start','probe','stop'):
                self.assertTrue((folder/('node_a-'+kind+'.sh')).exists())
            start=(folder/'node_a-start.sh').read_text()
            for instance in spec['deployments'][0]['instances']:
                filename='node_a-'+instance['id']+'.sh'
                self.assertTrue((folder/filename).exists());self.assertIn(filename,start)
