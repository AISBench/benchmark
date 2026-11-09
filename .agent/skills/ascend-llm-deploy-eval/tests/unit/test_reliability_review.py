"""Fault injection for collection, finalization, proxy and ownership review fixes."""
import contextlib
import io
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from test_compact_entry import inputs
from test_staged_entry import fixture
import run as entry
import container_dependencies as deps
import ownership_evidence as owners
from task_model import logging_assignments, validate_task


class ReliabilityReviewTests(unittest.TestCase):
    def test_collection_shared_local_and_remote_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.runtime').mkdir()
            group={'host':'remote','root':str(root)}
            with patch.object(entry,'transport_argv',side_effect=lambda h,a:a), patch.object(entry.subprocess,'run',side_effect=lambda argv,**kw:Mock(returncode=0,stdout=Path(argv[1]).read_text())):
                self.assertEqual(entry.collection_access(root,group),'shared')
            self.assertEqual(list((root/'.runtime').glob('collect-*')),[])
            # Identical path strings alone do not imply shared storage.
            with patch.object(entry,'transport_argv',return_value=['ssh']), patch.object(entry.subprocess,'run',return_value=Mock(returncode=1)), patch.object(entry,'is_local',return_value=False):
                self.assertEqual(entry.collection_access(root,group),'ssh')
            with patch.object(entry,'transport_argv',return_value=['local-wrapper']), patch.object(entry.subprocess,'run',return_value=Mock(returncode=1)), patch.object(entry,'is_local',return_value=True):
                self.assertEqual(entry.collection_access(root,group),'local')
            source=root/'separate';(source/'logs').mkdir(parents=True);(source/'logs/real.log').write_text('artifact')
            (root/'status.json').write_text('{"expected_artifacts":["logs/real.log"]}')
            plan={'deployments':{'a':{'groups':{'a':{'host':'same','root':str(source)}}}}}
            with patch.object(entry,'collection_access',return_value='local'), patch.object(entry,'transport_argv',return_value=['wrapper']), patch.object(entry.subprocess,'run',side_effect=[Mock(returncode=0),Mock(returncode=1)]):
                entry.collect(root,plan)
            self.assertEqual((root/'logs/real.log').read_text(),'artifact')

    def test_original_failure_survives_publish_and_unlock_errors(self):
        for original in (ValueError('original task error'),KeyboardInterrupt('user interruption')):
            with self.subTest(error=type(original).__name__),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'.runtime').mkdir()
                spec,plan=fixture();real_save=entry.save;real_unlink=Path.unlink
                def save(path,value):
                    if path.name=='status.json':raise OSError('publish failure')
                    return real_save(path,value)
                def unlink(path,*args,**kw):
                    if path.name=='task.lock':raise PermissionError('unlock failure')
                    return real_unlink(path,*args,**kw)
                with patch.object(entry,'distribute',side_effect=original),patch.object(entry,'collect') as collect,patch.object(entry,'save',side_effect=save),patch.object(Path,'unlink',unlink):
                    with self.assertRaises(type(original)) as raised:
                        entry.execute(root,spec,plan)
                    self.assertIs(raised.exception,original)
                    collect.assert_called_once()
                report=json.loads((root/'.runtime/finalization.json').read_text())
                self.assertIn(str(original),report['primary_error'])
                self.assertEqual(report['steps']['unlock']['status'],'FAILED')
                self.assertEqual(report['steps']['publish']['status'],'FAILED')
                self.assertEqual(report['status'],'INTERRUPTED' if isinstance(original,KeyboardInterrupt) else 'FAILED')

    def test_publish_failure_still_unlocks_and_does_not_report_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.runtime').mkdir();lock=Mock()
            status={'status':'PASSED','release':{},'cases':{},'expected_artifacts':[]}
            with self.assertRaisesRegex(RuntimeError,'finalization failed'):
                entry.finish_execution(root,status,[],Mock(),Mock(),Mock(side_effect=OSError('publish')),lock,None)
            lock.unlink.assert_called_once()
            self.assertEqual(status['status'],'FAILED')
            self.assertEqual(json.loads((root/'.runtime/finalization.json').read_text())['steps']['unlock']['status'],'PASSED')

    def test_logging_mentions_and_reference_exemption(self):
        key='ASCEND_GLOBAL_LOG_LEVEL'
        text='# '+key+'=1\necho "'+key+'=1"\ngrep "'+key+'=" file\nexport '+key+'=1\n'+key+'=2 command\n'
        self.assertEqual(list(logging_assignments(text,[key])),[(key,4),(key,5)])
        spec,_=inputs();inst=spec['deployments'][0]['instances'][0]
        inst['launch']['bootstrap']=text;inst['launch']['preserve_reference']=False
        errors=validate_task(spec)
        self.assertTrue(any('bootstrap line 4' in x for x in errors))
        inst['launch']['preserve_reference']=True
        self.assertFalse(any('operator logging' in x for x in validate_task(spec)))

    def test_owner_pid_labels_limits_and_disappeared_process(self):
        self.assertEqual(owners.owner_pids('/dev/davinci0: 123 456\n/dev/davinci3: 789m\nnoise 3\n0\n'),[123,456,789])
        with tempfile.TemporaryDirectory() as tmp,patch.object(owners,'command',return_value={'returncode':0,'stdout':' '.join(str(i) for i in range(1000,1300))}):
            report=owners.snapshot([0,3],Path(tmp))
        self.assertEqual(report['pid_selection'],{'limit':256,'parsed':300,'discarded':44})
        self.assertEqual(len(report['processes']),256)
        self.assertEqual(report['processes'][0]['error'],'FileNotFoundError')

    def test_proxy_gate_data_does_not_rewrite_program(self):
        url='http://host/TIMEOUT/URLS?text="quoted"'
        first=shlex.split(entry.proxy_health_gate([url],3))
        second=shlex.split(entry.proxy_health_gate(['http://other/'],5))
        self.assertEqual(first[2],second[2])
        self.assertEqual(json.loads(first[3]),[url])
        self.assertEqual(first[4],'3')
        compile(first[2],'<proxy gate>','exec')

    def test_proxy_marker_trailing_comments_continuation_and_cli_errors(self):
        spec,bindings=inputs();dep=spec['deployments'][0]
        proxy=next(i for i in dep['instances'] if i['role']=='proxy')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'scripts'/dep['id'];folder.mkdir(parents=True)
            program=root/'proxy with spaces.py';program.write_text('must not run')
            script=folder/('node_a-'+proxy['id']+'.sh')
            script.write_text('exec python3 /wrong.py\n# @proxy-command\n\n# comment\nexec python3 \\\n'+shlex.quote(str(program))+'\n\n# trailing comment\n')
            with contextlib.redirect_stdout(io.StringIO()):deps.proxy_check(root,dep,'node_a')
            script.write_text('# @proxy-command\nwrong command\n')
            with self.assertRaisesRegex(ValueError,'wrong command'):deps.proxy_check(root,dep,'node_a')
        for args in ([],['/task','dep','node'],['/task','dep','node','0']):
            with patch.object(sys,'argv',['helper',*args]),contextlib.redirect_stderr(io.StringIO()),patch.object(deps.subprocess,'Popen') as popen:
                with self.assertRaises(SystemExit) as raised:deps.main()
                self.assertEqual(raised.exception.code,2);popen.assert_not_called()
