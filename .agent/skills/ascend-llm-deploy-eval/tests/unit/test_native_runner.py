import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import case_runner

class RunnerTests(unittest.TestCase):
    def test_cli_failure_stops_sequence_and_records_cleanup(self):
        for exit_code in (0, 2):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as d:
                root=Path(d);(root/'scripts').mkdir();folder=root/'attempt';folder.mkdir()
                (root/'resolved.json').write_text(json.dumps({'benchmark':{'prefill_dp':1,'executable':'ais_bench'}}))
                (root/'scripts/service.json').write_text(json.dumps({'reset_urls':[]}))
                (folder/'case.json').write_text(json.dumps({'id':'c1','reset_cache':False,'formal_requests':2,'timeout_seconds':60}))
                process=Mock(pid=123);process.wait.return_value=exit_code
                with patch.object(case_runner,'write_config',return_value=folder/'scenario.json'), patch.object(case_runner,'reset_instances'), patch.object(case_runner,'ticks',return_value='identity'), patch.object(case_runner,'members',return_value=[]), patch.object(case_runner,'emit') as event, patch.object(case_runner,'validate',return_value={'status':'PASSED'}) as validate, patch.object(case_runner.subprocess,'Popen',return_value=process) as launch:
                    self.assertEqual(case_runner.run(root,folder),exit_code)
                    self.assertEqual(launch.call_count,1)
                    self.assertEqual(validate.call_count,1 if exit_code==0 else 0)
                    self.assertTrue(json.loads((folder/'cleanup.json').read_text())['released'])
                    if exit_code:
                        self.assertTrue(any(c.args[1:3]==('benchmark','fail') for c in event.call_args_list))
