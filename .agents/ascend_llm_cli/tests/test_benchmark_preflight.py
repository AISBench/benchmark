from pathlib import Path
import sys
import tempfile
import unittest
import json
import threading
import time
from unittest.mock import Mock,patch
from deploy_eval_v2.benchmark_preflight import inspect_environment
from deploy_eval_v2.protocol import ProtocolError
from deploy_eval_v2.process_guard import run_logged


class BenchmarkPreflightTests(unittest.TestCase):
    def test_real_preflight_child_fails_without_benchmark_entry_point(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            request=root/'request.json'
            request.write_text(json.dumps(dict(tester=dict(python=sys.executable,
                aisbench=str(root/'missing-ais-bench'),model_path=str(root),datasets={}),
                case={'formal_requests':1},config=str(root/'config.py'))))
            result=root/'result.json'
            log=root/'preflight.log'
            code=run_logged([sys.executable,'-B','-m','deploy_eval_v2.benchmark_preflight',
                             '--request',str(request),'--result',str(result)],log,
                             time.monotonic()+5,threading.Event())
            self.assertNotEqual(code,0)
            self.assertFalse(result.exists())
            self.assertIn('entry point',log.read_text(encoding='utf-8'))

    def setup_files(self,folder):
        root=Path(folder)
        executable=root/'ais_bench';executable.write_text('# test executable');executable.chmod(0o700)
        return dict(aisbench=str(executable),python=sys.executable,model_path=str(root),datasets={})

    def test_entry_point_owner_and_synthetic_capability_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            tester=self.setup_files(folder)
            distribution=Mock(metadata={'Name':'ais_bench'},version='test-version',
                              entry_points=[Mock(group='console_scripts',name='ais_bench')])
            # Mock.name is a Mock label, so assign the actual entry point attribute.
            distribution.entry_points[0].name='ais_bench'
            with patch('deploy_eval_v2.benchmark_preflight.sysconfig.get_path',return_value=folder),patch('deploy_eval_v2.benchmark_preflight.importlib.metadata.distributions',return_value=[distribution]),patch('deploy_eval_v2.benchmark_preflight.aisbench_cli.dataset_for',return_value={'type':'aisbench_synthetic'}),patch('deploy_eval_v2.benchmark_preflight.aisbench_cli.check_synthetic_support',return_value={'status':'CONFIG_CHECKED'}) as check:
                result=inspect_environment(tester,{'formal_requests':17})
                self.assertEqual(result['status'],'VERIFIED')
                self.assertEqual(result['distributions'][0]['version'],'test-version')
                check.assert_called_once_with({'formal_requests':17})

    def test_missing_distribution_fails_before_dataset_work(self):
        with tempfile.TemporaryDirectory() as folder:
            tester=self.setup_files(folder)
            with patch('deploy_eval_v2.benchmark_preflight.sysconfig.get_path',return_value=folder),patch('deploy_eval_v2.benchmark_preflight.importlib.metadata.distributions',return_value=[]),patch('deploy_eval_v2.benchmark_preflight.aisbench_cli.dataset_for') as dataset:
                with self.assertRaisesRegex(ProtocolError,'ownership'):
                    inspect_environment(tester,{'formal_requests':17})
                dataset.assert_not_called()
