import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cluster_runtime
import preflight

class ExecutionPolicyTests(unittest.TestCase):
    def test_failure_releases_entire_deployment(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(cluster_runtime,'release_all',return_value={'node':{'status':'PASSED'}}) as release:
                result=cluster_runtime.reconcile_failure(Path(tmp),{},'case')
                self.assertTrue(result['continue'])
                release.assert_called_once_with({},'stop_container','failed_sub_task')
            with patch.object(cluster_runtime,'release_all',return_value={'node':{'status':'FAILED'}}):
                self.assertFalse(cluster_runtime.reconcile_failure(Path(tmp),{},'case')['continue'])

    def test_preflight_collects_independent_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'scripts').mkdir()
            spec={'deployment':{'service':{'npu_ids':[0],'ports':[]}},'target':{'image':'missing'},
                'container':{'mounts':[{'source':str(root/'missing')}],'devices':[str(root/'absent')]},
                'model':{'path':str(root/'weights')},'benchmark':{'tool_dir':str(root/'tool')}}
            with patch.object(preflight,'read_plan',return_value=spec),patch.object(preflight,'command',side_effect=RuntimeError('missing')),patch.object(preflight,'ready',return_value=False),patch.object(preflight,'state',return_value={'service':'ABSENT'}):
                with self.assertRaises(ValueError): preflight.check(root)
            report=json.loads((root/'preflight-report.json').read_text())
            self.assertNotIn('NOT_RUN',report['checks'].values())
            for name in ['image','mounts','devices','weights','tokenizer','benchmark_tool','resource_or_reuse']:
                self.assertEqual(report['checks'][name],'FAILED')

if __name__=='__main__': unittest.main()
