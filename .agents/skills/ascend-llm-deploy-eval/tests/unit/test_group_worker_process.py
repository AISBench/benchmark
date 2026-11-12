"""Retired entry rejection and historical evidence checks; no Docker/NPU needed."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import group_worker
from group_worker import prepared, terminal
from stage_runtime import digest


@unittest.skipUnless(os.name == 'posix', 'Detached host worker requires Linux')
class WorkerProcessTests(unittest.TestCase):
    def test_retired_worker_rejects_execution_and_terminal_evidence_remains_strict(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'logs').mkdir()
            spec={'schema_version':'3.0','job_id':'20260101-997','cleanup':{'task_end':'stop_container'},
                  'deployments':[{'id':'dep'}],'sub_tasks':[{'id':'s1'}]}
            sha=digest(spec)
            (root/'common').mkdir()
            (root/'common/plan.json').write_text(json.dumps({'job_spec':spec,'job_spec_sha256':sha,'sequence':spec['sub_tasks']}))
            (root/'resolved-cluster.json').write_text(json.dumps({'plan_sha256':sha,'deployments':{'dep':{'groups':{'node':{
                'host':'node.example','run_dir':str(root),'script':str(root/'node_rpc.py')}}}}}))
            child='''import json,time
from pathlib import Path
root=Path(__file__).parent
with (root/'starts').open('a') as stream: stream.write('started\\n')
time.sleep(0.4)
(root/'sub_tasks/s1').mkdir(parents=True)
(root/'sub_tasks/s1/result.json').write_text(json.dumps({'status':'PASSED'}))
(root/'release-results.json').write_text(json.dumps({'dep':{'node':{'status':'PASSED'}}}))
(root/'summary.json').write_text(json.dumps({'status':'PASSED','sub_tasks':{'s1':{'status':'PASSED'}},'cleanup':{'status':'PASSED'}}))
'''
            (root/'synthetic.py').write_text(child)
            import shlex
            (root/'run.sh').write_text('exec '+shlex.join([sys.executable,str(root/'synthetic.py')])+'\n')
            # Retirement is intentional: never re-enable the old public coordinator.
            result=subprocess.run([sys.executable,str(Path(group_worker.__file__)),
                                   '--execute',str(root),'--sha256',sha],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Retired lifecycle entry',result.stderr)
            self.assertFalse((root/'starts').exists())
            self.assertFalse((root/'group-worker.json').exists())
            # Retained historical evidence readers are tested without launching the retired wrapper.
            subprocess.run([sys.executable,str(root/'synthetic.py')],check=True,timeout=10)
            parsed,resolved,_,_=prepared(root,sha)
            observed=terminal(root,parsed,resolved)
            self.assertEqual(observed['state'],'DONE',observed)
            self.assertTrue(observed['cleanup_verified'])
            self.assertEqual((root/'starts').read_text(),'started\n')
            # A summary is insufficient if any registered node lacks release evidence.
            (root/'release-results.json').write_text('{}')
            self.assertEqual(terminal(root,parsed,resolved)['state'],'BLOCKED')


if __name__ == '__main__': unittest.main()
