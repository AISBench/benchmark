"""Linux subprocess integration with synthetic job evidence; no Docker/NPU needed."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from group_worker import launch, status
from stage_runtime import digest


@unittest.skipUnless(os.name == 'posix', 'Detached host worker requires Linux')
class WorkerProcessTests(unittest.TestCase):
    def test_detached_worker_duplicate_launch_and_terminal_evidence(self):
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
            self.assertEqual(status(root,sha)['state'],'PENDING')
            launch(root,sha);launch(root,sha)
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                observed=status(root,sha)
                if observed['state']!='RUNNING': break
                time.sleep(.1)
            self.assertEqual(observed['state'],'DONE',observed)
            self.assertTrue(observed['cleanup_verified'])
            self.assertEqual((root/'starts').read_text(),'started\n')
            self.assertEqual(launch(root,sha)['state'],'DONE')
            # A summary is insufficient if any registered node lacks release evidence.
            (root/'release-results.json').write_text('{}')
            self.assertEqual(status(root,sha)['state'],'BLOCKED')


if __name__ == '__main__': unittest.main()
