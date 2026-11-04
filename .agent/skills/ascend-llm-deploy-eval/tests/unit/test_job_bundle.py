import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import submit_job
from stage_runtime import PACKAGE

class JobBundleTests(unittest.TestCase):
    def test_shared_destination_uploaded_once(self):
        with patch.object(submit_job, 'upload') as upload, patch.object(submit_job, 'ssh', side_effect=['', 'True', '']):
            submit_job.upload_destinations([('a','/work/job'),('b','/work/job')],Path('bundle'),[[('a','/work/job'),('b','/work/job')]])
            self.assertEqual(upload.call_count,1)

    def test_same_path_on_distinct_storage_uploaded_twice(self):
        with patch.object(submit_job, 'upload') as upload, patch.object(submit_job, 'ssh', return_value=''):
            submit_job.upload_destinations([('a','/work/job'),('b','/work/job')],Path('bundle'))
            self.assertEqual(upload.call_count,2)
    def test_single_host_bundle_and_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            spec=json.loads((PACKAGE/'examples/job_spec.serial.json').read_text(encoding='utf-8'))
            spec.pop('job_id',None);spec['state']='ready';spec['unresolved']=[]
            bindings=json.loads((PACKAGE/'examples/bindings.example.json').read_text(encoding='utf-8'))
            bindings['nodes']['node_a'].update(host='root@example',work_dir='/work')
            for name,data in [('spec',spec),('bindings',bindings)]: (tmp/(name+'.json')).write_text(json.dumps(data))
            args=(tmp/'spec.json',tmp/'bindings.json',tmp/'jobs','20261103','root@example','/work')
            root,archive,destinations=submit_job.build(*args)
            self.assertEqual(destinations,[('root@example','/work/'+root.name)])
            with tarfile.open(archive) as tar:
                names=tar.getnames()
            self.assertIn('run.sh',names)
            self.assertIn('common/plan.json', names)
            self.assertIn('common/inputs/bindings.json', names)
            self.assertNotIn('plan.json', names)
            self.assertNotIn('inputs/job_spec.json', names)
            self.assertTrue(any(x.startswith('cases/') for x in names))
            from task_runtime import read_task
            self.assertEqual(read_task(root)['job_id'], root.name)
            from prepare_benchmark_tool import prepare
            tool = root/'sub_tasks/test/attempt-01/tool'
            prepare(PACKAGE/'assets/aisbench-prefix', tool)
            self.assertFalse(any(p.is_symlink() for p in tool.iterdir()))
            self.assertFalse((tool/'GSM8K.jsonl').exists())
            self.assertFalse((tool/'aisbench_test.py').exists())
            compile((PACKAGE/'assets/aisbench-prefix/aisbench_test.py').read_text(encoding='utf-8'), 'shared-tool', 'exec')
            self.assertIn('common/skill-snapshot/scripts/tool_links.py',names)
            self.assertFalse(any('/references/' in x or '/test_' in x for x in names))
            self.assertEqual(len([x for x in names if x.endswith('GSM8K.jsonl')]),1)
            second,_,_=submit_job.build(*args)
            self.assertNotEqual(root.name,second.name)

    def test_existing_remote_directory_stops_upload(self):
        with patch.object(submit_job,'ssh',side_effect=['',RuntimeError('exists')]),patch.object(submit_job.subprocess,'run') as run:
            with self.assertRaises(RuntimeError): submit_job.upload('root@example','/work/job',Path('unused'))
            run.assert_not_called()

if __name__=='__main__':unittest.main()
