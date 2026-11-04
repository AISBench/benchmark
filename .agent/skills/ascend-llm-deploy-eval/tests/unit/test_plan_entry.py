import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
"""Regression checks for planning entry points; no remote operations."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import plan_run


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = plan_run.load_json(plan_run.PACKAGE/'examples/job_spec.serial.json')
        self.spec.pop('job_id')
        self.spec.update(state='ready', unresolved=[])

    def invoke(self, *args):
        source = self.root/'input.json'
        source.write_text(json.dumps(self.spec), encoding='utf-8')
        result = subprocess.run([sys.executable, '-B', str(plan_run.PACKAGE/'scripts/plan_run.py'),
            '--job-spec', str(source), *map(str, args)], capture_output=True, text=True)
        return result.returncode, json.loads(result.stdout)

    def test_read_only_validation_and_collision(self):
        jobs = self.root/'jobs'
        args = ('--jobs-dir', jobs, '--date', '20261102')
        code, result = self.invoke(*args, '--validate-only')
        self.assertEqual(code, 0, result)
        self.assertEqual(result['status'], 'VALIDATED')
        self.assertFalse(jobs.exists())
        (jobs/'20261102-001').mkdir(parents=True)
        code, result = self.invoke(*args)
        self.assertEqual(code, 0, result)
        self.assertEqual(result['job_id'], '20261102-002')
        frozen = plan_run.load_json(jobs/'20261102-002/common/plan.json')
        self.assertEqual(frozen['job_spec']['job_id'], result['job_id'])
        self.assertTrue((jobs/'20261102-002/run.sh').exists())

    def test_invalid_id_and_business_constraint(self):
        self.spec['job_id'] = '20261102-211-128k'
        code, result = self.invoke('--validate-only')
        self.assertEqual(code, 2)
        self.assertIn('job_id', result['message'])
        self.spec['job_id'] = '20261102-211'
        self.spec['cases'][0]['input_tokens'] = 2000000
        code, result = self.invoke('--validate-only')
        self.assertEqual(code, 2)
        self.assertIn('Context limit exceeded', result['message'])

    def test_draft_validates_but_cannot_freeze(self):
        self.spec['state'] = 'draft'
        jobs = self.root/'jobs'
        args = ('--jobs-dir', jobs, '--date', '20261102')
        self.assertEqual(self.invoke(*args, '--validate-only')[0], 0)
        self.assertEqual(self.invoke(*args)[0], 2)
        self.assertFalse(jobs.exists())

    def test_dependencies_and_date(self):
        self.assertEqual(self.invoke('--check-dependencies')[0], 0)
        with patch('importlib.metadata.version', return_value='0.0'):
            with self.assertRaises(ValueError):
                plan_run.check_dependencies()
        with self.assertRaises(ValueError):
            list(plan_run.candidate_ids('20260230'))


if __name__ == '__main__':
    unittest.main()
