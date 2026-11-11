"""Validate current public examples and exercise their offline generation."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / 'scripts'))
from task_model import load_json, normalize, validate, validate_bindings
import run as entry


class CurrentExampleTests(unittest.TestCase):
    def test_current_examples_validate(self):
        for path in (PACKAGE / 'examples').glob('bindings*.json'):
            with self.subTest(example=path.name):
                validate_bindings(load_json(path))
        for path in (PACKAGE / 'examples').glob('job_spec*.json'):
            with self.subTest(example=path.name):
                self.assertEqual(validate(normalize(load_json(path))), [])

    def test_documented_pairs_generate_without_remote_calls(self):
        for name in ('bindings.example.json', 'bindings.compact.json'):
            with self.subTest(bindings=name), tempfile.TemporaryDirectory() as tmp:
                spec = load_json(PACKAGE / 'examples/job_spec.compact.json')
                self.assertEqual(spec['state'], 'draft')
                self.assertTrue(spec['unresolved'])
                # Simulate review for this offline test only; examples stay draft.
                spec['state'] = 'ready'
                spec['unresolved'] = []
                bindings = load_json(PACKAGE / 'examples' / name)
                root = Path(tmp) / 'task'
                with patch.object(entry.subprocess, 'run', side_effect=AssertionError('Unexpected remote call')):
                    _, plan = entry.generate(spec, bindings, root)
                self.assertIn('pd-a', plan['deployments'])
                for instance in spec['deployments'][0]['instances']:
                    self.assertTrue((root / 'scripts/pd-a' / ('node_a-' + instance['id'] + '.sh')).is_file())
                self.assertTrue((root / 'scripts/pd-a/node_a-container.sh').is_file())
                for subtask in spec['sub_tasks']:
                    self.assertTrue((root / 'scripts/tests' / (subtask['id'] + '.sh')).is_file())
