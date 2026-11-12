import shlex
import tempfile
import unittest
from pathlib import Path
from test_compact_entry import inputs
from task_model import expand
import run as entry


class TemplateExpansionTests(unittest.TestCase):
    def test_go_templates_and_shell_quoted_binding(self):
        value = "docker inspect -f '{{ index .Config.Labels \"bench.task\" }}' {{paths.target}}"
        target = '/path with spaces/$(touch marker)/{{literal}}'
        actual = expand(value, {'paths': {'target': target}}, shell=True)
        self.assertEqual(shlex.split(actual), ['docker', 'inspect', '-f', '{{ index .Config.Labels "bench.task" }}', target])
        self.assertEqual(expand('{{.Id}}', {}), '{{.Id}}')

    def test_missing_binding_still_rejected_and_structures_preserved(self):
        with self.assertRaisesRegex(ValueError, 'Unresolved binding: paths.missing'):
            expand("docker inspect -f '{{.Id}}' {{paths.missing}}", {'paths': {}}, shell=True)
        self.assertEqual(expand('{{value}}', {'value': {'a': [1, 2]}}), {'a': [1, 2]})
        with self.assertRaisesRegex(ValueError, 'Structured binding'):
            expand('prefix {{value}}', {'value': {'a': 1}})

    def test_bootstrap_survives_script_generation(self):
        spec, bindings = inputs()
        instance = spec['deployments'][0]['instances'][0]
        instance['launch']['bootstrap'] = "docker inspect -f '{{ index .Config.Labels \"bench.task\" }}' {{paths.target}}"
        bindings['nodes']['node_a']['paths']['target'] = '/path with spaces/$(literal)'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'job'
            entry.generate(spec, bindings, root)
            script = (root/'scripts'/spec['deployments'][0]['id']/('node_a-'+instance['id']+'.sh')).read_text()
            command = next(line for line in script.splitlines() if line.startswith('docker inspect -f '))
            self.assertEqual(shlex.split(command), ['docker', 'inspect', '-f', '{{ index .Config.Labels "bench.task" }}', '/path with spaces/$(literal)'])
