"""Exercise real mmengine lazy config transforms without model/backend imports."""
import ast
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('functional_config_fixtures',
                                             Path(__file__).with_name('test_functional_profile.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class ConfigRuntimeTest(unittest.TestCase):
    def test_real_mmengine_models_and_datasets_match_plain_python_configs(self):
        try:
            from mmengine import Config
        except ImportError:
            self.skipTest('optional mmengine-lite is required for config integration')
        cases = [(fixtures.config_path(group, case), 'models', [dict(sentinel='base')])
                 for group, names in fixtures.MODEL_CASES.items() for case in names]
        cases += [(fixtures.config_path(group, case, 'datasets', dataset), variable,
                   [dict(name=f'subject-{i}', reader_cfg={})
                    for i in range(5 if dataset == 'ceval' else 1)])
                  for group, case, dataset, variable in fixtures.DATA_CASES]
        for source, variable, base in cases:
            with tempfile.TemporaryDirectory() as root:
                root = Path(root)
                source_text = source.read_text(encoding='utf-8')
                copied = root / source.name
                copied.write_text(source_text, encoding='utf-8')
                for node in ast.walk(ast.parse(source_text)):
                    if isinstance(node, ast.ImportFrom) and node.level == 1:
                        (root / (node.module + '.py')).write_text(
                            variable + ' = ' + repr(base) + '\n', encoding='utf-8')
                for fast in (None, '0', '1'):
                    environment = dict(os.environ)
                    environment.pop('AISBENCH_SMOKE_FAST', None)
                    if fast is not None:
                        environment['AISBENCH_SMOKE_FAST'] = fast
                    expected = fixtures.execute_config(source, fast, **{variable: base})[variable]
                    for lazy in (None, True):
                        with self.subTest(case=source.stem, fast=fast, lazy=lazy), \
                                patch.dict(os.environ, environment, clear=True):
                            actual = Config.fromfile(str(copied), lazy_import=lazy).to_dict()[variable]
                            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
