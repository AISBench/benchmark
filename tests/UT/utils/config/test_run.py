import unittest
from unittest.mock import patch, MagicMock
from mmengine.config import ConfigDict

from ais_bench.benchmark.utils.config.run import (
    try_fill_in_custom_cfgs,
    get_config_type,
    get_models_attr,
    fill_infer_cfg,
    fill_eval_cfg
)
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES


class TestRunConfig(unittest.TestCase):
    """Tests for run.py config functions."""

    def test_try_fill_in_custom_cfgs(self):
        """Test try_fill_in_custom_cfgs function."""
        config = ConfigDict({"key": "value"})
        result = try_fill_in_custom_cfgs(config)
        self.assertEqual(result, config)

    def test_get_config_type(self):
        """Test get_config_type function."""
        class TestClass:
            pass

        TestClass.__module__ = "test.module"
        TestClass.__name__ = "TestClass"

        result = get_config_type(TestClass)
        self.assertEqual(result, "test.module.TestClass")

    def test_get_models_attr_single_attr(self):
        """Test get_models_attr with single attribute."""
        cfg = {
            'models': [
                {'abbr': 'model1', 'attr': 'local'},
                {'abbr': 'model2', 'attr': 'local'}
            ]
        }
        result = get_models_attr(cfg)
        self.assertEqual(result, 'local')

    def test_get_models_attr_default_service(self):
        """Test get_models_attr with default service attribute."""
        cfg = {
            'models': [
                {'abbr': 'model1'},  # No attr, should default to 'service'
                {'abbr': 'model2', 'attr': 'service'}
            ]
        }
        result = get_models_attr(cfg)
        self.assertEqual(result, 'service')

    def test_get_models_attr_mixed_attrs_error(self):
        """Test get_models_attr raises error with mixed attributes."""
        cfg = {
            'models': [
                {'abbr': 'model1', 'attr': 'local'},
                {'abbr': 'model2', 'attr': 'service'}
            ]
        }
        with self.assertRaises(AISBenchConfigError) as cm:
            get_models_attr(cfg)
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.MIXED_MODEL_ATTRS.full_code)

    def test_get_models_attr_illegal_attr_error(self):
        """Test get_models_attr raises error with illegal attribute."""
        cfg = {
            'models': [
                {'abbr': 'model1', 'attr': 'invalid'}
            ]
        }
        with self.assertRaises(AISBenchConfigError) as cm:
            get_models_attr(cfg)
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.ILLEGAL_MODEL_ATTR.full_code)

    @patch('ais_bench.benchmark.utils.config.run.logger')
    def test_fill_infer_cfg(self, mock_logger):
        """Test fill_infer_cfg function."""
        from mmengine.config import Config

        cfg = Config({
            'datasets': [
                ConfigDict({
                    'infer_cfg': ConfigDict({
                        'retriever': ConfigDict({}),
                        'prompt_template': ConfigDict({'type': 'PromptTemplate'}),
                        'ice_template': ConfigDict({'type': 'IceTemplate'})
                    })
                })
            ]
        })

        class MockArgs:
            max_num_workers = 4
            max_workers_per_gpu = 2
            debug = False

        args = MockArgs()
        fill_infer_cfg(cfg, args)

        # Verify infer config was added
        self.assertIn('infer', cfg)
        self.assertIn('partitioner', cfg['infer'])
        self.assertIn('runner', cfg['infer'])
        self.assertEqual(cfg['infer']['runner']['max_num_workers'], 4)
        self.assertEqual(cfg['infer']['runner']['max_workers_per_gpu'], 2)
        self.assertEqual(cfg['infer']['runner']['debug'], False)

        # Verify prompt_template and ice_template were copied to retriever
        dataset = cfg['datasets'][0]
        self.assertIn('prompt_template', dataset['infer_cfg']['retriever'])
        self.assertIn('ice_template', dataset['infer_cfg']['retriever'])

    @patch('ais_bench.benchmark.utils.config.run.logger')
    def test_fill_eval_cfg(self, mock_logger):
        """Test fill_eval_cfg function."""
        from mmengine.config import Config

        cfg = Config({})

        class MockArgs:
            max_num_workers = 4
            max_workers_per_gpu = 2
            debug = False

        args = MockArgs()
        fill_eval_cfg(cfg, args)

        # Verify eval config was added
        self.assertIn('eval', cfg)
        self.assertIn('partitioner', cfg['eval'])
        self.assertIn('runner', cfg['eval'])
        self.assertEqual(cfg['eval']['runner']['max_num_workers'], 4)
        self.assertEqual(cfg['eval']['runner']['max_workers_per_gpu'], 2)
        self.assertEqual(cfg['eval']['runner']['debug'], False)

if __name__ == "__main__":
    unittest.main()

