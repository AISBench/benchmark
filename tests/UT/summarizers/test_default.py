import unittest
import os
from unittest.mock import patch, MagicMock, mock_open, call
from collections import OrderedDict
import tempfile
import sys
from datetime import datetime

from ais_bench.benchmark.summarizers.default import (
    DefaultSummarizer,
    model_abbr_from_cfg_used_in_summarizer,
    METRIC_WHITELIST,
    METRIC_BLACKLIST
)
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError
from mmengine import ConfigDict


class TestModelAbbrFromCfgUsedInSummarizer(unittest.TestCase):
    def test_with_summarizer_abbr(self):
        """测试当模型配置中包含summarizer_abbr时的情况"""
        model = {"summarizer_abbr": "custom_model"}
        result = model_abbr_from_cfg_used_in_summarizer(model)
        self.assertEqual(result, "custom_model")

    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg')
    def test_without_summarizer_abbr(self, mock_model_abbr_from_cfg):
        """测试当模型配置中不包含summarizer_abbr时的情况"""
        mock_model_abbr_from_cfg.return_value = "default_model"
        model = {"name": "test_model"}
        result = model_abbr_from_cfg_used_in_summarizer(model)
        self.assertEqual(result, "default_model")
        mock_model_abbr_from_cfg.assert_called_once_with(model)


class TestDefaultSummarizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 全局补丁，确保任何地方创建的ConfigDict都有path参数
        cls.original_config_dict = ConfigDict

        class PatchedConfigDict(ConfigDict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # 确保path参数存在
                if 'path' not in self:
                    self['path'] = './test_path'
                # 确保在所有情况下都能安全访问path
                def __getitem__(self, key):
                    try:
                        return super().__getitem__(key)
                    except KeyError:
                        if key == 'path':
                            return './test_path'
                        raise

                setattr(self, '__getitem__', __getitem__)

        # 替换模块中的ConfigDict
        import ais_bench.benchmark.summarizers.default
        cls.original_default_config_dict = ais_bench.benchmark.summarizers.default.ConfigDict
        ais_bench.benchmark.summarizers.default.ConfigDict = PatchedConfigDict

    @classmethod
    def tearDownClass(cls):
        # 恢复原始的ConfigDict
        import ais_bench.benchmark.summarizers.default
        ais_bench.benchmark.summarizers.default.ConfigDict = cls.original_default_config_dict

    def setUp(self):
        # 基础配置
        self.model_cfg = {
            "type": "TestModel",
            "abbr": "test_model",
            "path": "/the/path/to/test_path"  # 确保path参数存在
        }
        self.model_cfg2 = {
            "type": "TestModel2",
            "abbr": "test_model2",
            "summarizer_abbr": "custom_abbr"
        }
        self.dataset_cfg = {
            "type": "TestDataset",
            "abbr": "test_dataset",
            "infer_cfg": {
                "inferencer": {"type": "GenInferencer"}
            }
        }
        self.dataset_cfg2 = {
            "type": "TestDataset2",
            "abbr": "test_dataset2",
            "infer_cfg": {
                "inferencer": {"type": "PPLInferencer"}
            }
        }
        self.dataset_cfg3 = {
            "type": "TestDataset3",
            "abbr": "test_dataset3",
            "infer_cfg": {
                "inferencer": {"type": "UnknownInferencer"}
            }
        }
        # 确保config对象有path参数
        self.config = ConfigDict({
            "models": [self.model_cfg, self.model_cfg2],
            "datasets": [self.dataset_cfg, self.dataset_cfg2, self.dataset_cfg3],
            "work_dir": "/tmp/test_work_dir",
        })
        # 摘要组配置
        self.summary_groups = [
            {"name": "group1", "subsets": ["test_ds", "test_ds2"]},
            {"name": "group2", "subsets": ["test_ds", "test_ds2"], "weights": {"test_ds": 0.3, "test_ds2": 0.7}},
            {"name": "group3", "subsets": ["test_ds", "test_ds2"], "std": True},
            {"name": "group4", "subsets": ["test_ds", "test_ds2"], "sum": True},
            {"name": "group5", "subsets": [("test_ds", "score"), ("test_ds2", "accuracy")], "metric": "custom_metric"}
        ]

    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg_used_in_summarizer')
    def test_init(self, mock_model_abbr_from_cfg_used, mock_ais_logger):
        """测试DefaultSummarizer的初始化"""
        # 模拟model_abbr_from_cfg_used_in_summarizer函数的返回值
        mock_model_abbr_from_cfg_used.side_effect = ["test_model", "custom_abbr"]

        summarizer = DefaultSummarizer(self.config, summary_groups=self.summary_groups)

        self.assertEqual(summarizer.cfg, self.config)
        self.assertEqual(summarizer.model_cfgs, [self.model_cfg, self.model_cfg2])
        self.assertEqual(summarizer.dataset_cfgs, [self.dataset_cfg, self.dataset_cfg2, self.dataset_cfg3])
        self.assertEqual(summarizer.work_dir, "/tmp/test_work_dir")
        self.assertEqual(summarizer.summary_groups, self.summary_groups)
        self.assertEqual(summarizer.model_abbrs, ["test_model", "custom_abbr"])
        mock_ais_logger.assert_called_once()

    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_init_with_prompt_db(self, mock_ais_logger):
        """测试使用deprecated的prompt_db参数初始化"""
# 设置mock
        mock_logger = MagicMock()
        # 确保AISLogger不需要path参数
        mock_ais_logger.return_value = mock_logger
        mock_ais_logger.__call__ = MagicMock(return_value=mock_logger)

        summarizer = DefaultSummarizer(self.config, prompt_db="deprecated_db")

        mock_logger.warning.assert_called_with('prompt_db is deprecated and no longer used. Please remove it from your config.')

    @patch('ais_bench.benchmark.summarizers.default.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_update_dataset_abbrs(self, mock_ais_logger, mock_dataset_abbr_from_cfg):
        """测试_update_dataset_abbrs方法"""
        # 设置mock
        mock_logger = MagicMock()
        # 确保AISLogger不需要path参数
        mock_ais_logger.return_value = mock_logger
        # 重要：确保config对象有path参数
        if 'path' not in self.config:
            self.config['path'] = './test_path'
        mock_dataset_abbr_from_cfg.side_effect = ["test_ds", "test_ds2", "test_ds3"]

        summarizer = DefaultSummarizer(self.config)
        summarizer._update_dataset_abbrs()

        self.assertEqual(summarizer.dataset_abbrs, ["test_ds", "test_ds2", "test_ds3"])
        self.assertEqual(mock_dataset_abbr_from_cfg.call_count, 3)

    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.default.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default.get_infer_output_path')
    @patch('os.path.exists')
    @patch('mmengine.load')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_pick_up_results_normal(self, mock_ais_logger, mock_mmengine_load, mock_exists,
                                 mock_get_infer_output_path, mock_dataset_abbr_from_cfg,
                                 mock_model_abbr_from_cfg_used):
        """测试正常情况下的_pick_up_results方法"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
        mock_dataset_abbr_from_cfg.return_value = "test_ds"
        mock_get_infer_output_path.return_value = "/tmp/test_work_dir/results/test_model/test_ds/results.json"
        mock_exists.return_value = True
        mock_result = {"score": 0.8, "bp": 0.9, "details": {"some": "details"}}
        mock_mmengine_load.return_value = mock_result
# 设置mock
        mock_logger = MagicMock()
        # 确保AISLogger不需要path参数
        mock_ais_logger.return_value = mock_logger
        # 重要：确保config对象有path参数
        if 'path' not in self.config:
            self.config['path'] = './test_path'

        summarizer = DefaultSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        # 验证结果
        self.assertIn("test_model", raw_results)
        self.assertIn("test_ds", raw_results["test_model"])
        self.assertNotIn("details", raw_results["test_model"]["test_ds"])
        self.assertIn("score", raw_results["test_model"]["test_ds"])

        self.assertIn("test_model", parsed_results)
        self.assertIn("test_ds", parsed_results["test_model"])
        self.assertIn("score", parsed_results["test_model"]["test_ds"])
        self.assertNotIn("bp", parsed_results["test_model"]["test_ds"])

        self.assertIn("test_ds", dataset_metrics)
        self.assertEqual(dataset_metrics["test_ds"], ["score"])

        self.assertIn("test_ds", dataset_eval_mode)
        self.assertEqual(dataset_eval_mode["test_ds"], "unknown")

    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.default.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default.get_infer_output_path')
    @patch('os.path.exists')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_pick_up_results_file_not_exist(self, mock_ais_logger, mock_exists,
                                        mock_get_infer_output_path, mock_dataset_abbr_from_cfg,
                                        mock_model_abbr_from_cfg_used):
        """测试文件不存在的情况"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
        mock_dataset_abbr_from_cfg.return_value = "test_ds"
        mock_get_infer_output_path.return_value = "/tmp/test_work_dir/results/test_model/test_ds/results.json"
        mock_exists.return_value = False
        mock_logger = MagicMock()
        mock_ais_logger.return_value = mock_logger

        summarizer = DefaultSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        # 验证结果 - 应该是空的
        self.assertEqual(raw_results, {"test_model": {}})
        self.assertEqual(parsed_results, {"test_model": {}})
        self.assertEqual(dataset_metrics, {})

    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.default.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default.get_infer_output_path')
    @patch('os.path.exists')
    @patch('mmengine.load')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_pick_up_results_with_error(self, mock_ais_logger, mock_mmengine_load, mock_exists,
                                     mock_get_infer_output_path, mock_dataset_abbr_from_cfg,
                                     mock_model_abbr_from_cfg_used):
        """测试结果中有error的情况"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
        mock_dataset_abbr_from_cfg.return_value = "test_ds"
        mock_get_infer_output_path.return_value = "/tmp/test_work_dir/results/test_model/test_ds/results.json"
        mock_exists.return_value = True
        mock_result = {"error": "some error"}
        mock_mmengine_load.return_value = mock_result
        mock_logger = MagicMock()
        mock_ais_logger.return_value = mock_logger

        summarizer = DefaultSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        # 验证结果
        self.assertIn("test_model", raw_results)
        self.assertIn("test_ds", raw_results["test_model"])
        self.assertIn("error", raw_results["test_model"]["test_ds"])

        self.assertIn("test_model", parsed_results)
        self.assertNotIn("test_ds", parsed_results["test_model"])

        mock_logger.debug.assert_called_with("error in test_model test_ds some error")

    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.default.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default.get_infer_output_path')
    @patch('os.path.exists')
    @patch('mmengine.load')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_pick_up_results_unknown_format(self, mock_ais_logger, mock_mmengine_load, mock_exists,
                                         mock_get_infer_output_path, mock_dataset_abbr_from_cfg,
                                         mock_model_abbr_from_cfg_used):
        """测试未知结果格式的情况"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
        mock_dataset_abbr_from_cfg.return_value = "test_ds"
        mock_get_infer_output_path.return_value = "/tmp/test_work_dir/results/test_model/test_ds/results.json"
        mock_exists.return_value = True
        mock_result = {"non_numeric": "string", "bp": 0.9}
        mock_mmengine_load.return_value = mock_result
        mock_logger = MagicMock()
        mock_ais_logger.return_value = mock_logger
        # 模拟AISLogger的初始化

        summarizer = DefaultSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        # 验证结果
        self.assertIn("test_model", raw_results)
        self.assertIn("test_ds", raw_results["test_model"])

        self.assertIn("test_model", parsed_results)
        self.assertNotIn("test_ds", parsed_results["test_model"])

    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_calculate_group_metrics_naive_average(self, mock_ais_logger, mock_model_abbr_from_cfg_used):
        """测试基本的平均计算"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
# 设置mock
        mock_logger = MagicMock()
        # 确保AISLogger不需要path参数
        mock_ais_logger.return_value = mock_logger
        # 重要：确保config对象有path参数
        if 'path' not in self.config:
            self.config['path'] = './test_path'
        # 模拟AISLogger的初始化

        # 准备测试数据
        raw_results = {
            "test_model": {
                "test_ds": {"score": 0.8},
                "test_ds2": {"score": 0.6}
            }
        }
        parsed_results = {
            "test_model": {
                "test_ds": {"score": 0.8},
                "test_ds2": {"score": 0.6}
            }
        }
        dataset_metrics = {
            "test_ds": ["score"],
            "test_ds2": ["score"]
        }
        dataset_eval_mode = {
            "test_ds": "unknown",
            "test_ds2": "unknown"
        }

        # 使用默认的summary_group，没有指定sum、std或weights，应该使用naive_average
        summary_group = {"name": "group1", "subsets": ["test_ds", "test_ds2"]}
        summarizer = DefaultSummarizer(self.config, summary_groups=[summary_group])
        summarizer.model_abbrs = ["test_model"]  # 直接设置model_abbrs以避免mock问题
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果
        self.assertIn("group1", parsed_results["test_model"])
        self.assertIn("naive_average", parsed_results["test_model"]["group1"])
        self.assertAlmostEqual(parsed_results["test_model"]["group1"]["naive_average"], 0.7)

    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_calculate_group_metrics_weighted(self, mock_ais_logger, mock_model_abbr_from_cfg_used):
        """测试加权平均计算"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
        mock_logger = MagicMock()
        mock_ais_logger.return_value = mock_logger
        # 模拟AISLogger的初始化

        # 准备测试数据
        raw_results = {
            "test_model": {
                "test_ds": {"score": 0.8},
                "test_ds2": {"score": 0.6}
            }
        }
        parsed_results = {
            "test_model": {
                "test_ds": {"score": 0.8},
                "test_ds2": {"score": 0.6}
            }
        }
        dataset_metrics = {
            "test_ds": ["score"],
            "test_ds2": ["score"]
        }
        dataset_eval_mode = {
            "test_ds": "unknown",
            "test_ds2": "unknown"
        }

        # 使用weights的summary_group
        summary_group = {"name": "group2", "subsets": ["test_ds", "test_ds2"], "weights": {"test_ds": 0.3, "test_ds2": 0.7}}
        summarizer = DefaultSummarizer(self.config, summary_groups=[summary_group])
        summarizer.model_abbrs = ["test_model"]
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果
        self.assertIn("group2", parsed_results["test_model"])
        self.assertIn("weighted_average", parsed_results["test_model"]["group2"])
        # 0.8 * 0.3 + 0.6 * 0.7 = 0.66
        self.assertAlmostEqual(parsed_results["test_model"]["group2"]["weighted_average"], 0.66)

    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_calculate_group_metrics_std(self, mock_ais_logger, mock_model_abbr_from_cfg_used):
        """测试标准差计算"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
# 设置mock
        mock_logger = MagicMock()
        # 确保AISLogger不需要path参数
        mock_ais_logger.return_value = mock_logger
        # 重要：确保config对象有path参数
        if 'path' not in self.config:
            self.config['path'] = './test_path'

        # 准备测试数据
        raw_results = {
            "test_model": {
                "test_ds": {"score": 0.8},
                "test_ds2": {"score": 0.6}
            }
        }
        parsed_results = {
            "test_model": {
                "test_ds": {"score": 0.8},
                "test_ds2": {"score": 0.6}
            }
        }
        dataset_metrics = {
            "test_ds": ["score"],
            "test_ds2": ["score"]
        }
        dataset_eval_mode = {
            "test_ds": "unknown",
            "test_ds2": "unknown"
        }

        # 使用std=True的summary_group
        summary_group = {"name": "group3", "subsets": ["test_ds", "test_ds2"], "std": True}
        summarizer = DefaultSummarizer(self.config, summary_groups=[summary_group])
        summarizer.model_abbrs = ["test_model"]
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果 - 实际标准差是约0.1
        self.assertIn("group3", parsed_results["test_model"])
        self.assertIn("standard_deviation", parsed_results["test_model"]["group3"])
        self.assertAlmostEqual(parsed_results["test_model"]["group3"]["standard_deviation"], 0.1, places=5)

    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_calculate_group_metrics_sum(self, mock_ais_logger, mock_model_abbr_from_cfg_used):
        """测试计算总和"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
        mock_logger = MagicMock()
        mock_ais_logger.return_value = mock_logger
        # 模拟AISLogger的初始化

        # 准备测试数据
        raw_results = {
            "test_model": {
                "test_ds": {"score": 0.8},
                "test_ds2": {"score": 0.6}
            }
        }
        parsed_results = {
            "test_model": {
                "test_ds": {"score": 0.8},
                "test_ds2": {"score": 0.6}
            }
        }
        dataset_metrics = {
            "test_ds": ["score"],
            "test_ds2": ["score"]
        }
        dataset_eval_mode = {
            "test_ds": "unknown",
            "test_ds2": "unknown"
        }

        # 使用sum=True的summary_group
        summary_group = {'name': 'group4', 'subsets': ['test_ds', 'test_ds2'], 'sum': True}
        summarizer = DefaultSummarizer(self.config, summary_groups=[summary_group])
        summarizer.model_abbrs = ["test_model"]
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果
        self.assertIn("group4", parsed_results["test_model"])
        self.assertIn("sum", parsed_results["test_model"]["group4"])
        self.assertAlmostEqual(parsed_results["test_model"]["group4"]["sum"], 1.4)

    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_calculate_group_metrics_custom_metric_tuple(self, mock_ais_logger, mock_model_abbr_from_cfg_used):
        """测试使用元组形式的自定义metric"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
        mock_logger = MagicMock()
        mock_ais_logger.return_value = mock_logger
        # 模拟AISLogger的初始化

        # 准备测试数据
        raw_results = {
            "test_model": {
                "test_ds": {"score": 0.8},
                "test_ds2": {"accuracy": 0.9}
            }
        }
        parsed_results = {
            "test_model": {
                "test_ds": {"score": 0.8, "accuracy": 0.7},
                "test_ds2": {"score": 0.6, "accuracy": 0.9}
            }
        }
        dataset_metrics = {
            "test_ds": ["score", "accuracy"],
            "test_ds2": ["score", "accuracy"]
        }
        dataset_eval_mode = {
            "test_ds": "unknown",
            "test_ds2": "unknown"
        }

        # 使用自定义metric的summary_group
        summary_group = {"name": "group5", "subsets": [("test_ds", "score"), ("test_ds2", "accuracy")], "metric": "custom_metric"}
        summarizer = DefaultSummarizer(self.config, summary_groups=[summary_group])
        summarizer.model_abbrs = ["test_model"]
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果
        self.assertIn("group5", parsed_results["test_model"])
        # 0.8和0.9的平均值
        self.assertAlmostEqual(parsed_results["test_model"]["group5"]["custom_metric"], 0.85)

    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_calculate_group_metrics_missing_metrics(self, mock_ais_logger, mock_model_abbr_from_cfg_used):
        """测试缺少指标的情况"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
        mock_logger = MagicMock()
        mock_ais_logger.return_value = mock_logger
        # 初始化时不再需要'path'参数
        # 模拟AISLogger的初始化

        # 准备测试数据
        raw_results = {
            "test_model": {
                "test_ds": {"score": 0.8}
                # 缺少test_ds2
            }
        }
        parsed_results = {
            "test_model": {
                "test_ds": {"score": 0.8}
                # 缺少test_ds2
            }
        }
        dataset_metrics = {
            "test_ds": ["score"]
        }
        dataset_eval_mode = {
            "test_ds": "unknown"
        }

        # 使用包含缺少指标的summary_group
        summary_group = {"name": "group1", "subsets": ["test_ds", "test_ds2"]}
        summarizer = DefaultSummarizer(self.config, summary_groups=[summary_group])
        summarizer.model_abbrs = ["test_model"]
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果 - 应该有error信息
        self.assertIn("group1", raw_results["test_model"])
        self.assertIn("error", raw_results["test_model"]["group1"])
        self.assertIn("missing metrics", raw_results["test_model"]["group1"]["error"])

    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg_used_in_summarizer')
    def test_calculate_group_metrics_no_available_metrics(self, mock_model_abbr_from_cfg_used):
        """测试没有可用指标的情况"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"

        # 准备测试数据 - 空的parsed_results
        raw_results = {
            "test_model": {}
        }
        parsed_results = {
            "test_model": {}
        }
        dataset_metrics = {}
        dataset_eval_mode = {}

        # 使用summary_group
        summary_group = {"name": "group1", "subsets": ["test_ds", "test_ds2"]}
        summarizer = DefaultSummarizer(self.config, summary_groups=[summary_group])
        summarizer.model_abbrs = ["test_model"]
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果 - 应该没有添加group1
        self.assertNotIn("group1", raw_results["test_model"])
        self.assertNotIn("group1", parsed_results["test_model"])

    def test_format_md_table(self):
        """测试_format_md_table静态方法"""
        # 准备测试数据
        table = [
            ["dataset", "metric", "model1"],
            ["test_ds", "score", "0.8"]
        ]

        # 调用静态方法
        md_table = DefaultSummarizer._format_md_table(table)

        # 验证结果 - 允许表格分隔符中包含空格
        self.assertIn("| dataset | metric | model1 |", md_table)
        self.assertIn("|-----", md_table)
        self.assertIn("| test_ds | score | 0.8 |", md_table)

    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_format_raw_txt(self, mock_ais_logger):
        """测试_format_raw_txt方法"""
        # 设置mock
        mock_logger = MagicMock()
        # 确保AISLogger不需要path参数
        mock_ais_logger.return_value = mock_logger
        # 重要：确保config对象有path参数
        if 'path' not in self.config:
            self.config['path'] = './test_path'

        # 准备测试数据
        raw_results = {
            "test_model": {
                "test_ds": {"score": 0.8},
                "test_ds2": {"accuracy": 0.9}
            }
        }

        summarizer = DefaultSummarizer(self.config)
        summarizer.model_abbrs = ["test_model"]

        raw_txt = summarizer._format_raw_txt(raw_results)

        # 验证结果
        self.assertIn("Model: test_model", raw_txt)
        self.assertIn("test_ds:", raw_txt)
        self.assertIn("test_ds2:", raw_txt)

    @patch('mmengine.mkdir_or_exist')
    @patch('builtins.print')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.abspath')
    @patch('ais_bench.benchmark.summarizers.default.DefaultSummarizer._format_md_table')
    @patch('ais_bench.benchmark.summarizers.default.tabulate.tabulate')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_output_to_file_default_path(self, mock_ais_logger, mock_tabulate, mock_format_md_table,
                                     mock_abspath, mock_open_file, mock_print, mock_mkdir_or_exist):
        """测试_output_to_file方法使用默认路径"""
        # 设置mock
        mock_logger = MagicMock()
        # 确保AISLogger不需要path参数
        mock_ais_logger.return_value = mock_logger
        # 重要：确保config对象有path参数
        if 'path' not in self.config:
            self.config['path'] = './test_path'
        mock_abspath.return_value = "/abs/path/to/summary.txt"
        mock_format_md_table.return_value = "markdown table"
        mock_tabulate.return_value = "tabulate table"

        # 准备测试数据
        table = [["dataset", "metric"], ["test_ds", "0.8"]]
        raw_txts = "test raw text"

        summarizer = DefaultSummarizer(self.config)
        summarizer.work_dir = "/tmp/test_work_dir"

        summarizer._output_to_file(None, "20230101_000000", table, raw_txts)

        # 验证调用
        mock_mkdir_or_exist.assert_called_with("/tmp/test_work_dir/summary")
        mock_open_file.assert_any_call("/tmp/test_work_dir/summary/summary_20230101_000000.txt", "w", encoding="utf-8")
        mock_open_file.assert_any_call("/tmp/test_work_dir/summary/summary_20230101_000000.csv", "w", encoding="utf-8")
        mock_open_file.assert_any_call("/tmp/test_work_dir/summary/summary_20230101_000000.md", "w", encoding="utf-8")

    @patch('mmengine.mkdir_or_exist')
    @patch('builtins.print')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.abspath')
    @patch('ais_bench.benchmark.summarizers.default.DefaultSummarizer._format_md_table')
    @patch('ais_bench.benchmark.summarizers.default.tabulate.tabulate')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_output_to_file_custom_path(self, mock_ais_logger, mock_tabulate, mock_format_md_table,
                                     mock_abspath, mock_open_file, mock_print, mock_mkdir_or_exist):
        """测试_output_to_file方法使用自定义路径"""
        # 设置mock
        mock_logger = MagicMock()
        # 确保AISLogger不需要path参数
        mock_ais_logger.return_value = mock_logger
        # 重要：确保config对象有path参数
        if 'path' not in self.config:
            self.config['path'] = './test_path'
        # 模拟AISLogger的初始化
        mock_abspath.return_value = "/abs/path/to/custom_summary.txt"
        mock_format_md_table.return_value = "markdown table"
        mock_tabulate.return_value = "tabulate table"

        # 准备测试数据
        table = [["dataset", "metric"], ["test_ds", "0.8"]]
        raw_txts = "test raw text"

        summarizer = DefaultSummarizer(self.config)

        summarizer._output_to_file("/custom/path/summary.txt", "20230101_000000", table, raw_txts)

        # 验证调用
        mock_mkdir_or_exist.assert_called_with("/custom/path")
        mock_open_file.assert_any_call("/custom/path/summary.txt", "w", encoding="utf-8")
        mock_open_file.assert_any_call("/custom/path/summary.csv", "w", encoding="utf-8")
        mock_open_file.assert_any_call("/custom/path/summary.md", "w", encoding="utf-8")

    @patch('builtins.print')
    @patch('ais_bench.benchmark.summarizers.default.DefaultSummarizer._output_to_file')
    @patch('ais_bench.benchmark.summarizers.default.DefaultSummarizer._format_raw_txt')
    @patch('ais_bench.benchmark.summarizers.default.DefaultSummarizer._format_table')
    @patch('ais_bench.benchmark.summarizers.default.DefaultSummarizer._calculate_group_metrics')
    @patch('ais_bench.benchmark.summarizers.default.DefaultSummarizer._pick_up_results')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_summarize(self, mock_ais_logger, mock_pick_up_results, mock_calculate_group_metrics,
                    mock_format_table, mock_format_raw_txt, mock_output_to_file, mock_print):
        """测试summarize方法"""
        # 设置mock
        mock_logger = MagicMock()
        # 确保AISLogger不需要path参数
        mock_ais_logger.return_value = mock_logger
        mock_ais_logger.__call__ = MagicMock(return_value=mock_logger)
        mock_pick_up_results.return_value = ({"test_model": {"test_ds": {"score": 0.8}}},
                                          {"test_model": {"test_ds": {"score": 0.8}}},
                                          {"test_ds": ["score"]},
                                          {"test_ds": "unknown"})
        mock_calculate_group_metrics.return_value = mock_pick_up_results.return_value
        mock_format_table.return_value = [["dataset", "metric"], ["test_ds", "0.8"]]
        mock_format_raw_txt.return_value = "test raw text"

        summarizer = DefaultSummarizer(self.config)
        summarizer.dataset_abbrs = ["test_ds"]

        # 执行测试
        summarizer.summarize(output_path="/custom/path/summary.txt", time_str="20230101_000000")

        # 验证所有方法都被调用
        mock_pick_up_results.assert_called_once()
        mock_calculate_group_metrics.assert_called_once()
        mock_format_table.assert_called_once()
        mock_format_raw_txt.assert_called_once()
        mock_output_to_file.assert_called_with("/custom/path/summary.txt", "20230101_000000",
                                            [["dataset", "metric"], ["test_ds", "0.8"]], "test raw text")
        mock_print.assert_called_once()

    @patch('ais_bench.benchmark.summarizers.default.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_calculate_group_metrics_with_correct_total_count(self, mock_ais_logger, mock_model_abbr_from_cfg_used):
        """测试包含correct_count和total_count的情况"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
        mock_logger = MagicMock()
        mock_ais_logger.return_value = mock_logger
        # 初始化时不再需要'path'参数
        # 模拟AISLogger的初始化

        # 准备测试数据，包含correct_count和total_count
        raw_results = {
            "test_model": {
                "test_ds": {"accuracy": 0.8, "correct_count": 80, "total_count": 100}
            }
        }
        parsed_results = {
            "test_model": {
                "test_ds": {"accuracy": 0.8, "correct_count": 80, "total_count": 100}
            }
        }
        dataset_metrics = {
            "test_ds": ["accuracy"]
        }
        dataset_eval_mode = {
            "test_ds": "unknown"
        }

        # 测试_format_table方法对correct_count和total_count的处理
        summarizer = DefaultSummarizer(self.config)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = ["test_ds"]
        with patch('ais_bench.benchmark.summarizers.default.dataset_abbr_from_cfg', return_value="test_ds"), \
             patch('ais_bench.benchmark.summarizers.default.get_prompt_hash', return_value="abc123"):
            table = summarizer._format_table(parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果中包含correct_count/total_count的信息
        found = False
        for item in table[1]:
            if "0.80" in str(item):
                found = True
                break
        self.assertTrue(found, "表格中应该包含准确率值")


if __name__ == "__main__":
    unittest.main()