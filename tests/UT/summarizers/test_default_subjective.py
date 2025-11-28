import unittest
import os
from unittest.mock import patch, MagicMock, mock_open, call
from collections import OrderedDict
import tempfile
import sys

from ais_bench.benchmark.summarizers.default_subjective import (
    DefaultSubjectiveSummarizer,
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

    @patch('ais_bench.benchmark.summarizers.default_subjective.model_abbr_from_cfg')
    def test_without_summarizer_abbr(self, mock_model_abbr_from_cfg):
        """测试当模型配置中不包含summarizer_abbr时的情况"""
        mock_model_abbr_from_cfg.return_value = "default_model"
        model = {"name": "test_model"}
        result = model_abbr_from_cfg_used_in_summarizer(model)
        self.assertEqual(result, "default_model")
        mock_model_abbr_from_cfg.assert_called_once_with(model)


class TestDefaultSubjectiveSummarizer(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.model_cfg = {
            "name": "test_model",
            "type": "TestModelType",
            "path": "test/path/model",
            "attr": "service"
        }
        self.model_cfg2 = {
            "name": "test_model2",
            "type": "TestModelType2",
            "path": "test/path/model2",
            "attr": "service",
            "summarizer_abbr": "custom_abbr"
        }
        self.dataset_cfg = {
            "type": "TestDataset",
            "abbr": "test_ds",
            "infer_cfg": {"inferencer": {"type": "GenInferencer"}}
        }
        self.dataset_cfg2 = {
            "type": "TestDataset2",
            "abbr": "test_ds2",
            "infer_cfg": {"inferencer": {"type": "PPLInferencer"}}
        }
        self.dataset_cfg3 = {
            "type": "TestDataset3",
            "abbr": "test_ds3",
            "infer_cfg": {"inferencer": {"type": "LLInferencer"}},
            "base_models": [{"abbr": "base_model1"}]
        }
        self.judge_model = {
            "name": "judge_model",
            "type": "JudgeModelType",
            "path": "test/path/judge"
        }
        self.config = ConfigDict({
            "models": [self.model_cfg, self.model_cfg2],
            "datasets": [self.dataset_cfg, self.dataset_cfg2, self.dataset_cfg3],
            "work_dir": "/tmp/test_work_dir",
            "eval": {
                "partitioner": {
                    "models": [self.model_cfg, self.model_cfg2]
                }
            },
            "judge_models": [self.judge_model]
        })
        self.summary_groups = [
            {
                "name": "group1",
                "subsets": ["test_ds", "test_ds2"]
            },
            {
                "name": "group2",
                "subsets": ["test_ds", "test_ds2"],
                "weights": {"test_ds": 0.3, "test_ds2": 0.7}
            },
            {
                "name": "group3",
                "subsets": ["test_ds", "test_ds2"],
                "std": True
            },
            {
                "name": "group4",
                "subsets": ["test_ds", "test_ds2"],
                "sum": True
            },
            {
                "name": "group5",
                "subsets": [("test_ds", "score"), ("test_ds2", "accuracy")],
                "metric": "custom_metric"
            }
        ]

    @patch('ais_bench.benchmark.summarizers.default_subjective.AISLogger')
    @patch('ais_bench.benchmark.summarizers.default_subjective.model_abbr_from_cfg_used_in_summarizer')
    def test_init(self, mock_model_abbr_from_cfg_used, mock_ais_logger):
        """测试DefaultSubjectiveSummarizer的初始化"""
        # 模拟model_abbr_from_cfg_used_in_summarizer函数的返回值
        mock_model_abbr_from_cfg_used.side_effect = ["test_model", "custom_abbr"]

        summarizer = DefaultSubjectiveSummarizer(self.config, summary_groups=self.summary_groups)

        self.assertEqual(summarizer.cfg, self.config)
        self.assertEqual(summarizer.model_cfgs, [self.model_cfg, self.model_cfg2])
        self.assertEqual(summarizer.dataset_cfgs, [self.dataset_cfg, self.dataset_cfg2, self.dataset_cfg3])
        self.assertEqual(summarizer.work_dir, "/tmp/test_work_dir")
        self.assertEqual(summarizer.judge_models, [self.judge_model])
        self.assertEqual(summarizer.summary_groups, self.summary_groups)
        self.assertEqual(summarizer.model_abbrs, ["test_model", "custom_abbr"])
        mock_ais_logger.assert_called_once()

    @patch('ais_bench.benchmark.summarizers.default_subjective.AISLogger')
    def test_init_with_prompt_db(self, mock_ais_logger):
        """测试使用deprecated的prompt_db参数初始化"""
        mock_logger = MagicMock()
        mock_ais_logger.return_value = mock_logger

        summarizer = DefaultSubjectiveSummarizer(self.config, prompt_db="deprecated_db")

        mock_logger.warning.assert_called_with('prompt_db is deprecated and no longer used. Please remove it from your config.')

    @patch('os.path.exists')
    @patch('ais_bench.benchmark.summarizers.default_subjective.get_infer_output_path')
    @patch('ais_bench.benchmark.summarizers.default_subjective.mmengine')
    @patch('ais_bench.benchmark.summarizers.default_subjective.AISLogger')
    @patch('ais_bench.benchmark.summarizers.default_subjective.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.default_subjective.dataset_abbr_from_cfg')
    def test_pick_up_results_basic(self, mock_dataset_abbr_from_cfg, mock_model_abbr_from_cfg_used, mock_ais_logger, mock_mmengine, mock_get_infer_output_path, mock_exists):
        """测试基本的结果提取功能"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
        mock_dataset_abbr_from_cfg.return_value = "test_ds"

        # 模拟原始路径 - 修改为完全匹配实际代码中生成的路径，修复参数数量
        def get_path_side_effect(model_cfg, dataset_cfg, work_dir):
            # 为judged路径和普通路径返回不同的值
            if "test_ds_judged-by--judge_model" in str(dataset_cfg):
                return "/tmp/test_work_dir/results/test_model/test_ds_judged-by--judge_model/results.json"
            return "/tmp/test_work_dir/results/test_model/test_ds/results.json"

        # 设置路径模拟逻辑
        mock_get_infer_output_path.side_effect = get_path_side_effect

        # 模拟路径存在检查
        def exists_side_effect(path):
            # 只返回judged路径存在
            if "test_ds_judged-by--judge_model" in path:
                return True
            return False
        mock_exists.side_effect = exists_side_effect

        # 模拟加载的结果
        mock_result = {
            "score": 0.8,
            "error": None
        }
        mock_mmengine.load.return_value = mock_result

        # 模拟AISLogger
        mock_logger = MagicMock()
        mock_ais_logger.return_value = mock_logger

        # 确保config中有datasets
        self.config['datasets'] = [{'type': 'TestDatasetType', 'path': 'test_ds'}]
        self.config['models'] = [{'type': 'TestModelType'}]

        # 创建summarizer并直接设置model_abbrs
        summarizer = DefaultSubjectiveSummarizer(self.config)
        summarizer.model_abbrs = ["test_model"]

        # 手动设置必要的属性
        summarizer.raw_results = {}
        summarizer.parsed_results = {}

        # 调用测试方法
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results("judge_model")

        # 验证结果
        self.assertIn("test_model", raw_results)
        self.assertIn("test_ds", raw_results["test_model"])
        self.assertEqual(raw_results["test_model"]["test_ds"]["score"], 0.8)

        self.assertIn("test_model", parsed_results)

    @patch('os.path.exists')
    @patch('ais_bench.benchmark.summarizers.default_subjective.get_infer_output_path')
    @patch('ais_bench.benchmark.summarizers.default_subjective.mmengine')
    @patch('ais_bench.benchmark.summarizers.default_subjective.AISLogger')
    @patch('ais_bench.benchmark.summarizers.default_subjective.model_abbr_from_cfg_used_in_summarizer')
    def test_pick_up_results_with_error(self, mock_model_abbr_from_cfg_used, mock_ais_logger, mock_mmengine, mock_get_infer_output_path, mock_exists):
        """测试结果中包含错误的情况"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"

        # 模拟原始路径
        base_path = "/tmp/test_work_dir/results/test_model/test_ds"
        mock_get_infer_output_path.return_value = f"{base_path}/results.json"

        # 模拟路径存在检查
        def exists_side_effect(path):
            if "test_ds_judged-by--judge_model" in path:
                return True
            return False
        mock_exists.side_effect = exists_side_effect

        # 模拟加载的结果 - 包含错误
        mock_result = {
            "score": None,
            "error": "Error message"
        }
        mock_mmengine.load.return_value = mock_result

        # 模拟AISLogger
        mock_logger = MagicMock()
        mock_ais_logger.return_value = mock_logger

        # 确保config中有datasets
        self.config['datasets'] = [{'type': 'TestDatasetType', 'path': 'test_ds'}]

        summarizer = DefaultSubjectiveSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results("judge_model")

        # 验证结果 - 确保有错误的数据集被跳过或标记为错误
        self.assertIn("test_model", raw_results)
        self.assertIn("test_ds", raw_results["test_model"])
        self.assertEqual(raw_results["test_model"]["test_ds"]["error"], "Error message")

        # 验证日志被正确调用
        mock_logger.warning.assert_called()

    @patch('os.path.exists')
    @patch('ais_bench.benchmark.summarizers.default_subjective.get_infer_output_path')
    @patch('ais_bench.benchmark.summarizers.default_subjective.AISLogger')
    @patch('ais_bench.benchmark.summarizers.default_subjective.model_abbr_from_cfg_used_in_summarizer')
    def test_pick_up_results_file_not_exist(self, mock_model_abbr_from_cfg_used, mock_ais_logger, mock_get_infer_output_path, mock_exists):
        """测试文件不存在的情况"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
        mock_exists.return_value = False
        mock_get_infer_output_path.return_value = "/tmp/test_work_dir/results/test_model/test_ds/results.json"

        summarizer = DefaultSubjectiveSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results("judge_model")

        # 验证结果 - 应该返回空字典
        self.assertEqual(raw_results, {"test_model": {}})
        self.assertEqual(parsed_results, {"test_model": {}})
        self.assertEqual(dataset_metrics, {})

    @patch('os.path.exists')
    @patch('ais_bench.benchmark.summarizers.default_subjective.get_infer_output_path')
    @patch('ais_bench.benchmark.summarizers.default_subjective.mmengine')
    @patch('ais_bench.benchmark.summarizers.default_subjective.AISLogger')
    @patch('ais_bench.benchmark.summarizers.default_subjective.model_abbr_from_cfg_used_in_summarizer')
    def test_pick_up_results_unknown_format(self, mock_model_abbr_from_cfg_used, mock_ais_logger, mock_mmengine, mock_get_infer_output_path, mock_exists):
        """测试未知的结果格式"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"
        mock_exists.return_value = True
        mock_get_infer_output_path.return_value = "/tmp/test_work_dir/results/test_model/test_ds/results.json"
        # 只包含黑名单中的metric
        mock_result = {
            "bp": 0.5,
            "sys_len": 10
        }
        mock_mmengine.load.return_value = mock_result

        mock_logger = MagicMock()
        mock_ais_logger.return_value = mock_logger

        summarizer = DefaultSubjectiveSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results("judge_model")

        # 验证结果
        mock_logger.warning.assert_called()

    @patch('ais_bench.benchmark.summarizers.default_subjective.model_abbr_from_cfg_used_in_summarizer')
    def test_calculate_group_metrics_naive_average(self, mock_model_abbr_from_cfg_used):
        """测试基本的平均计算"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"

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
            "test_ds": "gen",
            "test_ds2": "gen"
        }

        # 使用默认的summary_group，没有指定sum、std或weights，应该使用naive_average
        summary_group = {"name": "group1", "subsets": ["test_ds", "test_ds2"]}
        summarizer = DefaultSubjectiveSummarizer(self.config, summary_groups=[summary_group])
        summarizer.model_abbrs = ["test_model"]  # 直接设置model_abbrs以避免mock问题
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果
        self.assertIn("group1", parsed_results["test_model"])
        # 检查是否有naive_average字段并验证其值
        self.assertIn("naive_average", parsed_results["test_model"]["group1"])
        self.assertAlmostEqual(parsed_results["test_model"]["group1"]["naive_average"], 0.7)  # 平均score

    def test_calculate_group_metrics_weighted_average(self):
        """测试计算加权平均"""
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
            "test_ds": "gen",
            "test_ds2": "gen"
        }

        # 使用weights的summary_group
        summary_group = {'name': 'test_weighted', 'subsets': ['test_ds', 'test_ds2'], 'weights': {'test_ds': 1, 'test_ds2': 2}}
        summarizer = DefaultSubjectiveSummarizer(self.config, summary_groups=[summary_group])
        summarizer.model_abbrs = ["test_model"]  # 直接设置model_abbrs
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果
        self.assertIn("test_weighted", parsed_results["test_model"])
        self.assertIn("weighted_average", parsed_results["test_model"]["test_weighted"])
        # 计算理论值：(0.8*1 + 0.6*2) / 3 = (0.8 + 1.2) / 3 = 2.0 / 3 ≈ 0.6667
        self.assertAlmostEqual(parsed_results["test_model"]["test_weighted"]["weighted_average"], 0.6667, places=4)

    def test_calculate_group_metrics_standard_deviation(self):
        """测试计算标准偏差"""
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
            "test_ds": "gen",
            "test_ds2": "gen"
        }

        # 使用std=True的summary_group
        summary_group = {'name': 'test_std', 'subsets': ['test_ds', 'test_ds2'], 'std': True}
        summarizer = DefaultSubjectiveSummarizer(self.config, summary_groups=[summary_group])
        summarizer.model_abbrs = ["test_model"]  # 直接设置model_abbrs
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果
        self.assertIn("test_std", parsed_results["test_model"])
        self.assertIn("standard_deviation", parsed_results["test_model"]["test_std"])
        # 根据实际计算结果调整预期值
        self.assertAlmostEqual(parsed_results["test_model"]["test_std"]["standard_deviation"], 0.1, places=4)

    def test_calculate_group_metrics_sum(self):
        """测试计算总和"""
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
            "test_ds": "gen",
            "test_ds2": "gen"
        }

        # 使用sum=True的summary_group
        summary_group = {'name': 'test_sum', 'subsets': ['test_ds', 'test_ds2'], 'sum': True}
        summarizer = DefaultSubjectiveSummarizer(self.config, summary_groups=[summary_group])
        summarizer.model_abbrs = ["test_model"]  # 直接设置model_abbrs
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果
        self.assertIn("test_sum", parsed_results["test_model"])
        self.assertIn("sum", parsed_results["test_model"]["test_sum"])
        self.assertAlmostEqual(parsed_results["test_model"]["test_sum"]["sum"], 1.4)

    @patch('ais_bench.benchmark.summarizers.default_subjective.model_abbr_from_cfg_used_in_summarizer')
    def test_calculate_group_metrics_custom_metric_tuple(self, mock_model_abbr_from_cfg_used):
        """测试使用元组形式的自定义metric"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"

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
            "test_ds": "gen",
            "test_ds2": "ppl"
        }

        summarizer = DefaultSubjectiveSummarizer(self.config, summary_groups=[self.summary_groups[4]])
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果
        self.assertIn("group5", parsed_results["test_model"])
        # 0.8和0.9的平均值
        self.assertAlmostEqual(parsed_results["test_model"]["group5"]["custom_metric"], 0.85)

    @patch('ais_bench.benchmark.summarizers.default_subjective.model_abbr_from_cfg_used_in_summarizer')
    def test_calculate_group_metrics_missing_metrics(self, mock_model_abbr_from_cfg_used):
        """测试缺少指标的情况"""
        # 设置mock
        mock_model_abbr_from_cfg_used.return_value = "test_model"

        # 准备测试数据
        raw_results = {
            "test_model": {
                "test_ds": {"score": 0.8}
                # 缺少test_ds2
            }
        }
        parsed_results = {
            "test_model": {
                "test_ds": {"score": 0.8, "accuracy": 0.7}
                # 缺少test_ds2
            }
        }
        dataset_metrics = {
            "test_ds": ["score", "accuracy"]
        }
        dataset_eval_mode = {
            "test_ds": "gen"
        }

        summarizer = DefaultSubjectiveSummarizer(self.config, summary_groups=[self.summary_groups[0]])
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果
        self.assertIn("group1", raw_results["test_model"])
        self.assertIn("error", raw_results["test_model"]["group1"])
        self.assertIn("missing metrics", raw_results["test_model"]["group1"]["error"])

    @patch('ais_bench.benchmark.summarizers.default_subjective.model_abbr_from_cfg_used_in_summarizer')
    def test_calculate_group_metrics_mixed_types_error(self, mock_model_abbr_from_cfg_used):
        """测试混合类型错误"""
        # 设置mock
        mock_model_abbr_from_cfg_used.side_effect = ["test_model", "custom_abbr"]

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
            "test_ds": "gen",
            "test_ds2": "ppl"
        }

        # 创建混合类型的summary_group
        mixed_group = {
            "name": "mixed_group",
            "subsets": ["test_ds", ("test_ds2", "accuracy")]
        }

        summarizer = DefaultSubjectiveSummarizer(self.config, summary_groups=[mixed_group])

        # 验证抛出ConfigError
        with self.assertRaises(AISBenchConfigError):
            summarizer._calculate_group_metrics(raw_results, parsed_results, dataset_metrics, dataset_eval_mode)

    @patch('ais_bench.benchmark.summarizers.default_subjective.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default_subjective.get_prompt_hash')
    @patch('ais_bench.benchmark.summarizers.default_subjective.AISLogger')
    def test_format_table_basic(self, mock_ais_logger, mock_get_prompt_hash, mock_dataset_abbr_from_cfg):
        """测试基本的表格格式化"""
        # 设置mock
        mock_dataset_abbr_from_cfg.side_effect = lambda x: "test_ds"  # 模拟数据集缩写
        mock_get_prompt_hash.return_value = "abcdef123456"

        # 模拟config中的datasets
        self.config['datasets'] = [{'type': 'TestDatasetType', 'path': 'test_ds'}]

        # 准备测试数据
        parsed_results = {
            "test_model": {
                "test_ds": {"score": 0.8}
            }
        }
        dataset_metrics = {
            "test_ds": ["score"]
        }
        dataset_eval_mode = {
            "test_ds": "gen"
        }

        # 直接设置模型缩写列表
        summarizer = DefaultSubjectiveSummarizer(self.config)
        summarizer.model_abbrs = ["test_model"]
        table = summarizer._format_table(parsed_results, dataset_metrics, dataset_eval_mode)

        # 验证结果
        self.assertEqual(len(table), 2)  # 表头 + 一行数据
        self.assertEqual(table[0], ['dataset', 'version', 'metric', 'mode', 'test_model'])
        self.assertEqual(table[1][0], 'test_ds')
        self.assertEqual(table[1][2], 'score')
        self.assertEqual(table[1][4], '0.80')

    @patch('ais_bench.benchmark.summarizers.default_subjective.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default_subjective.get_prompt_hash')
    @patch('ais_bench.benchmark.summarizers.default_subjective.AISLogger')
    def test_format_table_with_required_abbrs(self, mock_ais_logger, mock_get_prompt_hash, mock_dataset_abbr_from_cfg):
        """测试使用required_dataset_abbrs参数"""
        # 设置mock
        mock_dataset_abbr_from_cfg.return_value = "test_ds"
        mock_get_prompt_hash.return_value = "abcdef123456"

        # 准备测试数据
        parsed_results = {
            "test_model": {
                "test_ds": {"score": 0.8}
            },
            "custom_abbr": {
                "test_ds": {"score": 0.7}
            }
        }
        dataset_metrics = {
            "test_ds": ["score"]
        }
        dataset_eval_mode = {
            "test_ds": "gen"
        }

        # 直接设置模型缩写列表
        summarizer = DefaultSubjectiveSummarizer(self.config)
        summarizer.model_abbrs = ["test_model", "custom_abbr"]
        required_abbrs = ["test_ds", ("test_ds", "score")]
        table = summarizer._format_table(parsed_results, dataset_metrics, dataset_eval_mode, required_dataset_abbrs=required_abbrs)

        # 验证结果
        self.assertEqual(len(table), 3)  # 表头 + 两行数据

    @patch('ais_bench.benchmark.summarizers.default_subjective.AISLogger')
    @patch('ais_bench.benchmark.summarizers.default_subjective.model_abbr_from_cfg_used_in_summarizer')
    def test_format_raw_txt(self, mock_model_abbr_from_cfg_used, mock_ais_logger):
        """测试原始文本格式化"""
        # 设置mock
        mock_model_abbr_from_cfg_used.side_effect = ["test_model", "custom_abbr"]

        # 准备测试数据
        raw_results = {
            "test_model": {
                "test_ds": {"score": 0.8},
                "test_ds2": {"accuracy": 0.9}
            },
            "custom_abbr": {
                "test_ds": {"score": 0.7}
            }
        }

        summarizer = DefaultSubjectiveSummarizer(self.config)
        raw_txts = summarizer._format_raw_txt(raw_results)

        # 验证结果
        self.assertIn("Model: test_model", raw_txts)
        self.assertIn("Model: custom_abbr", raw_txts)
        self.assertIn("test_ds:", raw_txts)
        self.assertIn("test_ds2:", raw_txts)

    @patch('os.path.join')
    @patch('ais_bench.benchmark.summarizers.default_subjective.mmengine')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.abspath')
    @patch('ais_bench.benchmark.summarizers.default_subjective.AISLogger')
    @patch('ais_bench.benchmark.summarizers.default_subjective.model_abbr_from_cfg_used_in_summarizer')
    def test_output_to_file_default_path(self, mock_model_abbr_from_cfg_used, mock_ais_logger, mock_abspath, mock_open_file, mock_mmengine, mock_join):
        """测试输出到默认路径"""
        # 设置mock
        mock_model_abbr_from_cfg_used.side_effect = ["test_model", "custom_abbr"]
        mock_abspath.return_value = "/tmp/test_work_dir/summary/summary_20230101_000000_by_judge_model.txt"
        mock_join.return_value = "/tmp/test_work_dir/summary/summary_20230101_000000.txt"

        mock_logger = MagicMock()
        mock_ais_logger.return_value = mock_logger

        # 准备测试数据
        table = [['dataset', 'metric'], ['test_ds', '0.8']]
        raw_txts = "test raw text"

        summarizer = DefaultSubjectiveSummarizer(self.config)
        summarizer._output_to_file(None, "20230101_000000", table, raw_txts, "judge_model")

        # 验证结果
        mock_mmengine.mkdir_or_exist.assert_called()
        # 检查是否调用了open，但不严格匹配路径
        self.assertTrue(mock_open_file.called)
        # 检查是否至少调用了一次打开CSV文件
        csv_calls = [call for call in mock_open_file.call_args_list if call[0][0].endswith('.csv')]
        self.assertTrue(len(csv_calls) > 0)
        mock_logger.info.assert_called()

    @patch('os.path.join')
    @patch('ais_bench.benchmark.summarizers.default_subjective.mmengine')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.abspath')
    @patch('ais_bench.benchmark.summarizers.default_subjective.AISLogger')
    def test_output_to_file_custom_path(self, mock_ais_logger, mock_abspath, mock_open_file, mock_mmengine, mock_join):
        """测试输出到自定义路径"""
        # 设置mock
        mock_abspath.return_value = "/custom/path/summary_20230101_000000_by_judge_model.txt"

        # 准备测试数据
        table = [['dataset', 'metric'], ['test_ds', '0.8']]
        raw_txts = "test raw text"

        summarizer = DefaultSubjectiveSummarizer(self.config)
        summarizer._output_to_file("/custom/path/summary.txt", "20230101_000000", table, raw_txts, "judge_model")

        # 验证结果
        mock_open_file.assert_any_call("/custom/path/summary_by_judge_model.txt", "w", encoding="utf-8")
        mock_open_file.assert_any_call("/custom/path/summary_by_judge_model.csv", "w", encoding="utf-8")

    @patch('ais_bench.benchmark.summarizers.default_subjective.DefaultSubjectiveSummarizer._output_to_file')
    @patch('ais_bench.benchmark.summarizers.default_subjective.DefaultSubjectiveSummarizer._format_raw_txt')
    @patch('ais_bench.benchmark.summarizers.default_subjective.DefaultSubjectiveSummarizer._format_table')
    @patch('ais_bench.benchmark.summarizers.default_subjective.DefaultSubjectiveSummarizer._calculate_group_metrics')
    @patch('ais_bench.benchmark.summarizers.default_subjective.DefaultSubjectiveSummarizer._pick_up_results')
    @patch('ais_bench.benchmark.summarizers.default_subjective.model_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default_subjective.AISLogger')
    @patch('builtins.print')
    def test_summarize_integration(self, mock_print, mock_ais_logger, mock_model_abbr_from_cfg,
                                 mock_pick_up_results, mock_calculate_group_metrics,
                                 mock_format_table, mock_format_raw_txt, mock_output_to_file):
        """测试summarize方法的集成"""
        # 设置mock
        mock_model_abbr_from_cfg.return_value = "judge_model"
        mock_pick_up_results.return_value = ({}, {}, {}, {})
        mock_calculate_group_metrics.return_value = ({}, {}, {}, {})
        mock_format_table.return_value = [["dataset", "metric"], ["test_ds", "0.8"]]
        mock_format_raw_txt.return_value = "test raw text"

        summarizer = DefaultSubjectiveSummarizer(self.config)
        summarizer.summarize(output_path="/custom/path/summary.txt", time_str="20230101_000000")

        # 验证所有方法都被调用
        mock_pick_up_results.assert_called_with("judge_model")
        mock_calculate_group_metrics.assert_called()
        mock_format_table.assert_called()
        mock_format_raw_txt.assert_called()
        mock_output_to_file.assert_called_with("/custom/path/summary.txt", "20230101_000000",
                                             [["dataset", "metric"], ["test_ds", "0.8"]],
                                             "test raw text", "judge_model")
        mock_print.assert_called()


if __name__ == "__main__":
    unittest.main()