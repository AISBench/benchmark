import unittest
from unittest.mock import patch, MagicMock
from ais_bench.benchmark.cli.utils import (
    fill_model_path_if_datasets_need,
    fill_test_range_use_num_prompts,
    get_config_type,
    is_running_in_background,
    get_current_time_str
)
from ais_bench.benchmark.utils.logging.exceptions import ConfigError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES


class TestUtils(unittest.TestCase):
    def test_fill_model_path_if_datasets_need_synthetic_dataset(self):
        """测试当数据集是SyntheticDataset时，成功添加model_path"""
        # 准备数据
        model_cfg = {"path": "/path/to/model"}
        dataset_cfg = {
            "type": "ais_bench.benchmark.datasets.synthetic.SyntheticDataset"
        }

        # 调用函数
        fill_model_path_if_datasets_need(model_cfg, dataset_cfg)

        # 验证结果
        self.assertEqual(dataset_cfg.get("model_path"), "/path/to/model")

    def test_fill_model_path_if_datasets_need_sharegpt_dataset(self):
        """测试当数据集是ShareGPTDataset时，成功添加model_path"""
        # 准备数据
        model_cfg = {"path": "/path/to/model"}
        dataset_cfg = {
            "type": "ais_bench.benchmark.datasets.sharegpt.ShareGPTDataset"
        }

        # 调用函数
        fill_model_path_if_datasets_need(model_cfg, dataset_cfg)

        # 验证结果
        self.assertEqual(dataset_cfg.get("model_path"), "/path/to/model")

    def test_fill_model_path_if_datasets_need_missing_model_path(self):
        """测试当数据集需要模型但缺少model_path时，抛出ConfigError"""
        # 准备数据
        model_cfg = {}
        dataset_cfg = {
            "type": "ais_bench.benchmark.datasets.synthetic.SyntheticDataset"
        }

        # 验证异常
        with self.assertRaises(ConfigError) as context:
            fill_model_path_if_datasets_need(model_cfg, dataset_cfg)

        # 验证错误信息
        self.assertIn(UTILS_CODES.SYNTHETIC_DS_MISS_REQUIRED_PARAM.full_code, str(context.exception))
        self.assertIn("[path] in model config is required", str(context.exception))

    def test_fill_model_path_if_datasets_need_not_required_dataset(self):
        """测试当数据集不需要模型时，不做任何操作"""
        # 准备数据
        model_cfg = {"path": "/path/to/model"}
        dataset_cfg = {
            "type": "ais_bench.benchmark.datasets.custom.CustomDataset"
        }
        original_dataset_cfg = dataset_cfg.copy()

        # 调用函数
        fill_model_path_if_datasets_need(model_cfg, dataset_cfg)

        # 验证没有修改
        self.assertEqual(dataset_cfg, original_dataset_cfg)
        self.assertNotIn("model_path", dataset_cfg)

    @patch('ais_bench.benchmark.cli.utils.get_config_type')
    def test_fill_model_path_if_datasets_need_with_class_object(self, mock_get_config_type):
        """测试当dataset_cfg的type是类对象而不是字符串时的情况"""
        # 模拟get_config_type返回值
        mock_get_config_type.return_value = "ais_bench.benchmark.datasets.synthetic.SyntheticDataset"

        # 准备数据
        model_cfg = {"path": "/path/to/model"}
        dataset_cfg = {
            "type": object(),  # 模拟类对象
        }

        # 调用函数
        fill_model_path_if_datasets_need(model_cfg, dataset_cfg)

        # 验证结果
        self.assertEqual(dataset_cfg.get("model_path"), "/path/to/model")
        mock_get_config_type.assert_called_once_with(dataset_cfg.get("type"))

    def test_get_config_type_with_string(self):
        """测试get_config_type函数处理字符串类型"""
        # 测试字符串类型
        self.assertEqual(get_config_type("test_string"), "test_string")

    def test_get_config_type_with_class(self):
        """测试get_config_type函数处理类类型"""
        # 测试类类型
        class TestClass:
            pass

        expected_type = f"{TestClass.__module__}.{TestClass.__name__}"
        self.assertEqual(get_config_type(TestClass), expected_type)

    @patch('sys.stdin.isatty')
    @patch('sys.stdout.isatty')
    def test_is_running_in_background_true(self, mock_stdout_isatty, mock_stdin_isatty):
        """测试is_running_in_background函数返回True的情况"""
        # 模拟stdin和stdout都不是TTY
        mock_stdin_isatty.return_value = False
        mock_stdout_isatty.return_value = False

        result = is_running_in_background()
        self.assertTrue(result)

    @patch('sys.stdin.isatty')
    @patch('sys.stdout.isatty')
    def test_is_running_in_background_false(self, mock_stdout_isatty, mock_stdin_isatty):
        """测试is_running_in_background函数返回False的情况"""
        # 模拟stdin和stdout都是TTY
        mock_stdin_isatty.return_value = True
        mock_stdout_isatty.return_value = True

        result = is_running_in_background()
        self.assertFalse(result)

    @patch('sys.stdin.isatty')
    @patch('sys.stdout.isatty')
    def test_is_running_in_background_mixed(self, mock_stdout_isatty, mock_stdin_isatty):
        """测试is_running_in_background函数在混合情况下的行为"""
        # 模拟stdin是TTY但stdout不是TTY
        mock_stdin_isatty.return_value = True
        mock_stdout_isatty.return_value = False

        result = is_running_in_background()
        self.assertTrue(result)

    @patch('ais_bench.benchmark.cli.utils.datetime')
    def test_get_current_time_str(self, mock_datetime):
        """测试get_current_time_str函数"""
        # 模拟datetime.now()返回一个固定的datetime对象
        mock_now = MagicMock()
        mock_now.strftime.return_value = "20231201_143022"
        mock_datetime.now.return_value = mock_now

        result = get_current_time_str()

        # 验证结果
        self.assertEqual(result, "20231201_143022")
        mock_now.strftime.assert_called_once_with("%Y%m%d_%H%M%S")

    @patch('ais_bench.benchmark.cli.utils.logger')
    def test_fill_test_range_use_num_prompts_with_num_prompts(self, mock_logger):
        """测试fill_test_range_use_num_prompts函数，有num_prompts时设置test_range"""
        # 准备数据
        dataset_cfg = {
            "reader_cfg": {},
            "abbr": "test_dataset"
        }
        num_prompts = 10

        # 调用函数
        fill_test_range_use_num_prompts(num_prompts, dataset_cfg)

        # 验证结果
        self.assertEqual(dataset_cfg["reader_cfg"].get("test_range"), "[:10]")
        mock_logger.info.assert_called_once_with("Keeping the first 10 prompts for dataset [test_dataset]")

    @patch('ais_bench.benchmark.cli.utils.logger')
    def test_fill_test_range_use_num_prompts_with_existing_test_range(self, mock_logger):
        """测试fill_test_range_use_num_prompts函数，test_range已存在时发出警告"""
        # 准备数据
        dataset_cfg = {
            "reader_cfg": {"test_range": "[0:100]"},
            "abbr": "test_dataset"
        }
        num_prompts = 10

        # 调用函数
        fill_test_range_use_num_prompts(num_prompts, dataset_cfg)

        # 验证结果
        self.assertEqual(dataset_cfg["reader_cfg"].get("test_range"), "[0:100]")  # 不应该被修改
        mock_logger.warning.assert_called_once_with("`test_range` has been set, `--num-prompts` will be ignored")

    @patch('ais_bench.benchmark.cli.utils.logger')
    def test_fill_test_range_use_num_prompts_no_num_prompts(self, mock_logger):
        """测试fill_test_range_use_num_prompts函数，没有num_prompts时不操作"""
        # 准备数据
        dataset_cfg = {
            "reader_cfg": {},
            "abbr": "test_dataset"
        }
        num_prompts = None

        # 调用函数
        fill_test_range_use_num_prompts(num_prompts, dataset_cfg)

        # 验证结果
        self.assertNotIn("test_range", dataset_cfg["reader_cfg"])
        mock_logger.info.assert_not_called()
        mock_logger.warning.assert_not_called()

    @patch('ais_bench.benchmark.cli.utils.logger')
    def test_fill_test_range_use_num_prompts_zero_num_prompts(self, mock_logger):
        """测试fill_test_range_use_num_prompts函数，num_prompts为0时不操作"""
        # 准备数据
        dataset_cfg = {
            "reader_cfg": {},
            "abbr": "test_dataset"
        }
        num_prompts = 0

        # 调用函数
        fill_test_range_use_num_prompts(num_prompts, dataset_cfg)

        # 验证结果
        self.assertNotIn("test_range", dataset_cfg["reader_cfg"])
        mock_logger.info.assert_not_called()
        mock_logger.warning.assert_not_called()

    @patch('ais_bench.benchmark.cli.utils.logger')
    def test_fill_test_range_use_num_prompts_with_string_num_prompts(self, mock_logger):
        """测试fill_test_range_use_num_prompts函数，num_prompts为字符串时设置test_range"""
        # 准备数据
        dataset_cfg = {
            "reader_cfg": {},
            "abbr": "test_dataset"
        }
        num_prompts = "10"  # 字符串类型

        # 调用函数
        fill_test_range_use_num_prompts(num_prompts, dataset_cfg)

        # 验证结果 - 字符串会被转换为 "[:10]"
        self.assertEqual(dataset_cfg["reader_cfg"].get("test_range"), "[:10]")
        mock_logger.info.assert_called_once_with("Keeping the first 10 prompts for dataset [test_dataset]")


if __name__ == '__main__':
    unittest.main()