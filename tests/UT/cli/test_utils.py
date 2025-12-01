import unittest
from unittest.mock import patch, MagicMock
from ais_bench.benchmark.cli.utils import (
    fill_model_path_if_datasets_need,
    fill_test_range_use_num_prompts,
    get_config_type,
    get_current_time_str,
    validate_max_workers,
    validate_max_workers_per_gpu,
    validate_num_prompts,
    validate_num_warmups,
    validate_pressure_time,
    MAX_NUM_WORKERS,
    DEFAULT_PRESSURE_TIME
)
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError
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
        with self.assertRaises(AISBenchConfigError) as context:
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


class TestValidateMaxWorkers(unittest.TestCase):
    """测试 validate_max_workers 函数"""

    @patch('ais_bench.benchmark.cli.utils.logger')
    def test_validate_max_workers_valid_value(self, mock_logger):
        """测试有效的 max_workers 值"""
        result = validate_max_workers("10")
        self.assertEqual(result, 10)
        mock_logger.warning.assert_not_called()

    def test_validate_max_workers_exceeds_max(self):
        """测试超过最大值的 max_workers"""
        large_value = MAX_NUM_WORKERS + 10
        with self.assertRaises(AISBenchConfigError) as context:
            validate_max_workers(str(large_value))
        self.assertIn(UTILS_CODES.ARGUMENT_TOO_LARGE.full_code, str(context.exception))
        self.assertIn("must be <=", str(context.exception))

    def test_validate_max_workers_less_than_one(self):
        """测试小于1的 max_workers"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_max_workers("0")
        self.assertIn(UTILS_CODES.ARGUMENT_TOO_SMALL.full_code, str(context.exception))
        self.assertIn("must be >=", str(context.exception))

    def test_validate_max_workers_invalid_type(self):
        """测试无效类型的 max_workers"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_max_workers("invalid")
        self.assertIn(UTILS_CODES.INVALID_INTEGER_TYPE.full_code, str(context.exception))
        self.assertIn("must be an integer", str(context.exception))

    def test_validate_max_workers_negative(self):
        """测试负数的 max_workers"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_max_workers("-5")
        self.assertIn(UTILS_CODES.ARGUMENT_TOO_SMALL.full_code, str(context.exception))
        self.assertIn("must be >=", str(context.exception))


class TestValidateMaxWorkersPerGpu(unittest.TestCase):
    """测试 validate_max_workers_per_gpu 函数"""

    @patch('ais_bench.benchmark.cli.utils.logger')
    def test_validate_max_workers_per_gpu_valid_value(self, mock_logger):
        """测试有效的 max_workers_per_gpu 值"""
        result = validate_max_workers_per_gpu("5")
        self.assertEqual(result, 5)
        mock_logger.warning.assert_not_called()

    def test_validate_max_workers_per_gpu_less_than_one(self):
        """测试小于1的 max_workers_per_gpu"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_max_workers_per_gpu("0")
        self.assertIn(UTILS_CODES.ARGUMENT_TOO_SMALL.full_code, str(context.exception))
        self.assertIn("must be >=", str(context.exception))

    def test_validate_max_workers_per_gpu_invalid_type(self):
        """测试无效类型的 max_workers_per_gpu"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_max_workers_per_gpu("invalid")
        self.assertIn(UTILS_CODES.INVALID_INTEGER_TYPE.full_code, str(context.exception))
        self.assertIn("must be an integer", str(context.exception))

    def test_validate_max_workers_per_gpu_negative(self):
        """测试负数的 max_workers_per_gpu"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_max_workers_per_gpu("-3")
        self.assertIn(UTILS_CODES.ARGUMENT_TOO_SMALL.full_code, str(context.exception))
        self.assertIn("must be >=", str(context.exception))


class TestValidateNumPrompts(unittest.TestCase):
    """测试 validate_num_prompts 函数"""

    @patch('ais_bench.benchmark.cli.utils.logger')
    def test_validate_num_prompts_valid_value(self, mock_logger):
        """测试有效的 num_prompts 值"""
        result = validate_num_prompts("100")
        self.assertEqual(result, 100)
        mock_logger.warning.assert_not_called()

    def test_validate_num_prompts_less_than_one(self):
        """测试小于1的 num_prompts"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_num_prompts("0")
        self.assertIn(UTILS_CODES.ARGUMENT_TOO_SMALL.full_code, str(context.exception))
        self.assertIn("must be >=", str(context.exception))

    def test_validate_num_prompts_invalid_type(self):
        """测试无效类型的 num_prompts"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_num_prompts("invalid")
        self.assertIn(UTILS_CODES.INVALID_INTEGER_TYPE.full_code, str(context.exception))
        self.assertIn("must be an integer", str(context.exception))

    def test_validate_num_prompts_negative(self):
        """测试负数的 num_prompts"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_num_prompts("-5")
        self.assertIn(UTILS_CODES.ARGUMENT_TOO_SMALL.full_code, str(context.exception))
        self.assertIn("must be >=", str(context.exception))


class TestValidateNumWarmups(unittest.TestCase):
    """测试 validate_num_warmups 函数"""

    @patch('ais_bench.benchmark.cli.utils.logger')
    def test_validate_num_warmups_valid_value(self, mock_logger):
        """测试有效的 num_warmups 值"""
        result = validate_num_warmups("5")
        self.assertEqual(result, 5)
        mock_logger.warning.assert_not_called()

    @patch('ais_bench.benchmark.cli.utils.logger')
    def test_validate_num_warmups_zero(self, mock_logger):
        """测试 num_warmups 为0（允许）"""
        result = validate_num_warmups("0")
        self.assertEqual(result, 0)
        mock_logger.warning.assert_not_called()

    def test_validate_num_warmups_less_than_zero(self):
        """测试小于0的 num_warmups"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_num_warmups("-1")
        self.assertIn(UTILS_CODES.ARGUMENT_TOO_SMALL.full_code, str(context.exception))
        self.assertIn("must be >=", str(context.exception))

    def test_validate_num_warmups_invalid_type(self):
        """测试无效类型的 num_warmups"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_num_warmups("invalid")
        self.assertIn(UTILS_CODES.INVALID_INTEGER_TYPE.full_code, str(context.exception))
        self.assertIn("must be an integer", str(context.exception))


class TestValidatePressureTime(unittest.TestCase):
    """测试 validate_pressure_time 函数"""

    @patch('ais_bench.benchmark.cli.utils.logger')
    def test_validate_pressure_time_valid_value(self, mock_logger):
        """测试有效的 pressure_time 值"""
        result = validate_pressure_time("30")
        self.assertEqual(result, 30)
        mock_logger.warning.assert_not_called()

    def test_validate_pressure_time_less_than_one(self):
        """测试小于1的 pressure_time"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_pressure_time("0")
        self.assertIn(UTILS_CODES.ARGUMENT_TOO_SMALL.full_code, str(context.exception))
        self.assertIn("must be >=", str(context.exception))

    def test_validate_pressure_time_invalid_type(self):
        """测试无效类型的 pressure_time"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_pressure_time("invalid")
        self.assertIn(UTILS_CODES.INVALID_INTEGER_TYPE.full_code, str(context.exception))
        self.assertIn("must be an integer", str(context.exception))

    def test_validate_pressure_time_negative(self):
        """测试负数的 pressure_time"""
        with self.assertRaises(AISBenchConfigError) as context:
            validate_pressure_time("-5")
        self.assertIn(UTILS_CODES.ARGUMENT_TOO_SMALL.full_code, str(context.exception))
        self.assertIn("must be >=", str(context.exception))


if __name__ == '__main__':
    unittest.main()