import os
import unittest
from unittest import mock
import tempfile
import shutil

from ais_bench.benchmark.cli.config_manager import CustomConfigChecker, ConfigManager
from ais_bench.benchmark.models import VLLMCustomAPI, VLLMCustomAPIChat
from ais_bench.benchmark.utils.logging.exceptions import CommandError, AISBenchConfigError
from ais_bench.benchmark.utils.logging.error_codes import TMAN_CODES

class TestCustomConfigChecker(unittest.TestCase):
    def setUp(self):
        self.file_path = 'test_config.py'

    def test_check_valid_config(self):
        """测试有效配置的检查"""
        valid_config = {
            'models': [{'type': 'test_model', 'abbr': 'test', 'attr': {}}],
            'datasets': [{'type': 'test_dataset', 'abbr': 'test', 'reader_cfg': {}, 'infer_cfg': {}, 'eval_cfg': {}}],
            'summarizer': {'attr': {}}
        }
        checker = CustomConfigChecker(valid_config, self.file_path)
        # 不抛出异常即为通过
        checker.check()

    def test_check_missing_models(self):
        """测试缺少models配置"""
        invalid_config = {
            'datasets': [{'type': 'test_dataset', 'abbr': 'test', 'reader_cfg': {}, 'infer_cfg': {}, 'eval_cfg': {}}],
            'summarizer': {'attr': {}}
        }
        checker = CustomConfigChecker(invalid_config, self.file_path)
        with self.assertRaises(AISBenchConfigError) as cm:
            checker.check()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM.full_code)

    def test_check_models_not_list(self):
        """测试models不是列表类型"""
        invalid_config = {
            'models': {'type': 'test_model'},  # 应该是列表
            'datasets': [{'type': 'test_dataset', 'abbr': 'test', 'reader_cfg': {}, 'infer_cfg': {}, 'eval_cfg': {}}],
            'summarizer': {'attr': {}}
        }
        checker = CustomConfigChecker(invalid_config, self.file_path)
        with self.assertRaises(AISBenchConfigError) as cm:
            checker.check()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.TYPE_ERROR_IN_CFG_PARAM.full_code)

    def test_check_model_not_dict(self):
        """测试models中的元素不是字典类型"""
        invalid_config = {
            'models': ['test_model'],  # 应该是字典列表
            'datasets': [{'type': 'test_dataset', 'abbr': 'test', 'reader_cfg': {}, 'infer_cfg': {}, 'eval_cfg': {}}],
            'summarizer': {'attr': {}}
        }
        checker = CustomConfigChecker(invalid_config, self.file_path)
        with self.assertRaises(AISBenchConfigError) as cm:
            checker.check()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.TYPE_ERROR_IN_CFG_PARAM.full_code)

    def test_check_model_missing_required_field(self):
        """测试model缺少必需字段"""
        invalid_config = {
            'models': [{'type': 'test_model'}],  # 缺少abbr字段
            'datasets': [{'type': 'test_dataset', 'abbr': 'test', 'reader_cfg': {}, 'infer_cfg': {}, 'eval_cfg': {}}],
            'summarizer': {'attr': {}}
        }
        checker = CustomConfigChecker(invalid_config, self.file_path)
        with self.assertRaises(AISBenchConfigError) as cm:
            checker.check()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM.full_code)

    def test_check_missing_datasets(self):
        """测试缺少datasets配置"""
        invalid_config = {
            'models': [{'type': 'test_model', 'abbr': 'test', 'attr': {}}],
            'summarizer': {'attr': {}}
        }
        checker = CustomConfigChecker(invalid_config, self.file_path)
        with self.assertRaises(AISBenchConfigError) as cm:
            checker.check()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM.full_code)

    def test_check_datasets_not_list(self):
        """测试datasets不是列表类型"""
        invalid_config = {
            'models': [{'type': 'test_model', 'abbr': 'test', 'attr': {}}],
            'datasets': {'type': 'test_dataset'},  # 应该是列表
            'summarizer': {'attr': {}}
        }
        checker = CustomConfigChecker(invalid_config, self.file_path)
        with self.assertRaises(AISBenchConfigError) as cm:
            checker.check()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.TYPE_ERROR_IN_CFG_PARAM.full_code)

    def test_check_dataset_not_dict(self):
        """测试datasets中的元素不是字典类型"""
        invalid_config = {
            'models': [{'type': 'test_model', 'abbr': 'test', 'attr': {}}],
            'datasets': ['test_dataset'],  # 应该是字典列表
            'summarizer': {'attr': {}}
        }
        checker = CustomConfigChecker(invalid_config, self.file_path)
        with self.assertRaises(AISBenchConfigError) as cm:
            checker.check()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.TYPE_ERROR_IN_CFG_PARAM.full_code)

    def test_check_dataset_missing_required_field(self):
        """测试dataset缺少非必需字段（不应报错）"""
        invalid_config = {
            'models': [{'type': 'test_model', 'abbr': 'test', 'attr': {}}],
            'datasets': [{'type': 'test_dataset', 'abbr': 'test', 'reader_cfg': {}, 'infer_cfg': {}}],  # 缺少eval_cfg
            'summarizer': {'attr': {}}
        }
        checker = CustomConfigChecker(invalid_config, self.file_path)
        # 当前实现仅要求datasets中包含type和abbr，缺少eval_cfg不应抛错
        checker.check()

    def test_check_missing_summarizer(self):
        """测试缺少summarizer配置"""
        invalid_config = {
            'models': [{'type': 'test_model', 'abbr': 'test', 'attr': {}}],
            'datasets': [{'type': 'test_dataset', 'abbr': 'test', 'reader_cfg': {}, 'infer_cfg': {}, 'eval_cfg': {}}]
        }
        checker = CustomConfigChecker(invalid_config, self.file_path)
        with self.assertRaises(AISBenchConfigError) as cm:
            checker.check()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM.full_code)

    def test_check_summarizer_not_dict(self):
        """测试summarizer不是字典类型"""
        invalid_config = {
            'models': [{'type': 'test_model', 'abbr': 'test', 'attr': {}}],
            'datasets': [{'type': 'test_dataset', 'abbr': 'test', 'reader_cfg': {}, 'infer_cfg': {}, 'eval_cfg': {}}],
            'summarizer': 'test_summarizer'  # 应该是字典
        }
        checker = CustomConfigChecker(invalid_config, self.file_path)
        with self.assertRaises(AISBenchConfigError) as cm:
            checker.check()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.TYPE_ERROR_IN_CFG_PARAM.full_code)

    def test_check_summarizer_missing_required_field(self):
        """测试summarizer缺少必需字段"""
        invalid_config = {
            'models': [{'type': 'test_model', 'abbr': 'test', 'attr': {}}],
            'datasets': [{'type': 'test_dataset', 'abbr': 'test', 'reader_cfg': {}, 'infer_cfg': {}, 'eval_cfg': {}}],
            'summarizer': {}  # 缺少attr字段
        }
        checker = CustomConfigChecker(invalid_config, self.file_path)
        with self.assertRaises(AISBenchConfigError) as cm:
            checker.check()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM.full_code)

class TestConfigManager(unittest.TestCase):
    def setUp(self):
        # 创建临时目录作为工作目录
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

        # 创建模拟参数
        self.args = mock.MagicMock()
        self.args.debug = False
        self.args.config_dir = os.path.join(self.temp_dir, 'configs')
        self.args.work_dir = os.path.join(self.temp_dir, 'outputs')
        self.args.dir_time_str = '20230101_000000'
        self.args.reuse = None
        self.args.num_prompts = 10
        self.args.merge_ds = False
        self.args.config = None
        self.args.models = None
        self.args.datasets = None
        self.args.summarizer = None
        self.args.custom_dataset_path = None
        self.args.custom_dataset_infer_method = None
        self.args.custom_dataset_data_type = None
        self.args.custom_dataset_meta_path = None
        self.args.response_anomaly_payload_retention = None

        # Local tokenizer directory consumed by response anomaly model-path
        # fallback tests.
        self.tokenizer_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tokenizer_dir)

        # 创建配置目录结构
        os.makedirs(os.path.join(self.args.config_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.args.config_dir, 'datasets'), exist_ok=True)
        os.makedirs(os.path.join(self.args.config_dir, 'summarizers'), exist_ok=True)

    @mock.patch('ais_bench.benchmark.cli.config_manager.match_cfg_file')
    @mock.patch('tabulate.tabulate')
    def test_search_configs_location(self, mock_tabulate, mock_match_cfg_file):
        """测试搜索配置文件位置"""
        # 配置模拟返回值
        mock_match_cfg_file.side_effect = [
            [('test_model', os.path.join(self.args.config_dir, 'models', 'test_model.py'))],
            [('test_dataset', os.path.join(self.args.config_dir, 'datasets', 'test_dataset.py'))],
            [('test_summarizer', os.path.join(self.args.config_dir, 'summarizers', 'test_summarizer.py'))]
        ]
        mock_tabulate.return_value = "Mocked table output"

        # 设置参数
        self.args.models = ['test_model']
        self.args.datasets = ['test_dataset']
        self.args.summarizer = 'test_summarizer'

        # 创建ConfigManager实例
        config_manager = ConfigManager(self.args)

        # 测试输出捕获
        with mock.patch('builtins.print') as mock_print:
            config_manager.search_configs_location()

        # 验证结果
        self.assertEqual(len(config_manager.table), 4)  # 1 header + 3 entries
        mock_tabulate.assert_called_once()
        mock_print.assert_called_once_with("Mocked table output")

    @mock.patch('ais_bench.benchmark.cli.config_manager.Config.fromfile')
    @mock.patch('ais_bench.benchmark.cli.config_manager.try_fill_in_custom_cfgs')
    @mock.patch('ais_bench.benchmark.cli.config_manager.CustomConfigChecker')
    def test_get_config_from_arg_with_config_file(self, mock_checker, mock_fill_in, mock_fromfile):
        """测试从配置文件获取配置"""
        # 配置模拟返回值
        mock_config = mock.MagicMock()
        mock_fromfile.return_value = mock_config
        mock_fill_in.return_value = mock_config

        # 设置参数
        self.args.config = os.path.join(self.args.config_dir, 'test_config.py')

        # 创建ConfigManager实例并调用方法
        config_manager = ConfigManager(self.args)
        result = config_manager._get_config_from_arg()

        # 验证结果
        mock_fromfile.assert_called_once_with(self.args.config, format_python_code=False)
        mock_fill_in.assert_called_once_with(mock_config)
        mock_checker.assert_called_once_with(mock_config, self.args.config)
        mock_checker.return_value.check.assert_called_once()
        mock_config.merge_from_dict.assert_called_once()
        self.assertEqual(result, mock_config)

    @mock.patch('ais_bench.benchmark.cli.config_manager.Config.fromfile')
    def test_get_config_from_arg_with_config_file_error(self, mock_fromfile):
        """测试从配置文件获取配置时出现错误"""
        # 配置模拟抛出异常
        mock_fromfile.side_effect = Exception("Invalid syntax")

        # 设置参数
        self.args.config = os.path.join(self.args.config_dir, 'invalid_config.py')

        # 创建ConfigManager实例并调用方法
        config_manager = ConfigManager(self.args)

        # 验证异常
        with self.assertRaises(AISBenchConfigError) as cm:
            config_manager._get_config_from_arg()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.INVAILD_SYNTAX_IN_CFG_CONTENT.full_code)

    @mock.patch('ais_bench.benchmark.cli.config_manager.ConfigManager._load_models_config')
    @mock.patch('ais_bench.benchmark.cli.config_manager.ConfigManager._load_datasets_config')
    @mock.patch('ais_bench.benchmark.cli.config_manager.ConfigManager._load_summarizers_config')
    @mock.patch('ais_bench.benchmark.cli.config_manager.Config')
    def test_get_config_from_arg_with_components(self, mock_config_class, mock_load_summarizers,
                                              mock_load_datasets, mock_load_models):
        """测试从组件获取配置"""
        # 配置模拟返回值
        mock_models = [{'type': 'test_model'}]
        mock_datasets = [{'type': 'test_dataset'}]
        mock_summarizer = {'type': 'test_summarizer'}
        mock_config = mock.MagicMock()

        mock_load_models.return_value = mock_models
        mock_load_datasets.return_value = mock_datasets
        mock_load_summarizers.return_value = mock_summarizer
        mock_config_class.return_value = mock_config

        # 设置参数
        self.args.config = None

        # 创建ConfigManager实例并调用方法
        config_manager = ConfigManager(self.args)
        result = config_manager._get_config_from_arg()

        # 验证结果
        mock_load_models.assert_called_once()
        mock_load_datasets.assert_called_once()
        mock_load_summarizers.assert_called_once()
        mock_config_class.assert_called_once()
        self.assertEqual(result, mock_config)

    @mock.patch('ais_bench.benchmark.cli.config_manager.match_cfg_file')
    @mock.patch('ais_bench.benchmark.cli.config_manager.Config.fromfile')
    def test_load_models_config(self, mock_fromfile, mock_match_cfg_file):
        """测试加载模型配置"""
        # 配置模拟返回值
        mock_model_file = ('test_model', os.path.join(self.args.config_dir, 'models', 'test_model.py'))
        mock_match_cfg_file.return_value = [mock_model_file]
        mock_cfg = {'models': [{'type': 'test_model', 'abbr': 'test', 'attr': {}}]}
        mock_fromfile.return_value = mock_cfg

        # 设置参数
        self.args.models = ['test_model']

        # 创建ConfigManager实例并调用方法
        config_manager = ConfigManager(self.args)
        result = config_manager._load_models_config()

        # 验证结果
        mock_match_cfg_file.assert_called_once()
        mock_fromfile.assert_called_once_with(mock_model_file[1])
        self.assertEqual(result, mock_cfg['models'])

    def test_load_models_config_no_models(self):
        """测试未指定模型时的错误处理"""
        # 设置参数
        self.args.models = None

        # 创建ConfigManager实例并调用方法
        config_manager = ConfigManager(self.args)

        # 验证异常
        with self.assertRaises(CommandError) as cm:
            config_manager._load_models_config()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.CMD_MISS_REQUIRED_ARG.full_code)

    @mock.patch('ais_bench.benchmark.cli.config_manager.match_cfg_file')
    @mock.patch('ais_bench.benchmark.cli.config_manager.Config.fromfile')
    def test_load_models_config_missing_models_param(self, mock_fromfile, mock_match_cfg_file):
        """测试模型配置文件缺少models参数"""
        # 配置模拟返回值
        mock_model_file = ('test_model', os.path.join(self.args.config_dir, 'models', 'test_model.py'))
        mock_match_cfg_file.return_value = [mock_model_file]
        mock_cfg = {'other_param': 'value'}  # 缺少models参数
        mock_fromfile.return_value = mock_cfg

        # 设置参数
        self.args.models = ['test_model']

        # 创建ConfigManager实例并调用方法
        config_manager = ConfigManager(self.args)

        # 验证异常
        with self.assertRaises(AISBenchConfigError) as cm:
            config_manager._load_models_config()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM.full_code)

    @mock.patch('ais_bench.benchmark.cli.config_manager.match_cfg_file')
    @mock.patch('ais_bench.benchmark.cli.config_manager.make_custom_dataset_config')
    def test_load_datasets_config_custom_dataset(self, mock_make_config, mock_match_cfg_file):
        """测试加载自定义数据集配置"""
        # 设置参数
        self.args.datasets = None
        self.args.custom_dataset_path = '/path/to/custom/dataset.jsonl'
        self.args.custom_dataset_infer_method = 'test_method'
        self.args.custom_dataset_data_type = 'text'
        self.args.custom_dataset_meta_path = '/path/to/meta'

        # 模拟make_custom_dataset_config的返回值
        expected_result = {
            'path': self.args.custom_dataset_path,
            'infer_method': self.args.custom_dataset_infer_method,
            'data_type': self.args.custom_dataset_data_type,
            'meta_path': self.args.custom_dataset_meta_path
        }
        mock_make_config.return_value = expected_result

        # 创建ConfigManager实例并调用方法
        config_manager = ConfigManager(self.args)
        result = config_manager._load_datasets_config()

        # 验证结果
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['path'], self.args.custom_dataset_path)
        self.assertEqual(result[0]['infer_method'], self.args.custom_dataset_infer_method)
        self.assertEqual(result[0]['data_type'], self.args.custom_dataset_data_type)
        self.assertEqual(result[0]['meta_path'], self.args.custom_dataset_meta_path)

    def test_load_datasets_config_no_datasets_no_custom(self):
        """测试未指定数据集且未指定自定义数据集路径时的错误处理"""
        # 设置参数
        self.args.datasets = None
        self.args.custom_dataset_path = None

        # 创建ConfigManager实例并调用方法
        config_manager = ConfigManager(self.args)

        # 验证异常
        with self.assertRaises(CommandError) as cm:
            config_manager._load_datasets_config()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.CMD_MISS_REQUIRED_ARG.full_code)

    @mock.patch('ais_bench.benchmark.cli.config_manager.match_cfg_file')
    @mock.patch('ais_bench.benchmark.cli.config_manager.Config.fromfile')
    def test_load_datasets_config_with_suffix(self, mock_fromfile, mock_match_cfg_file):
        """测试加载带后缀的数据集配置"""
        # 配置模拟返回值
        mock_dataset_file = ('test_dataset', os.path.join(self.args.config_dir, 'datasets', 'test_dataset.py'))
        mock_match_cfg_file.return_value = [mock_dataset_file]
        mock_cfg = {'custom_suffix': [{'type': 'test_dataset', 'abbr': 'test', 'reader_cfg': {}, 'infer_cfg': {}, 'eval_cfg': {}}]}
        mock_fromfile.return_value = mock_cfg

        # 设置参数
        self.args.datasets = ['test_dataset/custom_suffix']

        # 创建ConfigManager实例并调用方法
        config_manager = ConfigManager(self.args)
        result = config_manager._load_datasets_config()

        # 验证结果
        self.assertEqual(result, mock_cfg['custom_suffix'])

    @mock.patch('ais_bench.benchmark.cli.config_manager.match_cfg_file')
    @mock.patch('ais_bench.benchmark.cli.config_manager.Config.fromfile')
    def test_load_datasets_config_missing_suffix_param(self, mock_fromfile, mock_match_cfg_file):
        """测试数据集配置文件缺少指定后缀参数"""
        # 配置模拟返回值
        mock_dataset_file = ('test_dataset', os.path.join(self.args.config_dir, 'datasets', 'test_dataset.py'))
        mock_match_cfg_file.return_value = [mock_dataset_file]
        mock_cfg = {'other_param': 'value'}  # 缺少_datasets后缀参数
        mock_fromfile.return_value = mock_cfg

        # 设置参数
        self.args.datasets = ['test_dataset']

        # 创建ConfigManager实例并调用方法
        config_manager = ConfigManager(self.args)

        # 验证异常
        with self.assertRaises(AISBenchConfigError) as cm:
            config_manager._load_datasets_config()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.CFG_CONTENT_MISS_REQUIRED_PARAM.full_code)

    @mock.patch('ais_bench.benchmark.cli.config_manager.match_cfg_file')
    @mock.patch('ais_bench.benchmark.cli.config_manager.Config.fromfile')
    def test_load_summarizers_config_with_key(self, mock_fromfile, mock_match_cfg_file):
        """测试加载带键的摘要器配置"""
        # 配置模拟返回值
        mock_summarizer_file = ('test_summarizer', os.path.join(self.args.config_dir, 'summarizers', 'test_summarizer.py'))
        mock_match_cfg_file.return_value = [mock_summarizer_file]
        mock_cfg = {'custom_summarizer': {'attr': {}}}
        mock_fromfile.return_value = mock_cfg

        # 设置参数
        self.args.summarizer = 'test_summarizer/custom_summarizer'

        # 创建ConfigManager实例并调用方法
        config_manager = ConfigManager(self.args)
        result = config_manager._load_summarizers_config()

        # 验证结果
        self.assertEqual(result, mock_cfg['custom_summarizer'])

    @mock.patch('ais_bench.benchmark.cli.config_manager.match_cfg_file')
    @mock.patch('ais_bench.benchmark.cli.config_manager.Config.fromfile')
    def test_load_summarizers_config_default(self, mock_fromfile, mock_match_cfg_file):
        """测试加载默认摘要器配置"""
        # 配置模拟返回值
        mock_summarizer_file = ('example', os.path.join(self.args.config_dir, 'summarizers', 'example.py'))
        mock_match_cfg_file.return_value = [mock_summarizer_file]
        mock_cfg = {'summarizer': {'attr': {}}}
        mock_fromfile.return_value = mock_cfg

        # 设置参数
        self.args.summarizer = None

        # 创建ConfigManager实例并调用方法
        config_manager = ConfigManager(self.args)
        result = config_manager._load_summarizers_config()

        # 验证结果
        self.assertEqual(result, mock_cfg['summarizer'])

    @mock.patch('os.makedirs')
    def test_update_and_init_work_dir_with_work_dir(self, mock_makedirs):
        """测试使用指定工作目录"""
        # 创建ConfigManager实例
        config_manager = ConfigManager(self.args)
        # 使用模拟对象而不是字典，因为代码中使用了属性访问
        mock_cfg = mock.MagicMock()
        mock_cfg.work_dir = self.args.work_dir
        config_manager.cfg = mock_cfg

        # 调用方法
        config_manager._update_and_init_work_dir()

        # 验证结果
        expected_work_dir = os.path.join(self.args.work_dir, self.args.dir_time_str)
        mock_cfg.__setitem__.assert_called_with('work_dir', expected_work_dir)
        # 修正断言：实际使用的是基础工作目录而不是带时间戳的目录
        mock_makedirs.assert_called_with(os.path.join(self.args.work_dir, 'configs'), exist_ok=True)

    @mock.patch('os.makedirs')
    def test_update_and_init_work_dir_default_work_dir(self, mock_makedirs):
        """测试使用默认工作目录"""
        # 设置参数
        self.args.work_dir = None

        # 创建ConfigManager实例
        config_manager = ConfigManager(self.args)
        # 使用模拟对象而不是字典
        mock_cfg = mock.MagicMock()
        # 模拟setdefault方法
        default_work_dir = 'outputs/default'
        mock_cfg.setdefault.return_value = default_work_dir
        # 确保属性访问也返回相同的值
        mock_cfg.work_dir = default_work_dir
        config_manager.cfg = mock_cfg

        # 调用方法
        config_manager._update_and_init_work_dir()

        # 验证结果
        mock_cfg.setdefault.assert_called_with('work_dir', os.path.join('outputs', 'default'))
        expected_work_dir = os.path.join(default_work_dir, self.args.dir_time_str)
        mock_cfg.__setitem__.assert_called_with('work_dir', expected_work_dir)
        # 修正断言：实际使用的是基础工作目录而不是带时间戳的目录
        mock_makedirs.assert_called_with(os.path.join(default_work_dir, 'configs'), exist_ok=True)

    @mock.patch('os.makedirs')
    @mock.patch('os.path.exists')
    @mock.patch('os.listdir')
    def test_update_and_init_work_dir_reuse_latest(self, mock_listdir, mock_exists, mock_makedirs):
        """测试重用最新实验结果"""
        # 配置模拟返回值
        mock_exists.return_value = True
        mock_listdir.return_value = ['20230101_000000', '20230102_000000']

        # 设置参数
        self.args.reuse = 'latest'

        # 创建ConfigManager实例
        config_manager = ConfigManager(self.args)
        # 使用模拟对象而不是字典
        mock_cfg = mock.MagicMock()
        mock_cfg.work_dir = self.args.work_dir
        config_manager.cfg = mock_cfg

        # 调用方法
        config_manager._update_and_init_work_dir()

        # 验证结果
        expected_work_dir = os.path.join(self.args.work_dir, '20230102_000000')
        mock_cfg.__setitem__.assert_called_with('work_dir', expected_work_dir)
        mock_exists.assert_called_with(self.args.work_dir)
        mock_listdir.assert_called_with(self.args.work_dir)
        # 修正断言：实际使用的是基础工作目录而不是带时间戳的目录
        mock_makedirs.assert_called_with(os.path.join(self.args.work_dir, 'configs'), exist_ok=True)

    @mock.patch('os.makedirs')
    @mock.patch('os.path.exists')
    def test_update_and_init_work_dir_reuse_no_results(self, mock_exists, mock_makedirs):
        """测试重用不存在的实验结果"""
        # 配置模拟返回值
        mock_exists.return_value = False

        # 设置参数
        self.args.reuse = 'latest'
        work_dir_value = self.args.work_dir

        # 创建ConfigManager实例
        config_manager = ConfigManager(self.args)
        # 使用模拟对象而不是字典
        mock_cfg = mock.MagicMock()
        mock_cfg.work_dir = work_dir_value
        config_manager.cfg = mock_cfg

        # 调用方法
        config_manager._update_and_init_work_dir()

        # 验证结果
        expected_work_dir = os.path.join(work_dir_value, self.args.dir_time_str)
        mock_cfg.__setitem__.assert_called_with('work_dir', expected_work_dir)
        mock_exists.assert_called_with(work_dir_value)
        # 修正断言：实际使用的是基础工作目录而不是带时间戳的目录
        mock_makedirs.assert_called_with(os.path.join(work_dir_value, 'configs'), exist_ok=True)

    def test_update_cfg_of_workflow(self):
        """测试更新工作流配置"""
        # 创建模拟工作流
        mock_work1 = mock.MagicMock()
        mock_work1.update_cfg.return_value = {'updated': True}
        mock_work2 = mock.MagicMock()
        mock_work2.update_cfg.return_value = {'final': 'config'}
        workflow = [mock_work1, mock_work2]

        # 创建ConfigManager实例
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {'initial': 'config'}

        # 调用方法
        config_manager._update_cfg_of_workflow(workflow)

        # 验证结果
        mock_work1.update_cfg.assert_called_once_with({'initial': 'config'})
        mock_work2.update_cfg.assert_called_once_with({'updated': True})
        self.assertEqual(config_manager.cfg, {'final': 'config'})

    @mock.patch('os.makedirs')
    @mock.patch('ais_bench.benchmark.cli.config_manager.Config.fromfile')
    def test_dump_and_reload_config(self, mock_fromfile, mock_makedirs):
        """测试转储和重新加载配置"""
        # 创建模拟配置
        mock_cfg = mock.MagicMock()
        mock_cfg.work_dir = self.args.work_dir
        mock_loaded_cfg = mock.MagicMock()
        mock_fromfile.return_value = mock_loaded_cfg

        # 创建ConfigManager实例
        config_manager = ConfigManager(self.args)
        config_manager.cfg = mock_cfg
        config_manager.cfg_time_str = self.args.dir_time_str

        # 调用方法
        config_manager._dump_and_reload_config()

        # 验证结果
        mock_cfg.dump.assert_called_once()
        mock_fromfile.assert_called_once()

    @mock.patch('os.makedirs')
    def test_dump_and_reload_config_invalid_num_prompts(self, mock_makedirs):
        """测试无效的提示数量"""
        # 设置无效的提示数量
        self.args.num_prompts = 0

        # 创建模拟配置
        mock_cfg = mock.MagicMock()
        mock_cfg.work_dir = self.args.work_dir

        # 创建ConfigManager实例
        config_manager = ConfigManager(self.args)
        config_manager.cfg = mock_cfg
        config_manager.cfg_time_str = self.args.dir_time_str

        # 验证异常
        with self.assertRaises(CommandError) as cm:
            config_manager._dump_and_reload_config()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.INVALID_ARG_VALUE_IN_CMD.full_code)

    @mock.patch('os.makedirs')
    @mock.patch('ais_bench.benchmark.cli.config_manager.Config.fromfile')
    def test_dump_and_reload_config_load_error(self, mock_fromfile, mock_makedirs):
        """测试重新加载配置时出错"""
        # 配置模拟抛出异常
        mock_fromfile.side_effect = Exception("Invalid syntax")

        # 创建模拟配置
        mock_cfg = mock.MagicMock()
        mock_cfg.work_dir = self.args.work_dir

        # 创建ConfigManager实例
        config_manager = ConfigManager(self.args)
        config_manager.cfg = mock_cfg
        config_manager.cfg_time_str = self.args.dir_time_str

        # 验证异常
        with self.assertRaises(AISBenchConfigError) as cm:
            config_manager._dump_and_reload_config()
        self.assertEqual(cm.exception.error_code_str, TMAN_CODES.INVAILD_SYNTAX_IN_CFG_CONTENT.full_code)

    @mock.patch('ais_bench.benchmark.cli.config_manager.ConfigManager._get_config_from_arg')
    @mock.patch('ais_bench.benchmark.cli.config_manager.ConfigManager._update_and_init_work_dir')
    @mock.patch('ais_bench.benchmark.cli.config_manager.ConfigManager._update_cfg_of_workflow')
    @mock.patch('ais_bench.benchmark.cli.config_manager.ConfigManager._dump_and_reload_config')
    @mock.patch('ais_bench.benchmark.cli.config_manager.ConfigManager._fill_dataset_configs')
    def test_load_config(self, mock_fill_dataset_configs, mock_dump_reload, mock_update_workflow, mock_update_work_dir, mock_get_config):
        """测试加载配置"""
        # 配置模拟返回值 - 需要包含datasets键
        mock_cfg = {
            'test': 'config',
            'datasets': [{'reader_cfg': {}, 'infer_cfg': {'retriever': {}}}],
            'models': [{'path': '/path/to/model'}],
            'cli_args': {'num_prompts': None}
        }
        mock_get_config.return_value = mock_cfg

        # 创建模拟工作流
        mock_workflow = [mock.MagicMock()]

        # 创建ConfigManager实例
        config_manager = ConfigManager(self.args)

        # 调用方法
        result = config_manager.load_config(mock_workflow)

        # 验证结果
        mock_get_config.assert_called_once()
        mock_update_work_dir.assert_called_once()
        mock_fill_dataset_configs.assert_called_once()
        mock_update_workflow.assert_called_once_with(mock_workflow)
        mock_dump_reload.assert_called_once()
        self.assertEqual(result, config_manager.cfg)

    def _service_model(self, **overrides):
        """A service model that passes the response anomaly whitelist and
        model-resource checks (chat backend + tokenizer path fallback)."""
        model = {
            'abbr': 'model',
            'attr': 'service',
            'type': VLLMCustomAPIChat,
            'generation_kwargs': {},
            'path': self.tokenizer_dir,
        }
        model.update(overrides)
        return model

    def test_response_anomaly_rejected_in_perf_mode(self):
        """响应异常检测不支持性能模式。"""
        self.args.mode = 'perf'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [{'abbr': 'model', 'attr': 'service'}],
            'datasets': [],
            'cli_args': {},
        }

        with self.assertRaises(AISBenchConfigError):
            config_manager._init_response_anomaly_config()

    def test_response_anomaly_rejected_for_agent_model(self):
        """响应异常检测不支持 Agent 模型。"""
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [{'abbr': 'agent-model', 'agent_name': 'x', 'attr': 'service'}],
            'datasets': [],
            'cli_args': {},
        }

        with self.assertRaises(AISBenchConfigError):
            config_manager._init_response_anomaly_config()

    def test_response_anomaly_config_enabled_key_is_ignored(self):
        """配置文件中的 response_anomaly.enabled 不再生效：开关仅支持命令行。"""
        self.args.mode = 'all'
        self.args.response_anomaly = False  # 命令行未传 --response-anomaly
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'response_anomaly': {'enabled': True},  # 配置文件写 enabled=True
            'models': [self._service_model()],
            'datasets': [{'abbr': 'dataset'}],
            'cli_args': {},
        }

        config_manager._init_response_anomaly_config()

        # 配置文件的 enabled 被忽略，最终以命令行为准：未启用
        self.assertFalse(config_manager.cfg['response_anomaly']['enabled'])
        # 未启用时不注入 logprobs 请求参数
        generation_kwargs = config_manager.cfg['models'][0]['generation_kwargs']
        self.assertNotIn('response_anomaly_enabled', generation_kwargs)

    def test_response_anomaly_cli_switch_enables_detection(self):
        """仅命令行 --response-anomaly 能开启检测（无 --no- 关闭形态）。"""
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [self._service_model()],
            'datasets': [{'abbr': 'dataset'}],
            'cli_args': {},
        }

        config_manager._init_response_anomaly_config()

        self.assertTrue(config_manager.cfg['response_anomaly']['enabled'])
        generation_kwargs = config_manager.cfg['models'][0]['generation_kwargs']
        self.assertTrue(generation_kwargs['response_anomaly_enabled'])

    def test_response_anomaly_injects_request_kwargs(self):
        """启用响应异常检测时为 service 模型注入 logprobs 与内部开关。"""
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [self._service_model()],
            'datasets': [{'abbr': 'dataset'}],
            'cli_args': {},
        }

        config_manager._init_response_anomaly_config()

        generation_kwargs = config_manager.cfg['models'][0]['generation_kwargs']
        self.assertTrue(generation_kwargs['logprobs'])
        self.assertEqual(generation_kwargs['top_logprobs'], 20)
        self.assertTrue(generation_kwargs['response_anomaly_enabled'])
        self.assertTrue(config_manager.cfg['response_anomaly']['enabled'])
        self.assertEqual(
            config_manager.cfg['response_anomaly']['payload_retention'],
            'anomalies',
        )
        self.assertEqual(
            config_manager.cfg['response_anomaly']['payload_storage'],
            {
                'format': 'jsonl',
                'compression': 'zstd',
                'compression_level': 3,
                'rows_per_shard': 2000,
            },
        )

    def test_response_anomaly_injects_payload_storage_into_service_models(self):
        """work_dir 初始化后，为 service 模型注入 payload 运行时配置。"""
        config_manager = ConfigManager(self.args)
        service_model = self._service_model(abbr='service-model')
        local_model = {'abbr': 'local-model', 'attr': 'local'}
        config_manager.cfg = {
            'work_dir': '/test/workdir',
            'response_anomaly': {
                'enabled': True,
                'payload_storage': {
                    'format': 'jsonl',
                    'compression': 'zstd',
                },
            },
            'models': [service_model, local_model],
        }

        config_manager._inject_response_anomaly_payload_storage()

        self.assertEqual(
            service_model['response_anomaly_payload_storage'],
            {
                'work_dir': '/test/workdir',
                'model_abbr': 'service-model',
                'format': 'jsonl',
                'compression': 'zstd',
            },
        )
        self.assertNotIn('response_anomaly_payload_storage', local_model)

    def test_response_anomaly_skips_payload_storage_when_disabled(self):
        """异常检测关闭时不向模型配置添加运行时字段。"""
        config_manager = ConfigManager(self.args)
        model_cfg = self._service_model()
        config_manager.cfg = {
            'work_dir': '/test/workdir',
            'response_anomaly': {'enabled': False},
            'models': [model_cfg],
        }

        config_manager._inject_response_anomaly_payload_storage()

        self.assertNotIn('response_anomaly_payload_storage', model_cfg)

    def test_response_anomaly_rejects_invalid_payload_retention(self):
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'response_anomaly': {'payload_retention': 'sometimes'},
            'models': [self._service_model()],
            'datasets': [],
            'cli_args': {},
        }

        with self.assertRaises(AISBenchConfigError):
            config_manager._init_response_anomaly_config()

    def test_response_anomaly_cli_payload_retention_overrides_config(self):
        self.args.mode = 'all'
        self.args.response_anomaly = True
        self.args.response_anomaly_payload_retention = 'none'
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'response_anomaly': {'payload_retention': 'all'},
            'models': [self._service_model()],
            'datasets': [],
            'cli_args': {},
        }

        config_manager._init_response_anomaly_config()

        self.assertEqual(
            config_manager.cfg['response_anomaly']['payload_retention'],
            'none',
        )

    def test_response_anomaly_rejects_invalid_payload_storage(self):
        self.args.mode = 'all'
        self.args.response_anomaly = True
        for payload_storage in (
            {'format': 'parquet'},
            {'compression': 'gzip'},
            {'compression_level': 0},
            {'compression_level': True},
            {'rows_per_shard': 0},
            {'rows_per_shard': True},
        ):
            config_manager = ConfigManager(self.args)
            config_manager.cfg = {
                'response_anomaly': {
                    'payload_storage': payload_storage,
                },
                'models': [self._service_model()],
                'datasets': [],
                'cli_args': {},
            }

            with self.subTest(payload_storage=payload_storage):
                with self.assertRaises(AISBenchConfigError):
                    config_manager._init_response_anomaly_config()

    def test_response_anomaly_overrides_explicit_logprobs_config(self):
        """启用异常检测时强制覆盖模型里显式的 logprobs/top_logprobs。"""
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [
                self._service_model(
                    generation_kwargs={
                        'logprobs': False,
                        'top_logprobs': 5,
                    },
                )
            ],
            'datasets': [{'abbr': 'dataset'}],
            'cli_args': {},
        }

        config_manager._init_response_anomaly_config()

        generation_kwargs = config_manager.cfg['models'][0]['generation_kwargs']
        self.assertIs(generation_kwargs['logprobs'], True)
        self.assertEqual(generation_kwargs['top_logprobs'], 20)
        self.assertTrue(generation_kwargs['response_anomaly_enabled'])

    def test_response_anomaly_rejects_configurable_top_logprobs(self):
        """异常检测使用固定 top_logprobs，不允许外部修改。"""
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'response_anomaly': {'top_logprobs': 30},
            'models': [self._service_model()],
            'datasets': [{'abbr': 'dataset'}],
            'cli_args': {},
        }

        with self.assertRaises(AISBenchConfigError):
            config_manager._init_response_anomaly_config()

    def test_response_anomaly_merges_model_level_config(self):
        """模型级 response_anomaly 配置覆盖全局配置。"""
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [
                self._service_model(
                    response_anomaly={
                        'model_name': 'Custom-Name',
                        'model_path': self.tokenizer_dir,
                        'top_logprobs': 20,
                    },
                )
            ],
            'datasets': [{'abbr': 'dataset'}],
            'cli_args': {},
        }

        config_manager._init_response_anomaly_config()

        model_anomaly_cfg = config_manager.cfg['models'][0]['response_anomaly']
        self.assertEqual(model_anomaly_cfg['model_name'], 'Custom-Name')
        self.assertEqual(model_anomaly_cfg['model_path'], self.tokenizer_dir)
        self.assertNotIn('top_logprobs', model_anomaly_cfg)
        self.assertEqual(
            config_manager.cfg['models'][0]['generation_kwargs']['top_logprobs'],
            20,
        )

    def test_response_anomaly_rejected_in_eval_viz_judge_modes(self):
        """单开 eval/viz/judge 模式不支持响应异常检测。"""
        for mode in ('eval', 'viz', 'judge'):
            self.args.mode = mode
            self.args.response_anomaly = True
            config_manager = ConfigManager(self.args)
            config_manager.cfg = {
                'models': [self._service_model()],
                'datasets': [],
                'cli_args': {},
            }
            with self.subTest(mode=mode):
                with self.assertRaises(AISBenchConfigError) as cm:
                    config_manager._init_response_anomaly_config()
                self.assertIn('not supported in mode', str(cm.exception))

    def test_response_anomaly_rejected_for_unsupported_model_class(self):
        """completions 后端（VLLMCustomAPI）无法返回 token id，应被拦截。"""
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [
                self._service_model(type=VLLMCustomAPI),
            ],
            'datasets': [],
            'cli_args': {},
        }

        with self.assertRaises(AISBenchConfigError) as cm:
            config_manager._init_response_anomaly_config()
        self.assertIn('VLLMCustomAPIChat', str(cm.exception))

    def test_response_anomaly_rejected_for_local_models(self):
        """本地模型不在检测范围，白名单检查应跳过（由 service 校验兜底）。"""
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [
                {
                    'abbr': 'local-model',
                    'attr': 'local',
                    'path': self.tokenizer_dir,
                }
            ],
            'datasets': [],
            'cli_args': {},
        }

        # 无 service 模型时报错（白名单不应先对 local 模型误报）。
        with self.assertRaises(AISBenchConfigError) as cm:
            config_manager._init_response_anomaly_config()
        self.assertIn('no service model', str(cm.exception))

    def test_response_anomaly_model_path_falls_back_to_model_path_field(self):
        """response_anomaly.model_path 未配置时回退模型 path 字段（tokenizer 目录）。"""
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [self._service_model()],
            'datasets': [],
            'cli_args': {},
        }

        config_manager._init_response_anomaly_config()

        model_anomaly_cfg = config_manager.cfg['models'][0]['response_anomaly']
        self.assertEqual(model_anomaly_cfg['model_path'], self.tokenizer_dir)

    def test_response_anomaly_rejects_invalid_model_path_field(self):
        """模型 path 指向不存在目录时应报错并说明原因。"""
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [self._service_model(path='/nonexistent/tokenizer-dir')],
            'datasets': [],
            'cli_args': {},
        }

        with self.assertRaises(AISBenchConfigError) as cm:
            config_manager._init_response_anomaly_config()
        self.assertIn('non-existent directory', str(cm.exception))

    def test_response_anomaly_rejects_missing_model_resources(self):
        """无 model_path 也无 msprobe 三件套时直接报错，不回退 msProbe 默认。"""
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [self._service_model(path='')],
            'datasets': [],
            'cli_args': {},
        }

        with self.assertRaises(AISBenchConfigError) as cm:
            config_manager._init_response_anomaly_config()
        message = str(cm.exception)
        self.assertIn('msprobe_mtype_path', message)
        self.assertIn('msprobe_token2category_dir', message)
        self.assertIn('ais_bench-gen-response-anomaly-config', message)

    def test_response_anomaly_accepts_explicit_msprobe_paths(self):
        """显式配置 mtype + token2category（真实存在）+ model_name 时无需 model_path。"""
        import os as _os

        mtype_path = _os.path.join(self.tokenizer_dir, 'mtype_config.json')
        tk2cat_dir = _os.path.join(self.tokenizer_dir, 'token2category')
        open(mtype_path, 'w').close()
        _os.makedirs(tk2cat_dir, exist_ok=True)

        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [
                self._service_model(
                    path='',
                    response_anomaly={
                        'model_name': 'Qwen3-30B-A3B',
                        'msprobe_mtype_path': mtype_path,
                        'msprobe_token2category_dir': tk2cat_dir,
                    },
                )
            ],
            'datasets': [],
            'cli_args': {},
        }

        config_manager._init_response_anomaly_config()

        model_anomaly_cfg = config_manager.cfg['models'][0]['response_anomaly']
        self.assertEqual(model_anomaly_cfg['msprobe_mtype_path'], mtype_path)
        self.assertEqual(model_anomaly_cfg['model_name'], 'Qwen3-30B-A3B')
        self.assertIsNone(model_anomaly_cfg['model_path'])

    def test_response_anomaly_model_name_falls_back_to_path_basename(self):
        """未配置 model_name 时回退 model_path 目录名，而不是模型 abbr。"""
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [
                self._service_model(
                    abbr='vllm-api-general-chat',
                    response_anomaly={},
                )
            ],
            'datasets': [],
            'cli_args': {},
        }

        config_manager._init_response_anomaly_config()

        model_anomaly_cfg = config_manager.cfg['models'][0]['response_anomaly']
        self.assertEqual(
            model_anomaly_cfg['model_name'],
            os.path.basename(os.path.normpath(self.tokenizer_dir)),
        )
        self.assertNotEqual(model_anomaly_cfg['model_name'], 'vllm-api-general-chat')

    def test_response_anomaly_model_name_falls_back_to_global_model_path(self):
        """顶层全局块 model_path（未填 model_name）也参与 model_name 推导。"""
        import os as _os

        global_model_dir = _os.path.join(self.tokenizer_dir, 'Qwen3-30B-A3B')
        _os.makedirs(global_model_dir, exist_ok=True)

        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'response_anomaly': {'model_path': global_model_dir},
            'models': [
                self._service_model(path='', response_anomaly={}),
            ],
            'datasets': [],
            'cli_args': {},
        }

        config_manager._init_response_anomaly_config()

        model_anomaly_cfg = config_manager.cfg['models'][0]['response_anomaly']
        self.assertEqual(model_anomaly_cfg['model_name'], 'Qwen3-30B-A3B')

    def test_response_anomaly_model_level_model_path_beats_global(self):
        """模型级 model_path 优先于全局块 model_path 推导 model_name。"""
        import os as _os

        global_dir = _os.path.join(self.tokenizer_dir, 'GlobalModel')
        model_dir = _os.path.join(self.tokenizer_dir, 'ModelLevelModel')
        _os.makedirs(global_dir, exist_ok=True)
        _os.makedirs(model_dir, exist_ok=True)

        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'response_anomaly': {'model_path': global_dir},
            'models': [
                self._service_model(
                    path='',
                    response_anomaly={'model_path': model_dir},
                ),
            ],
            'datasets': [],
            'cli_args': {},
        }

        config_manager._init_response_anomaly_config()

        model_anomaly_cfg = config_manager.cfg['models'][0]['response_anomaly']
        self.assertEqual(model_anomaly_cfg['model_name'], 'ModelLevelModel')

    def test_response_anomaly_explicit_paths_require_model_name(self):
        """显式 msprobe 路径但缺 model_name 时应启动报错，而非静默用 abbr 匹配失败。"""
        import os as _os

        mtype_path = _os.path.join(self.tokenizer_dir, 'mtype_config.json')
        tk2cat_dir = _os.path.join(self.tokenizer_dir, 'token2category')
        open(mtype_path, 'w').close()
        _os.makedirs(tk2cat_dir, exist_ok=True)

        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [
                self._service_model(
                    path='',
                    response_anomaly={
                        'msprobe_mtype_path': mtype_path,
                        'msprobe_token2category_dir': tk2cat_dir,
                    },
                )
            ],
            'datasets': [],
            'cli_args': {},
        }

        with self.assertRaises(AISBenchConfigError) as cm:
            config_manager._init_response_anomaly_config()
        self.assertIn('model_name', str(cm.exception))

    def test_response_anomaly_rejects_nonexistent_msprobe_paths(self):
        """显式配置的 msProbe 路径不存在时启动即报错，而不是运行期全 failed。"""
        self.args.mode = 'all'
        self.args.response_anomaly = True
        config_manager = ConfigManager(self.args)
        config_manager.cfg = {
            'models': [
                self._service_model(
                    path='',
                    response_anomaly={
                        'model_name': 'Qwen3-30B-A3B',
                        'msprobe_mtype_path': '/workspace/missing/mtype_config.json',
                        'msprobe_token2category_dir': '/workspace/missing/token2category',
                    },
                )
            ],
            'datasets': [],
            'cli_args': {},
        }

        with self.assertRaises(AISBenchConfigError) as cm:
            config_manager._init_response_anomaly_config()
        message = str(cm.exception)
        self.assertIn('do not exist', message)
        self.assertIn('/workspace/missing/mtype_config.json', message)
        self.assertIn('/workspace/missing/token2category', message)

if __name__ == '__main__':
    unittest.main()
