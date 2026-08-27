import unittest
from unittest.mock import patch, MagicMock
import sys
from ais_bench.benchmark.cli.argument_parser import ArgumentParser


class TestArgumentParser(unittest.TestCase):
    def setUp(self):
        # 保存原始的sys.argv
        self.original_argv = sys.argv.copy()

    def tearDown(self):
        # 恢复原始的sys.argv
        sys.argv = self.original_argv.copy()

    @patch('ais_bench.benchmark.cli.argument_parser.get_current_time_str')
    def test_parse_args_default(self, mock_get_current_time_str):
        """测试默认参数解析"""
        # 模拟返回值
        mock_get_current_time_str.return_value = "20230516_144254"

        # 设置命令行参数
        sys.argv = ['benchmark.py']

        # 创建解析器并解析参数
        parser = ArgumentParser()
        args = parser.parse_args()

        # 验证结果
        self.assertEqual(args.cfg_time_str, "20230516_144254")
        self.assertEqual(args.dir_time_str, "20230516_144254")
        self.assertFalse(args.debug)
        self.assertFalse(args.dry_run)
        self.assertEqual(args.mode, 'all')
        self.assertEqual(args.config_dir, 'configs')
        self.assertEqual(args.max_num_workers, 1)
        self.assertEqual(args.max_workers_per_gpu, 1)
        self.assertEqual(args.num_warmups, 1)
        self.assertIsNone(args.num_prompts)
        self.assertIsNone(args.response_anomaly_payload_retention)

    @patch('ais_bench.benchmark.cli.argument_parser.get_current_time_str')
    def test_parse_args_response_anomaly_payload_retention(
        self, mock_get_current_time_str
    ):
        mock_get_current_time_str.return_value = "20230516_144254"
        for retention in ('all', 'anomalies', 'none'):
            sys.argv = [
                'benchmark.py',
                '--response-anomaly-payload-retention',
                retention,
            ]

            with self.subTest(retention=retention):
                args = ArgumentParser().parse_args()
                self.assertEqual(
                    args.response_anomaly_payload_retention, retention
                )

    @patch('ais_bench.benchmark.cli.argument_parser.get_current_time_str')
    def test_parse_args_with_config(self, mock_get_current_time_str):
        """测试带有配置文件路径的参数解析"""
        # 模拟返回值
        mock_get_current_time_str.return_value = "20230516_144254"

        # 设置命令行参数
        sys.argv = ['benchmark.py', 'configs/test_config.py']

        # 创建解析器并解析参数
        parser = ArgumentParser()
        args = parser.parse_args()

        # 验证结果
        self.assertEqual(args.config, 'configs/test_config.py')

    @patch('ais_bench.benchmark.cli.argument_parser.get_current_time_str')
    def test_parse_args_dry_run_sets_debug(self, mock_get_current_time_str):
        """测试dry_run模式会设置debug为True"""
        # 模拟返回值
        mock_get_current_time_str.return_value = "20230516_144254"

        # 设置命令行参数
        sys.argv = ['benchmark.py', '--dry-run']

        # 创建解析器并解析参数
        parser = ArgumentParser()
        args = parser.parse_args()

        # 验证结果
        self.assertTrue(args.dry_run)
        self.assertTrue(args.debug)  # dry_run应该设置debug为True

    @patch('ais_bench.benchmark.cli.argument_parser.get_current_time_str')
    def test_parse_args_base_options(self, mock_get_current_time_str):
        """测试基础选项参数解析"""
        # 模拟返回值
        mock_get_current_time_str.return_value = "20230516_144254"

        # 设置命令行参数
        sys.argv = ['benchmark.py', '--debug', '--search', '--mode', 'perf',
                    '--models', 'model1', 'model2', '--datasets', 'dataset1', 'dataset2',
                    '--summarizer', 'summarizer1', '--work-dir', '/path/to/workdir',
                    '--config-dir', 'custom_configs', '--max-num-workers', '4',
                    '--max-workers-per-gpu', '2', '--num-prompts', '10', '--num-warmups', '3']

        # 创建解析器并解析参数
        parser = ArgumentParser()
        args = parser.parse_args()

        # 验证结果
        self.assertTrue(args.debug)
        self.assertTrue(args.search)
        self.assertEqual(args.mode, 'perf')
        self.assertEqual(args.models, ['model1', 'model2'])
        self.assertEqual(args.datasets, ['dataset1', 'dataset2'])
        self.assertEqual(args.summarizer, 'summarizer1')
        self.assertEqual(args.work_dir, '/path/to/workdir')
        self.assertEqual(args.config_dir, 'custom_configs')
        self.assertEqual(args.max_num_workers, 4)
        self.assertEqual(args.max_workers_per_gpu, 2)
        self.assertEqual(args.num_prompts, 10)
        self.assertEqual(args.num_warmups, 3)

    @patch('ais_bench.benchmark.cli.argument_parser.get_current_time_str')
    def test_parse_args_reuse_option(self, mock_get_current_time_str):
        """测试reuse选项参数解析"""
        # 模拟返回值
        mock_get_current_time_str.return_value = "20230516_144254"

        # 测试reuse不带参数
        sys.argv = ['benchmark.py', '--reuse']
        parser = ArgumentParser()
        args = parser.parse_args()
        self.assertEqual(args.reuse, 'latest')

        # 测试reuse带参数
        sys.argv = ['benchmark.py', '--reuse', '20230516_144254']
        parser = ArgumentParser()
        args = parser.parse_args()
        self.assertEqual(args.reuse, '20230516_144254')

    @patch('ais_bench.benchmark.cli.argument_parser.get_current_time_str')
    def test_parse_args_accuracy_options(self, mock_get_current_time_str):
        """测试精度评估相关选项参数解析"""
        # 模拟返回值
        mock_get_current_time_str.return_value = "20230516_144254"

        # 设置命令行参数
        sys.argv = ['benchmark.py', '--merge-ds', '--dump-eval-details', '--dump-extract-rate']

        # 创建解析器并解析参数
        parser = ArgumentParser()
        args = parser.parse_args()

        # 验证结果
        self.assertTrue(args.merge_ds)
        self.assertTrue(args.dump_eval_details)
        self.assertTrue(args.dump_extract_rate)

    @patch('ais_bench.benchmark.cli.argument_parser.get_current_time_str')
    def test_parse_args_perf_options(self, mock_get_current_time_str):
        """测试性能评估相关选项参数解析"""
        # 模拟返回值
        mock_get_current_time_str.return_value = "20230516_144254"

        # 测试pressure选项
        sys.argv = ['benchmark.py', '--pressure']
        parser = ArgumentParser()
        args = parser.parse_args()
        self.assertTrue(args.pressure)
        self.assertEqual(args.pressure_time, 15)  # 默认值

        # 测试pressure和pressure-time选项
        sys.argv = ['benchmark.py', '--pressure', '--pressure-time', '30']
        parser = ArgumentParser()
        args = parser.parse_args()
        self.assertTrue(args.pressure)
        self.assertEqual(args.pressure_time, 30)

    @patch('ais_bench.benchmark.cli.argument_parser.get_current_time_str')
    def test_parse_args_custom_dataset_options(self, mock_get_current_time_str):
        """测试自定义数据集相关选项参数解析"""
        # 模拟返回值
        mock_get_current_time_str.return_value = "20230516_144254"

        # 设置命令行参数
        sys.argv = ['benchmark.py', '--custom-dataset-path', '/path/to/dataset',
                    '--custom-dataset-meta-path', '/path/to/meta',
                    '--custom-dataset-data-type', 'mcq',
                    '--custom-dataset-infer-method', 'gen']

        # 创建解析器并解析参数
        parser = ArgumentParser()
        args = parser.parse_args()

        # 验证结果
        self.assertEqual(args.custom_dataset_path, '/path/to/dataset')
        self.assertEqual(args.custom_dataset_meta_path, '/path/to/meta')
        self.assertEqual(args.custom_dataset_data_type, 'mcq')
        self.assertEqual(args.custom_dataset_infer_method, 'gen')

    @patch('ais_bench.benchmark.cli.argument_parser.get_current_time_str')
    def test_parse_args_mcq_and_gen_options(self, mock_get_current_time_str):
        """测试mcq数据类型和gen推理方法选项"""
        # 模拟返回值
        mock_get_current_time_str.return_value = "20230516_144254"

        # 设置命令行参数
        sys.argv = ['benchmark.py', '--custom-dataset-data-type', 'qa',
                    '--custom-dataset-infer-method', 'gen']

        # 创建解析器并解析参数
        parser = ArgumentParser()
        args = parser.parse_args()

        # 验证结果
        self.assertEqual(args.custom_dataset_data_type, 'qa')
        self.assertEqual(args.custom_dataset_infer_method, 'gen')

    @patch('ais_bench.benchmark.cli.argument_parser.get_current_time_str')
    def test_parse_args_api_model_options(self, mock_get_current_time_str):
        """测试API模型通用覆盖选项参数解析"""
        mock_get_current_time_str.return_value = "20230516_144254"

        sys.argv = [
            'benchmark.py',
            '--path', '/tmp/model',
            '--model-name', 'Qwen',
            '--request-rate', '10',
            '--retry', '3',
            '--api-key', 'sk-test',
            '--host-ip', '127.0.0.1',
            '--host-port', '8000',
            '--url', 'http://example.com/v1',
            '--max-out-len', '256',
            '--batch-size', '4',
            '--trust-remote-code',
            '--generation-kwargs', '{"temperature": 0.5, "ignore_eos": false}',
        ]

        parser = ArgumentParser()
        args = parser.parse_args()

        self.assertEqual(args.path, '/tmp/model')
        self.assertEqual(args.model_name, 'Qwen')
        self.assertEqual(args.request_rate, 10.0)
        self.assertEqual(args.retry, 3)
        self.assertEqual(args.api_key, 'sk-test')
        self.assertEqual(args.host_ip, '127.0.0.1')
        self.assertEqual(args.host_port, 8000)
        self.assertEqual(args.url, 'http://example.com/v1')
        self.assertEqual(args.max_out_len, 256)
        self.assertEqual(args.batch_size, 4)
        self.assertTrue(args.trust_remote_code)
        self.assertEqual(args.generation_kwargs,
                         {'temperature': 0.5, 'ignore_eos': False})

    @patch('ais_bench.benchmark.cli.argument_parser.get_current_time_str')
    def test_parse_args_api_model_options_default_none(self, mock_get_current_time_str):
        """未显式指定时 api_model 覆盖参数默认均为 None"""
        mock_get_current_time_str.return_value = "20230516_144254"

        sys.argv = ['benchmark.py']
        args = ArgumentParser().parse_args()

        for attr in ('path', 'model_name', 'request_rate', 'retry', 'api_key',
                     'host_ip', 'host_port', 'url', 'max_out_len', 'batch_size',
                     'trust_remote_code', 'generation_kwargs'):
            self.assertIsNone(getattr(args, attr))

    @patch('ais_bench.benchmark.cli.argument_parser.get_current_time_str')
    def test_parse_args_api_model_no_trust_remote_code(self, mock_get_current_time_str):
        """测试 --no-trust-remote-code 关闭信任远程代码"""
        mock_get_current_time_str.return_value = "20230516_144254"

        sys.argv = ['benchmark.py', '--no-trust-remote-code']
        args = ArgumentParser().parse_args()
        self.assertFalse(args.trust_remote_code)

    def test_parse_args_api_model_invalid_generation_kwargs_json(self):
        """非法 JSON 的 --generation-kwargs 应导致解析失败"""
        sys.argv = ['benchmark.py', '--generation-kwargs', '{bad json']
        with self.assertRaises(SystemExit):
            ArgumentParser().parse_args()

    def test_init_method_creates_parser(self):
        """测试初始化方法创建了解析器并添加了所有参数组"""
        parser = ArgumentParser()

        # 验证parser是否存在
        self.assertIsNotNone(parser.parser)

        # 验证参数组是否包含我们期望的参数
        # 由于argparse的内部结构，我们通过检查是否存在某些参数来验证
        with patch('sys.argv', ['benchmark.py']):
            args = parser.parse_args()
            # 检查各个组中的代表性参数
            self.assertTrue(hasattr(args, 'debug'))  # base_args
            self.assertTrue(hasattr(args, 'merge_ds'))  # accuracy_args
            self.assertTrue(hasattr(args, 'pressure'))  # perf_args
            self.assertTrue(hasattr(args, 'custom_dataset_path'))  # custom_dataset_args
            self.assertTrue(hasattr(args, 'model_name'))  # api_model_args
            self.assertTrue(hasattr(args, 'host_port'))  # api_model_args
            self.assertTrue(hasattr(args, 'generation_kwargs'))  # api_model_args
            self.assertTrue(hasattr(args, 'trust_remote_code'))  # api_model_args


if __name__ == '__main__':
    unittest.main()
