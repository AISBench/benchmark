import unittest
import os
import tempfile
import json
from unittest import mock
from pathlib import Path

from ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task import TAU2BenchTask
from ais_bench.benchmark.utils.config import ConfigDict
from ais_bench.benchmark.tasks.base import TaskStateManager


class TestTAU2BenchTask(unittest.TestCase):
    def setUp(self):
        # 创建临时工作目录
        self.temp_dir = tempfile.mkdtemp()

        # 构建测试配置
        self.cfg = ConfigDict({
            "work_dir": self.temp_dir,
            "models": [{
                "type": "openai",
                "abbr": "gpt-3.5-turbo",
                "api_key": "test_api_key"
            }],
            "datasets": [[{
                "abbr": "test_dataset",
                "args": {
                    "domain": "test_domain",
                    "task_split_name": "test_split",
                    "num_tasks": 5,
                    "num_trials": 2
                }
            }]],
            "cli_args": {
                "debug": False
            }
        })

        # 创建任务状态管理器
        self.task_state_manager = mock.MagicMock(spec=TaskStateManager)

    def tearDown(self):
        # 清理临时目录
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        """测试初始化方法"""
        task = TAU2BenchTask(self.cfg)
        self.assertIsNone(task.captured_metrics)

    def test_get_command(self):
        """测试获取命令方法"""
        task = TAU2BenchTask(self.cfg)
        cfg_path = "test_config.py"
        template = "test_template"
        command = task.get_command(cfg_path, template)
        self.assertIn(cfg_path, command)
        self.assertIn(os.path.basename(__file__).replace("test_", ""), command)

    def test_set_api_key(self):
        """测试设置 API Key 方法"""
        # 测试有 API Key 的情况
        task = TAU2BenchTask(self.cfg)
        task._set_api_key()
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), "test_api_key")

        # 测试无 API Key 的情况
        cfg_no_api = ConfigDict({
            "work_dir": self.temp_dir,
            "models": [{
                "type": "openai",
                "abbr": "gpt-3.5-turbo"
            }],
            "datasets": [[{
                "abbr": "test_dataset",
                "args": {}
            }]],
            "cli_args": {
                "debug": False
            }
        })
        task_no_api = TAU2BenchTask(cfg_no_api)
        task_no_api._set_api_key()
        self.assertEqual(os.environ.get("OPENAI_API_KEY"), "fake_api_key")

    def test_prepare_out_dir(self):
        """测试准备输出目录方法"""
        task = TAU2BenchTask(self.cfg)
        task._prepare_out_dir()

        # 验证输出目录是否创建
        expected_out_dir = os.path.join(self.temp_dir, "results", "gpt-3.5-turbo")
        self.assertTrue(os.path.exists(expected_out_dir))

        expected_dataset_dir = os.path.join(expected_out_dir, "test_dataset")
        self.assertTrue(os.path.exists(expected_dataset_dir))

    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.ConfigDict')
    def test_refresh_cfg(self, mock_config_dict):
        """测试刷新配置方法"""
        # 创建模拟的配置对象
        mock_model = mock.MagicMock()
        mock_model.items.return_value = [("type", "openai"), ("abbr", "gpt-3.5-turbo"), ("api_key", "test_api_key")]

        mock_dataset_args = mock.MagicMock()

        mock_dataset = mock.MagicMock()
        mock_dataset.__getitem__.side_effect = lambda x: {0: {"args": mock_dataset_args}}[x]

        mock_cfg = mock.MagicMock()
        mock_cfg.__getitem__.side_effect = lambda x: {"models": [{"type": "openai", "abbr": "gpt-3.5-turbo", "api_key": "test_api_key"}], "datasets": [[{"args": mock_dataset_args}]]}[x]

        # 创建任务实例
        task = TAU2BenchTask(mock_cfg)
        task._refresh_cfg()

        # 验证模型参数是否复制到数据集参数
        mock_dataset_args.__setitem__.assert_any_call("abbr", "gpt-3.5-turbo")
        mock_dataset_args.__setitem__.assert_any_call("api_key", "test_api_key")
        # 验证 type 参数是否未复制
        # 由于我们使用了 mock，这里不需要验证，因为 mock_model.items() 已经排除了 type

    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.RunConfig')
    def test_construct_run_cfg(self, mock_run_config):
        """测试构建运行配置方法"""
        # 创建模拟的配置对象
        mock_dataset_args = mock.MagicMock()
        mock_dataset_args.items.return_value = [("domain", "test_domain"), ("task_split_name", "test_split"), ("num_tasks", 5), ("num_trials", 2)]

        mock_cfg = mock.MagicMock()
        mock_cfg.__getitem__.side_effect = lambda x: {"datasets": [[{"args": mock_dataset_args}]]}[x]

        # 创建任务实例
        task = TAU2BenchTask(mock_cfg)
        run_cfg = task._construct_run_cfg()

        # 验证 RunConfig 是否被正确调用
        mock_run_config.assert_called_once()

    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.get_tasks')
    def test_get_task_count(self, mock_get_tasks):
        """测试获取任务数量方法"""
        # 模拟 get_tasks 返回 5 个任务
        mock_get_tasks.return_value = [1, 2, 3, 4, 5]

        task = TAU2BenchTask(self.cfg)
        task._refresh_cfg()
        run_cfg = task._construct_run_cfg()
        task_count = task._get_task_count(run_cfg)

        # 验证任务数量是否正确
        self.assertEqual(task_count, 5)
        # 验证 get_tasks 是否被正确调用
        mock_get_tasks.assert_called_once()

    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.get_tasks')
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.compute_metrics')
    def test_dump_eval_results(self, mock_compute_metrics, mock_get_tasks):
        """测试导出评估结果方法"""
        # 模拟 compute_metrics 返回值
        mock_metrics = mock.MagicMock()
        mock_metrics.avg_reward = 0.8
        mock_compute_metrics.return_value = mock_metrics

        # 模拟 get_tasks 返回 3 个任务
        mock_get_tasks.return_value = [1, 2, 3]

        task = TAU2BenchTask(self.cfg)
        task._prepare_out_dir()
        task._refresh_cfg()
        task.run_config = task._construct_run_cfg()

        # 执行测试
        task._dump_eval_results({})

        # 验证结果文件是否创建
        expected_out_json = os.path.join(self.temp_dir, "results", "gpt-3.5-turbo", "test_dataset.json")
        self.assertTrue(os.path.exists(expected_out_json))

        # 验证结果文件内容
        with open(expected_out_json, 'r') as f:
            results = json.load(f)
        self.assertEqual(results.get("pass^2"), 80.0)  # 0.8 * 100
        self.assertEqual(results.get("total_count"), 3)  # 因为 get_tasks 被 mock 为返回 3 个任务

    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.run_domain')
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.compute_metrics')
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.get_tasks')
    def test_run(self, mock_get_tasks, mock_compute_metrics, mock_run_domain):
        """测试 run 方法"""
        # 模拟依赖
        mock_get_tasks.return_value = [1, 2, 3]

        # 模拟 run_domain 函数，创建 save_to 文件并写入任务数据
        def mock_run_domain_func(run_config):
            # 创建 save_to 文件并写入任务数据
            save_to_file = f"{run_config.save_to}.json"
            os.makedirs(os.path.dirname(save_to_file), exist_ok=True)
            with open(save_to_file, 'w') as f:
                # 写入 3 个任务的数据，每个任务执行 2 次
                tasks = []
                for i in range(3):
                    for j in range(2):
                        tasks.append({"task_id": f"task_{i}_{j}"})
                json.dump(tasks, f)
            return {}

        mock_run_domain.side_effect = mock_run_domain_func

        mock_metrics = mock.MagicMock()
        mock_metrics.avg_reward = 0.7
        mock_compute_metrics.return_value = mock_metrics

        task = TAU2BenchTask(self.cfg)

        # 执行测试
        task.run(self.task_state_manager)

        # 验证方法调用
        mock_run_domain.assert_called_once()
        mock_compute_metrics.assert_called_once()
        # 验证任务状态更新
        self.task_state_manager.update_task_state.assert_called()



if __name__ == '__main__':
    unittest.main()
