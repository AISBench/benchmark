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

        # 验证 save_to 参数是否设置
        save_to = self.cfg["datasets"][0][0]["args"].get("save_to")
        self.assertIsNotNone(save_to)
        self.assertTrue(save_to.endswith("tau2_run_detail"))

    def test_refresh_cfg(self):
        """测试刷新配置方法"""
        task = TAU2BenchTask(self.cfg)
        task._refresh_cfg()

        # 验证模型参数是否复制到数据集参数
        dataset_args = self.cfg["datasets"][0][0]["args"]
        self.assertEqual(dataset_args.get("abbr"), "gpt-3.5-turbo")
        self.assertEqual(dataset_args.get("api_key"), "test_api_key")
        # 验证 type 参数是否未复制
        self.assertNotIn("type", dataset_args)

    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.RunConfig')
    def test_construct_run_cfg(self, mock_run_config):
        """测试构建运行配置方法"""
        task = TAU2BenchTask(self.cfg)
        task._refresh_cfg()
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

    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.compute_metrics')
    def test_dump_eval_results(self, mock_compute_metrics):
        """测试导出评估结果方法"""
        # 模拟 compute_metrics 返回值
        mock_metrics = mock.MagicMock()
        mock_metrics.avg_reward = 0.8
        mock_compute_metrics.return_value = mock_metrics

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
        self.assertEqual(results.get("total_count"), 0)  # 因为 get_tasks 被 mock 了

    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.run_domain')
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.compute_metrics')
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.get_tasks')
    def test_run(self, mock_get_tasks, mock_compute_metrics, mock_run_domain):
        """测试 run 方法"""
        # 模拟依赖
        mock_get_tasks.return_value = [1, 2, 3]
        mock_run_domain.return_value = {}
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
