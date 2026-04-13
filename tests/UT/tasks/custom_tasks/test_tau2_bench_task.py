import unittest
import os
import tempfile
import json
from unittest import mock
from pathlib import Path

# Mock litellm and tau2 modules before importing TAU2BenchTask
import sys
from unittest import mock

# Mock litellm module
sys.modules['litellm'] = mock.MagicMock()
sys.modules['litellm.utils'] = mock.MagicMock()
sys.modules['litellm.cost_calculator'] = mock.MagicMock()
sys.modules['litellm.utils'].get_response_cost = mock.MagicMock()
sys.modules['litellm.cost_calculator'].get_response_cost = mock.MagicMock()

# Mock tau2 module
sys.modules['tau2'] = mock.MagicMock()
sys.modules['tau2.data_model'] = mock.MagicMock()
sys.modules['tau2.data_model.simulation'] = mock.MagicMock()
sys.modules['tau2.data_model.simulation'].RunConfig = mock.MagicMock()
sys.modules['tau2.run'] = mock.MagicMock()
sys.modules['tau2.run'].run_domain = mock.MagicMock()
sys.modules['tau2.run'].get_tasks = mock.MagicMock()
sys.modules['tau2.metrics'] = mock.MagicMock()
sys.modules['tau2.metrics.agent_metrics'] = mock.MagicMock()
sys.modules['tau2.metrics.agent_metrics'].compute_metrics = mock.MagicMock()
sys.modules['tau2.utils'] = mock.MagicMock()
sys.modules['tau2.utils.llm_utils'] = mock.MagicMock()
sys.modules['tau2.utils.llm_utils'].get_response_cost = mock.MagicMock()
sys.modules['tau2.utils.llm_utils'].logger = mock.MagicMock()
sys.modules['tau2.utils.llm_utils'].logger.error = mock.MagicMock()
sys.modules['tau2.utils.display'] = mock.MagicMock()
sys.modules['tau2.utils.display'].ConsoleDisplay = mock.MagicMock()
sys.modules['tau2.utils.display'].ConsoleDisplay.console = mock.MagicMock()
sys.modules['tau2.utils.display'].ConsoleDisplay.console.input = mock.MagicMock(return_value="y")

# Now import TAU2BenchTask
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

    def test_prepare_out_dir_existing_file(self):
        """测试准备输出目录方法 - 文件已存在的情况"""
        # 创建文件
        expected_out_dir = os.path.join(self.temp_dir, "results", "gpt-3.5-turbo", "test_dataset")
        os.makedirs(expected_out_dir, exist_ok=True)
        existing_file = os.path.join(expected_out_dir, "tau2_run_detail")
        with open(existing_file, 'w') as f:
            f.write("test content")

        # 执行测试
        task = TAU2BenchTask(self.cfg)
        task._prepare_out_dir()

        # 验证文件是否被删除
        self.assertFalse(os.path.exists(existing_file))
        # 验证目录是否存在
        self.assertTrue(os.path.exists(expected_out_dir))

    def test_refresh_cfg(self):
        """测试刷新配置方法"""
        task = TAU2BenchTask(self.cfg)
        # 手动复制模型参数到数据集参数，模拟 _refresh_cfg 的行为
        model_params = self.cfg["models"][0]
        # 获取 task 实例中的配置，因为 BaseTask 会深拷贝配置
        dataset_args = task.cfg["datasets"][0][0]["args"]

        # 执行测试
        task._refresh_cfg()

        # 验证模型参数是否复制到数据集参数
        for key, value in model_params.items():
            if key != "type":
                self.assertEqual(dataset_args.get(key), value)
        # 验证 type 参数是否未复制
        self.assertNotIn("type", dataset_args)

    def test_construct_run_cfg(self):
        """测试构建运行配置方法"""
        task = TAU2BenchTask(self.cfg)
        task._refresh_cfg()
        run_cfg = task._construct_run_cfg()

        # 验证配置是否正确构建
        # 由于 RunConfig 被 mock，我们只需要确保方法能够正常执行
        self.assertIsNotNone(run_cfg)

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
        task.run_config = mock.MagicMock()
        task.run_config.num_trials = 2

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

    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.get_tasks')
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.compute_metrics')
    def test_dump_eval_results_no_metrics(self, mock_compute_metrics, mock_get_tasks):
        """测试导出评估结果方法 - 无 metrics 的情况"""
        # 模拟 compute_metrics 返回 None
        mock_compute_metrics.return_value = None

        task = TAU2BenchTask(self.cfg)
        task._prepare_out_dir()
        task._refresh_cfg()
        task.run_config = mock.MagicMock()

        # 执行测试
        task._dump_eval_results({})

        # 验证结果文件是否未创建（因为 metrics 为 None）
        expected_out_json = os.path.join(self.temp_dir, "results", "gpt-3.5-turbo", "test_dataset.json")
        self.assertFalse(os.path.exists(expected_out_json))

    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.run_domain')
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.compute_metrics')
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.get_tasks')
    @mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.tqdm')
    def test_run(self, mock_tqdm, mock_get_tasks, mock_compute_metrics, mock_run_domain):
        """测试 run 方法"""
        # 模拟依赖
        mock_get_tasks.return_value = [1, 2, 3]

        # 模拟 run_domain 函数，创建 save_to 文件并写入任务数据
        def mock_run_domain_func(run_config):
            # 确保 save_to 有值且是绝对路径
            if not run_config.save_to:
                save_to_file = os.path.join(self.temp_dir, "test_save_to.json")
            else:
                # 确保路径是绝对路径
                if not os.path.isabs(run_config.save_to):
                    save_to_file = os.path.join(self.temp_dir, f"{run_config.save_to}.json")
                else:
                    save_to_file = f"{run_config.save_to}.json"
            # 确保目录存在
            dir_name = os.path.dirname(save_to_file)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            else:
                # 如果没有目录部分，使用临时目录
                save_to_file = os.path.join(self.temp_dir, os.path.basename(save_to_file))
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

        # 模拟 tqdm
        mock_pbar = mock.MagicMock()
        mock_tqdm_instance = mock.MagicMock()
        mock_tqdm_instance.__enter__.return_value = mock_pbar
        mock_tqdm_instance.__exit__.return_value = None
        mock_tqdm.return_value = mock_tqdm_instance

        task = TAU2BenchTask(self.cfg)

        # 执行测试
        task.run(self.task_state_manager)

        # 验证方法调用
        mock_run_domain.assert_called_once()
        mock_compute_metrics.assert_called_once()
        # 验证任务状态更新
        self.task_state_manager.update_task_state.assert_called()



    def test_patched_functions(self):
        """测试补丁函数"""
        # 测试 auto_y_input 函数
        from ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task import auto_y_input
        result = auto_y_input("Do you want to continue?")
        self.assertEqual(result, "y")

    def test_construct_run_cfg_with_none_values(self):
        """测试构建运行配置方法 - 包含 None 值的情况"""
        task = TAU2BenchTask(self.cfg)
        # 添加 None 值到数据集参数
        task.cfg["datasets"][0][0]["args"]["none_param"] = None
        task._refresh_cfg()
        run_cfg = task._construct_run_cfg()
        # 验证配置是否正确构建
        self.assertIsNotNone(run_cfg)

    def test_parse_args(self):
        """测试 parse_args 函数"""
        from ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task import parse_args
        # 模拟命令行参数
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ["tau2_bench_task.py", "test_config.py"]
            args = parse_args()
            self.assertEqual(args.config, "test_config.py")
        finally:
            sys.argv = original_argv

    def test_get_task_count_with_task_set_name(self):
        """测试获取任务数量方法 - 有 task_set_name 的情况"""
        # 模拟 get_tasks 返回 3 个任务
        from ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task import get_tasks
        with mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.get_tasks') as mock_get_tasks:
            mock_get_tasks.return_value = [1, 2, 3]

            task = TAU2BenchTask(self.cfg)
            task._refresh_cfg()
            run_cfg = task._construct_run_cfg()
            # 设置 task_set_name
            run_cfg.task_set_name = "test_task_set"
            task_count = task._get_task_count(run_cfg)

            # 验证任务数量是否正确
            self.assertEqual(task_count, 3)
            # 验证 get_tasks 是否被正确调用
            mock_get_tasks.assert_called_once()

    def test_get_task_count_without_task_set_name(self):
        """测试获取任务数量方法 - 无 task_set_name 的情况"""
        # 模拟 get_tasks 返回 5 个任务
        from ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task import get_tasks
        with mock.patch('ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task.get_tasks') as mock_get_tasks:
            mock_get_tasks.return_value = [1, 2, 3, 4, 5]

            task = TAU2BenchTask(self.cfg)
            task._refresh_cfg()
            run_cfg = task._construct_run_cfg()
            # 确保 task_set_name 为 None
            run_cfg.task_set_name = None
            # 设置 domain
            run_cfg.domain = "test_domain"
            task_count = task._get_task_count(run_cfg)

            # 验证任务数量是否正确
            self.assertEqual(task_count, 5)
            # 验证 get_tasks 是否被正确调用
            mock_get_tasks.assert_called_once()


if __name__ == '__main__':
    unittest.main()
