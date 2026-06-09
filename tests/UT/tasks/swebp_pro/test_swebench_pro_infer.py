"""Unit tests for ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer"""
import unittest
import tempfile
import os
import sys
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ais_bench.benchmark.tasks.swebench_pro import swebench_pro_infer as infer_module


class TestGetMinisweagentConfig(unittest.TestCase):
    """Test _get_minisweagent_config function."""

    @classmethod
    def setUpClass(cls):
        cls.infer_module = infer_module

    def test_returns_dict_with_model_key(self):
        cfg = {"model": "test-model", "generation_kwargs": {"temperature": 0.7}}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertIn("model", result)
        self.assertIn("model_name", result["model"])
        self.assertIn("model_class", result["model"])
        self.assertIn("model_kwargs", result["model"])

    def test_handles_empty_config(self):
        result = self.infer_module._get_minisweagent_config({})
        self.assertIsInstance(result, dict)
        self.assertIn("model", result)

    def test_sets_model_name_from_model_key(self):
        cfg = {"model": "gpt-4"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_name"], "gpt-4")

    def test_sets_model_name_from_model_name_key(self):
        cfg = {"model_name": "claude-3"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_name"], "claude-3")

    def test_includes_api_key_in_model_kwargs(self):
        cfg = {"api_key": "sk-test-123"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_kwargs"]["api_key"], "sk-test-123")

    def test_includes_url_as_api_base(self):
        cfg = {"url": "https://api.example.com/v1"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_kwargs"]["api_base"], "https://api.example.com/v1")


class TestAISBenchProgressManager(unittest.TestCase):
    """Test _AISBenchProgressManager class."""

    @classmethod
    def setUpClass(cls):
        cls.infer_module = infer_module

    def test_on_instance_start(self):
        mock_tsm = MagicMock()
        manager = self.infer_module._AISBenchProgressManager(mock_tsm, 10)

        manager.on_instance_start("instance-1")

        mock_tsm.update_task_state.assert_called_once()
        call_kwargs = mock_tsm.update_task_state.call_args[0][0]
        self.assertEqual(call_kwargs["status"], "inferencing")
        self.assertEqual(call_kwargs["total_count"], 10)

    def test_on_instance_end(self):
        mock_tsm = MagicMock()
        manager = self.infer_module._AISBenchProgressManager(mock_tsm, 10)

        manager.on_instance_end("instance-1")

        mock_tsm.update_task_state.assert_called_once()
        call_kwargs = mock_tsm.update_task_state.call_args[0][0]
        self.assertEqual(call_kwargs["finish_count"], 1)
        self.assertEqual(call_kwargs["total_count"], 10)

    def test_finish_count_increments(self):
        mock_tsm = MagicMock()
        manager = self.infer_module._AISBenchProgressManager(mock_tsm, 10)

        manager.on_instance_end("id1")
        manager.on_instance_end("id2")
        manager.on_instance_end("id3")

        last_call = mock_tsm.update_task_state.call_args_list[-1][0][0]
        self.assertEqual(last_call["finish_count"], 3)

    def test_update_instance_status(self):
        mock_tsm = MagicMock()
        manager = self.infer_module._AISBenchProgressManager(mock_tsm, 10)

        manager.update_instance_status("instance-1", "building image")

        mock_tsm.update_task_state.assert_called_once()
        call_kwargs = mock_tsm.update_task_state.call_args[0][0]
        self.assertEqual(call_kwargs["other_kwargs"]["message"], "building image")


class TestParseArgs(unittest.TestCase):
    """Test parse_args function."""

    @classmethod
    def setUpClass(cls):
        cls.infer_module = infer_module

    @patch('argparse.ArgumentParser.parse_args')
    def test_parses_config_argument(self, mock_parse):
        mock_parse.return_value = MagicMock(config="test_config.py", infer_kwargs="{}")

        args = self.infer_module.parse_args()

        self.assertEqual(args.config, "test_config.py")

    @patch('argparse.ArgumentParser.parse_args')
    def test_parses_infer_kwargs(self, mock_parse):
        mock_parse.return_value = MagicMock(
            config="test_config.py",
            infer_kwargs='{"key": "value"}'
        )

        args = self.infer_module.parse_args()

        self.assertEqual(args.infer_kwargs, '{"key": "value"}')


class TestAISBenchProgressManagerOnUncaughtException(unittest.TestCase):
    """Test _AISBenchProgressManager.on_uncaught_exception method."""

    @classmethod
    def setUpClass(cls):
        cls.infer_module = infer_module

    def test_on_uncaught_exception(self):
        mock_tsm = MagicMock()
        manager = self.infer_module._AISBenchProgressManager(mock_tsm, 10)

        manager.on_uncaught_exception("instance-1", ValueError("test error"))

        self.assertEqual(mock_tsm.update_task_state.call_count, 1)
        call_kwargs = mock_tsm.update_task_state.call_args[0][0]
        self.assertEqual(call_kwargs["finish_count"], 1)


class TestCompositeProgressManager(unittest.TestCase):
    """Test _CompositeProgressManager class."""

    @classmethod
    def setUpClass(cls):
        cls.infer_module = infer_module

    def test_forwards_on_instance_start(self):
        mock1 = MagicMock()
        mock2 = MagicMock()
        manager = self.infer_module._CompositeProgressManager(mock1, mock2)

        manager.on_instance_start("instance-1")

        mock1.on_instance_start.assert_called_once_with("instance-1")
        mock2.on_instance_start.assert_called_once_with("instance-1")

    def test_forwards_update_instance_status(self):
        mock1 = MagicMock()
        mock2 = MagicMock()
        manager = self.infer_module._CompositeProgressManager(mock1, mock2)

        manager.update_instance_status("instance-1", "message")

        mock1.update_instance_status.assert_called_once_with("instance-1", "message")
        mock2.update_instance_status.assert_called_once_with("instance-1", "message")

    def test_forwards_on_instance_end(self):
        mock1 = MagicMock()
        mock2 = MagicMock()
        manager = self.infer_module._CompositeProgressManager(mock1, mock2)

        manager.on_instance_end("instance-1", "success")

        mock1.on_instance_end.assert_called_once_with("instance-1", "success")
        mock2.on_instance_end.assert_called_once_with("instance-1", "success")

    def test_forwards_on_uncaught_exception(self):
        mock1 = MagicMock()
        mock2 = MagicMock()
        manager = self.infer_module._CompositeProgressManager(mock1, mock2)
        exception = ValueError("test")

        manager.on_uncaught_exception("instance-1", exception)

        mock1.on_uncaught_exception.assert_called_once_with("instance-1", exception)
        mock2.on_uncaught_exception.assert_called_once_with("instance-1", exception)

    def test_filters_none_delegates(self):
        mock1 = MagicMock()
        manager = self.infer_module._CompositeProgressManager(mock1, None)

        manager.on_instance_start("instance-1")

        mock1.on_instance_start.assert_called_once()

    def test_n_completed_property(self):
        mock1 = MagicMock()
        mock2 = MagicMock()
        mock2.n_completed = 5
        manager = self.infer_module._CompositeProgressManager(mock1, mock2)

        self.assertEqual(manager.n_completed, 5)


class TestSWEBenchProInferTask(unittest.TestCase):
    """Test SWEBenchProInferTask class."""

    @classmethod
    def setUpClass(cls):
        cls.infer_module = infer_module

    def test_get_command(self):
        with patch.object(self.infer_module.SWEBenchProInferTask, '__init__', lambda self, cfg: None):
            task = self.infer_module.SWEBenchProInferTask.__new__(self.infer_module.SWEBenchProInferTask)
            command = task.get_command("config_path.yaml", "python {task_cmd}")
            self.assertIn("python", command)
            self.assertIn("config_path.yaml", command)


class TestMakeSWEBenchProProgressManager(unittest.TestCase):
    """Test _make_swebench_pro_progress_manager function."""

    @classmethod
    def setUpClass(cls):
        cls.infer_module = infer_module

    @patch('minisweagent.run.extra.utils.batch_progress.RunBatchProgressManager')
    def test_returns_composite_manager(self, mock_rbpm_class):
        mock_tsm = MagicMock()
        mock_rbpm = MagicMock()
        mock_rbpm_class.return_value = mock_rbpm
        mock_rbpm.render_group = "render_group"

        manager, render_group = self.infer_module._make_swebench_pro_progress_manager(mock_tsm, 10, Path("out_dir"))

        self.assertIsNotNone(manager)
        self.assertEqual(render_group, "render_group")

    def test_handles_import_error(self):
        mock_tsm = MagicMock()
        original_import = __builtins__['__import__']

        def mock_import(name, *args, **kwargs):
            if 'minisweagent.run.extra.utils.batch_progress' in name:
                raise ImportError("module not found")
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            manager, render_group = self.infer_module._make_swebench_pro_progress_manager(mock_tsm, 10, "out_dir")

        self.assertIsNotNone(manager)
        self.assertIsNone(render_group)


class TestGetMinisweagentConfigEdgeCases(unittest.TestCase):
    """Test _get_minisweagent_config function with edge cases."""

    @classmethod
    def setUpClass(cls):
        cls.infer_module = infer_module

    def test_sets_openrouter_class(self):
        cfg = {"type": "OpenRouterModel", "model": "test-model"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_class"], "openrouter")

    def test_sets_litellm_class_by_default(self):
        cfg = {"model": "test-model"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_class"], "litellm")

    def test_sets_hosted_vllm_prefix(self):
        cfg = {"url": "http://localhost:8000/v1", "model": "qwen"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_name"], "hosted_vllm/qwen")

    def test_includes_cost_tracking(self):
        cfg = {"model": "test-model"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["cost_tracking"], "ignore_errors")

    def test_extracts_type_name_from_string(self):
        cfg = {"type": "ais_bench.benchmark.models.OpenRouter", "model": "test"}
        result = self.infer_module._get_minisweagent_config(cfg)
        self.assertEqual(result["model"]["model_class"], "openrouter")


class TestSWEBenchProInferTaskRun(unittest.TestCase):
    """Test SWEBenchProInferTask.run method."""

    @classmethod
    def setUpClass(cls):
        cls.infer_module = infer_module

    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.cleanup_swebench_pro_containers')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.ensure_swebench_pro_docker_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer._make_swebench_pro_progress_manager')
    @patch('yaml.safe_load')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.get_infer_output_path')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.model_abbr_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.task_abbr_from_cfg')
    @patch('concurrent.futures.ThreadPoolExecutor')
    @patch('concurrent.futures.as_completed')
    def test_run_all_instances_done(
        self, mock_as_completed, mock_executor_class, mock_task_abbr, mock_model_abbr,
        mock_build_dataset, mock_get_output_path, mock_yaml_load,
        mock_make_progress, mock_ensure_images, mock_cleanup
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MagicMock()
            cfg.cli_args = {"debug": False}
            cfg.work_dir = tmpdir

            task = self.infer_module.SWEBenchProInferTask.__new__(self.infer_module.SWEBenchProInferTask)
            task.cfg = cfg
            task.work_dir = tmpdir
            task.logger = MagicMock()
            task.model_cfg = {"type": "test", "model": "test-model", "api_key": "test-key", "url": "http://test"}
            task.dataset_cfgs = [{}]

            mock_model_abbr.return_value = "model1"
            mock_task_abbr.return_value = "task1"

            mock_dataset = MagicMock()
            mock_dataset.test = [{"instance_id": "id1"}, {"instance_id": "id2"}]
            mock_build_dataset.return_value = mock_dataset

            mock_get_output_path.return_value = os.path.join(tmpdir, "preds.json")

            with open(os.path.join(tmpdir, "preds.json"), "w") as f:
                json.dump({"id1": {}, "id2": {}}, f)

            mock_yaml_load.return_value = {"model": {}}

            mock_make_progress.return_value = (MagicMock(), None)

            task_state_manager = MagicMock()

            task.run(task_state_manager)

            task.logger.info.assert_any_call("All instances already done, nothing to run.")

    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.cleanup_swebench_pro_containers')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.ensure_swebench_pro_docker_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer._make_swebench_pro_progress_manager')
    @patch('yaml.safe_load')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.get_infer_output_path')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.model_abbr_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.task_abbr_from_cfg')
    @patch('concurrent.futures.ThreadPoolExecutor')
    @patch('concurrent.futures.as_completed')
    def test_run_no_model_set(
        self, mock_as_completed, mock_executor_class, mock_task_abbr, mock_model_abbr,
        mock_build_dataset, mock_get_output_path, mock_yaml_load,
        mock_make_progress, mock_ensure_images, mock_cleanup
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MagicMock()
            cfg.cli_args = {"debug": False}
            cfg.work_dir = tmpdir

            task = self.infer_module.SWEBenchProInferTask.__new__(self.infer_module.SWEBenchProInferTask)
            task.cfg = cfg
            task.work_dir = tmpdir
            task.logger = MagicMock()
            task.model_cfg = {"type": "test"}
            task.dataset_cfgs = [{}]

            mock_model_abbr.return_value = "model1"
            mock_task_abbr.return_value = "task1"

            mock_dataset = MagicMock()
            mock_dataset.test = [{"instance_id": "id1"}]
            mock_build_dataset.return_value = mock_dataset

            mock_get_output_path.return_value = os.path.join(tmpdir, "preds.json")

            mock_yaml_load.return_value = {"model": {}}

            task_state_manager = MagicMock()

            with self.assertRaises(Exception):
                task.run(task_state_manager)

    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.cleanup_swebench_pro_containers')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.ensure_swebench_pro_docker_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer._make_swebench_pro_progress_manager')
    @patch('yaml.safe_load')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.get_infer_output_path')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.model_abbr_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer.task_abbr_from_cfg')
    @patch('concurrent.futures.ThreadPoolExecutor')
    @patch('concurrent.futures.as_completed')
    def test_run_with_instances(
        self, mock_as_completed, mock_executor_class, mock_task_abbr, mock_model_abbr,
        mock_build_dataset, mock_get_output_path, mock_yaml_load,
        mock_make_progress, mock_ensure_images, mock_cleanup
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MagicMock()
            cfg.cli_args = {"debug": False}
            cfg.work_dir = tmpdir

            task = self.infer_module.SWEBenchProInferTask.__new__(self.infer_module.SWEBenchProInferTask)
            task.cfg = cfg
            task.work_dir = tmpdir
            task.logger = MagicMock()
            task.model_cfg = {"type": "test", "model": "test-model", "api_key": "test-key", "batch_size": 1}
            task.dataset_cfgs = [{}]

            mock_model_abbr.return_value = "model1"
            mock_task_abbr.return_value = "task1"

            mock_dataset = MagicMock()
            mock_dataset.test = [{
                "instance_id": "id1",
                "problem_statement": "test problem",
                "repo": "test/repo",
                "base_commit": "abc123"
            }]
            mock_build_dataset.return_value = mock_dataset

            mock_get_output_path.return_value = os.path.join(tmpdir, "preds.json")

            mock_yaml_load.return_value = {"model": {}}

            mock_make_progress.return_value = (MagicMock(), None)

            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor
            mock_future = MagicMock()
            mock_future.result.return_value = None
            mock_executor.submit.return_value = mock_future
            mock_executor.__enter__ = MagicMock(return_value=mock_executor)
            mock_executor.__exit__ = MagicMock(return_value=False)

            mock_as_completed.return_value = []

            task_state_manager = MagicMock()

            task.run(task_state_manager)

            mock_ensure_images.assert_called_once()
            mock_executor_class.assert_called_once()


if __name__ == '__main__':
    unittest.main()
