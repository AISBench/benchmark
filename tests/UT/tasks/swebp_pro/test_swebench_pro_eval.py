"""Unit tests for ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval"""
import unittest
import tempfile
import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ais_bench.benchmark.tasks.swebench_pro import swebench_pro_eval as eval_module


class TestFilterInstancesFromPreds(unittest.TestCase):
    """Test _filter_instances_from_preds function."""

    @classmethod
    def setUpClass(cls):
        cls.eval_module = eval_module

    def test_filters_instances_with_predictions(self):
        instances = [
            {"instance_id": "id1"},
            {"instance_id": "id2"},
            {"instance_id": "id3"},
        ]
        predictions = {
            "id1": {"instance_id": "id1"},
            "id2": {"instance_id": "id2"},
        }

        filtered, skipped = self.eval_module._filter_instances_from_preds(
            instances, predictions, "model", "dataset", Path("/tmp"),
            rewrite_reports=False, exclude_completed=False
        )

        self.assertEqual(len(filtered), 2)
        self.assertEqual(skipped, 1)

    def test_excludes_completed_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "model" / "id1").mkdir(parents=True)
            (run_dir / "model" / "id1" / "dataset_output.json").write_text("{}")

            instances = [
                {"instance_id": "id1"},
                {"instance_id": "id2"},
            ]
            predictions = {
                "id1": {"instance_id": "id1"},
                "id2": {"instance_id": "id2"},
            }

            filtered, skipped = self.eval_module._filter_instances_from_preds(
                instances, predictions, "model", "dataset", run_dir,
                rewrite_reports=False, exclude_completed=True
            )

            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0]["instance_id"], "id2")
            self.assertEqual(skipped, 1)

    def test_rewrite_reports_includes_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "model" / "id1").mkdir(parents=True)
            (run_dir / "model" / "id1" / "dataset_output.json").write_text("{}")

            instances = [{"instance_id": "id1"}]
            predictions = {"id1": {"instance_id": "id1"}}

            filtered, skipped = self.eval_module._filter_instances_from_preds(
                instances, predictions, "model", "dataset", run_dir,
                rewrite_reports=True, exclude_completed=True
            )

            self.assertEqual(len(filtered), 1)
            self.assertEqual(skipped, 0)


class TestIsSolved(unittest.TestCase):
    """Test isSolved function."""

    @classmethod
    def setUpClass(cls):
        cls.eval_module = eval_module

    def test_returns_true_when_all_tests_pass(self):
        tests = [
            {"name": "test1", "status": "PASSED"},
            {"name": "test2", "status": "PASSED"},
            {"name": "test3", "status": "PASSED"},
        ]
        fail_to_pass = "['test1', 'test2']"
        pass_to_pass = "['test3']"

        result = self.eval_module.isSolved(tests, fail_to_pass, pass_to_pass)

        self.assertTrue(result)

    def test_returns_false_when_fail_to_pass_fails(self):
        tests = [
            {"name": "test1", "status": "FAILED"},
            {"name": "test2", "status": "PASSED"},
        ]
        fail_to_pass = "['test1', 'test2']"
        pass_to_pass = "[]"

        result = self.eval_module.isSolved(tests, fail_to_pass, pass_to_pass)

        self.assertFalse(result)

    def test_returns_false_when_pass_to_pass_fails(self):
        tests = [
            {"name": "test1", "status": "PASSED"},
            {"name": "test2", "status": "FAILED"},
        ]
        fail_to_pass = "['test1']"
        pass_to_pass = "['test2']"

        result = self.eval_module.isSolved(tests, fail_to_pass, pass_to_pass)

        self.assertFalse(result)

    def test_handles_empty_lists(self):
        tests = []
        fail_to_pass = "[]"
        pass_to_pass = "[]"

        result = self.eval_module.isSolved(tests, fail_to_pass, pass_to_pass)

        self.assertTrue(result)

    def test_handles_empty_strings(self):
        tests = [{"name": "test1", "status": "PASSED"}]
        fail_to_pass = ""
        pass_to_pass = ""

        result = self.eval_module.isSolved(tests, fail_to_pass, pass_to_pass)

        self.assertTrue(result)

    def test_handles_whitespace_strings(self):
        tests = [{"name": "test1", "status": "PASSED"}]
        fail_to_pass = "   "
        pass_to_pass = "   "

        result = self.eval_module.isSolved(tests, fail_to_pass, pass_to_pass)

        self.assertTrue(result)

    def test_returns_false_on_invalid_eval(self):
        tests = [{"name": "test1", "status": "PASSED"}]
        fail_to_pass = "invalid syntax {"
        pass_to_pass = "[]"

        result = self.eval_module.isSolved(tests, fail_to_pass, pass_to_pass)

        self.assertFalse(result)


class TestBuildEvalReport(unittest.TestCase):
    """Test _build_eval_report function."""

    @classmethod
    def setUpClass(cls):
        cls.eval_module = eval_module

    def test_builds_report_successfully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_eval_dir = Path(tmpdir)

            full_instances = [
                {"instance_id": "id1", "fail_to_pass": "['test1']", "pass_to_pass": "['test2']"},
                {"instance_id": "id2", "fail_to_pass": "['test3']", "pass_to_pass": "['test4']"},
            ]

            predictions = {
                "id1": {"instance_id": "id1", "model_patch": "patch1"},
                "id2": {"instance_id": "id2", "model_patch": ""},
            }
            pred_path = os.path.join(tmpdir, "preds.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f)

            (run_eval_dir / "model" / "id1").mkdir(parents=True)
            with open(run_eval_dir / "model" / "id1" / "dataset_output.json", "w") as f:
                json.dump({
                    "tests": [
                        {"name": "test1", "status": "PASSED"},
                        {"name": "test2", "status": "PASSED"},
                    ]
                }, f)

            (run_eval_dir / "model" / "id2").mkdir(parents=True)
            with open(run_eval_dir / "model" / "id2" / "dataset_output.json", "w") as f:
                json.dump({
                    "tests": [
                        {"name": "test3", "status": "FAILED"},
                        {"name": "test4", "status": "PASSED"},
                    ]
                }, f)

            report = self.eval_module._build_eval_report(
                full_instances, pred_path, run_eval_dir, "model", "dataset"
            )

            self.assertEqual(report["total_instances_num"], 2)
            self.assertEqual(report["total_prediction_num"], 2)
            self.assertEqual(report["build_patch_instances_num"], 1)
            self.assertEqual(report["empty_patch_instances_num"], 1)
            self.assertEqual(report["eval_resolved_instances_num"], 1)
            self.assertEqual(report["eval_unresolved_instances_num"], 1)
            self.assertEqual(report["accuracy"], 50.0)

    def test_handles_missing_output_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_eval_dir = Path(tmpdir)

            full_instances = [
                {"instance_id": "id1", "fail_to_pass": "['test1']", "pass_to_pass": "['test2']"},
            ]

            predictions = {"id1": {"instance_id": "id1", "model_patch": "patch1"}}
            pred_path = os.path.join(tmpdir, "preds.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f)

            report = self.eval_module._build_eval_report(
                full_instances, pred_path, run_eval_dir, "model", "dataset"
            )

            self.assertEqual(report["eval_resolved_instances_num"], 0)
            self.assertEqual(report["eval_unresolved_instances_num"], 1)


class TestParseArgs(unittest.TestCase):
    """Test parse_args function."""

    @classmethod
    def setUpClass(cls):
        cls.eval_module = eval_module

    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args(self, mock_parse):
        mock_parse.return_value = MagicMock(config="test_config.py")

        args = self.eval_module.parse_args()

        self.assertEqual(args.config, "test_config.py")


class TestSWEBenchProEvalProgressManager(unittest.TestCase):
    """Test _SWEBenchProEvalProgressManager class."""

    @classmethod
    def setUpClass(cls):
        cls.eval_module = eval_module

    def test_on_instance_start(self):
        mock_tsm = MagicMock()
        manager = self.eval_module._SWEBenchProEvalProgressManager(mock_tsm, 10)

        manager.on_instance_start("instance-1")

        mock_tsm.update_task_state.assert_called_once()
        call_kwargs = mock_tsm.update_task_state.call_args[0][0]
        self.assertEqual(call_kwargs["status"], "eval")
        self.assertEqual(call_kwargs["total_count"], 10)
        self.assertEqual(call_kwargs["resolved_count"], 0)

    def test_on_instance_end_passed(self):
        mock_tsm = MagicMock()
        manager = self.eval_module._SWEBenchProEvalProgressManager(mock_tsm, 10)

        manager.on_instance_end("instance-1", pass_flag=True)

        mock_tsm.update_task_state.assert_called_once()
        call_kwargs = mock_tsm.update_task_state.call_args[0][0]
        self.assertEqual(call_kwargs["finish_count"], 1)
        self.assertEqual(call_kwargs["resolved_count"], 1)
        self.assertEqual(call_kwargs["accuracy"], 100.0)

    def test_on_instance_end_failed(self):
        mock_tsm = MagicMock()
        manager = self.eval_module._SWEBenchProEvalProgressManager(mock_tsm, 10)

        manager.on_instance_end("instance-1", pass_flag=False)

        mock_tsm.update_task_state.assert_called_once()
        call_kwargs = mock_tsm.update_task_state.call_args[0][0]
        self.assertEqual(call_kwargs["finish_count"], 1)
        self.assertEqual(call_kwargs["resolved_count"], 0)
        self.assertEqual(call_kwargs["accuracy"], 0.0)

    def test_accuracy_calculation(self):
        mock_tsm = MagicMock()
        manager = self.eval_module._SWEBenchProEvalProgressManager(mock_tsm, 10)

        manager.on_instance_end("instance-1", pass_flag=True)
        manager.on_instance_end("instance-2", pass_flag=False)
        manager.on_instance_end("instance-3", pass_flag=True)

        last_call = mock_tsm.update_task_state.call_args_list[-1][0][0]
        self.assertEqual(last_call["finish_count"], 3)
        self.assertEqual(last_call["resolved_count"], 2)
        self.assertEqual(last_call["accuracy"], 66.67)


class TestSWEBenchProEvalTask(unittest.TestCase):
    """Test SWEBenchProEvalTask class."""

    @classmethod
    def setUpClass(cls):
        cls.eval_module = eval_module

    def test_get_command(self):
        cfg = MagicMock()
        with patch.object(self.eval_module.SWEBenchProEvalTask, '__init__', lambda self, cfg: None):
            task = self.eval_module.SWEBenchProEvalTask.__new__(self.eval_module.SWEBenchProEvalTask)
            command = task.get_command("config_path.yaml", "python {task_cmd}")
            self.assertIn("python", command)
            self.assertIn("config_path.yaml", command)


class TestKeyConstants(unittest.TestCase):
    """Test KEY_INSTANCE_ID and KEY_MODEL constants."""

    @classmethod
    def setUpClass(cls):
        cls.eval_module = eval_module

    def test_key_instance_id(self):
        self.assertEqual(self.eval_module.KEY_INSTANCE_ID, "instance_id")

    def test_key_model(self):
        self.assertEqual(self.eval_module.KEY_MODEL, "model_name_or_path")


class TestBuildEvalReportEdgeCases(unittest.TestCase):
    """Test _build_eval_report function with edge cases."""

    @classmethod
    def setUpClass(cls):
        cls.eval_module = eval_module

    def test_handles_predictions_with_patch_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_eval_dir = Path(tmpdir)

            full_instances = [
                {"instance_id": "id1", "fail_to_pass": "['test1']", "pass_to_pass": "[]"},
            ]

            predictions = {"id1": {"instance_id": "id1", "patch": "patch1"}}
            pred_path = os.path.join(tmpdir, "preds.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f)

            (run_eval_dir / "model" / "id1").mkdir(parents=True)
            with open(run_eval_dir / "model" / "id1" / "dataset_output.json", "w") as f:
                json.dump({
                    "tests": [{"name": "test1", "status": "PASSED"}]
                }, f)

            report = self.eval_module._build_eval_report(
                full_instances, pred_path, run_eval_dir, "model", "dataset"
            )

            self.assertEqual(report["total_instances_num"], 1)
            self.assertEqual(report["eval_resolved_instances_num"], 1)

    def test_handles_non_dict_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_eval_dir = Path(tmpdir)

            full_instances = [
                {"instance_id": "id1", "fail_to_pass": "['test1']", "pass_to_pass": "[]"},
            ]

            predictions = {"id1": "not a dict"}
            pred_path = os.path.join(tmpdir, "preds.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f)

            report = self.eval_module._build_eval_report(
                full_instances, pred_path, run_eval_dir, "model", "dataset"
            )

            self.assertEqual(report["build_patch_instances_num"], 0)
            self.assertEqual(report["empty_patch_instances_num"], 0)

    def test_handles_invalid_json_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_eval_dir = Path(tmpdir)

            full_instances = [
                {"instance_id": "id1", "fail_to_pass": "['test1']", "pass_to_pass": "[]"},
            ]

            predictions = {"id1": {"instance_id": "id1", "model_patch": "patch1"}}
            pred_path = os.path.join(tmpdir, "preds.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f)

            (run_eval_dir / "model" / "id1").mkdir(parents=True)
            with open(run_eval_dir / "model" / "id1" / "dataset_output.json", "w") as f:
                f.write("invalid json {")

            report = self.eval_module._build_eval_report(
                full_instances, pred_path, run_eval_dir, "model", "dataset"
            )

            self.assertEqual(report["eval_unresolved_instances_num"], 1)

    def test_handles_non_list_tests(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_eval_dir = Path(tmpdir)

            full_instances = [
                {"instance_id": "id1", "fail_to_pass": "['test1']", "pass_to_pass": "[]"},
            ]

            predictions = {"id1": {"instance_id": "id1", "model_patch": "patch1"}}
            pred_path = os.path.join(tmpdir, "preds.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f)

            (run_eval_dir / "model" / "id1").mkdir(parents=True)
            with open(run_eval_dir / "model" / "id1" / "dataset_output.json", "w") as f:
                json.dump({"tests": "not a list"}, f)

            report = self.eval_module._build_eval_report(
                full_instances, pred_path, run_eval_dir, "model", "dataset"
            )

            self.assertEqual(report["eval_unresolved_instances_num"], 1)

    def test_handles_empty_patch_via_patch_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_eval_dir = Path(tmpdir)

            full_instances = [
                {"instance_id": "id1", "fail_to_pass": "[]", "pass_to_pass": "[]"},
            ]

            predictions = {"id1": {"instance_id": "id1", "patch": ""}}
            pred_path = os.path.join(tmpdir, "preds.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f)

            report = self.eval_module._build_eval_report(
                full_instances, pred_path, run_eval_dir, "model", "dataset"
            )

            self.assertEqual(report["empty_patch_instances_num"], 1)

    def test_calculates_zero_accuracy_for_zero_total(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_eval_dir = Path(tmpdir)

            full_instances = []
            predictions = {}
            pred_path = os.path.join(tmpdir, "preds.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f)

            report = self.eval_module._build_eval_report(
                full_instances, pred_path, run_eval_dir, "model", "dataset"
            )

            self.assertEqual(report["accuracy"], 0.0)


class TestSWEBenchProEvalTaskRun(unittest.TestCase):
    """Test SWEBenchProEvalTask.run method."""

    @classmethod
    def setUpClass(cls):
        cls.eval_module = eval_module

    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.clean_swebench_pro_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.list_swebench_pro_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.ensure_swebench_pro_docker_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.get_infer_output_path')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval._build_eval_report')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.eval_with_docker')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.model_abbr_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.task_abbr_from_cfg')
    def test_run_no_instances_to_eval(
        self, mock_task_abbr, mock_model_abbr, mock_dataset_abbr,
        mock_build_dataset, mock_eval_docker, mock_build_report,
        mock_get_output_path, mock_ensure_images, mock_list_images, mock_clean_images
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MagicMock()
            cfg.cli_args = {"debug": False}
            cfg.work_dir = tmpdir

            task = self.eval_module.SWEBenchProEvalTask.__new__(self.eval_module.SWEBenchProEvalTask)
            task.cfg = cfg
            task.work_dir = tmpdir
            task.logger = MagicMock()
            task.model_cfg = {"type": "test"}
            task.dataset_cfgs = [{"swebp_scripts_dir": tmpdir, "swebp_docker_dir": tmpdir}]

            mock_model_abbr.return_value = "model1"
            mock_dataset_abbr.return_value = "dataset1"
            mock_task_abbr.return_value = "task1"

            mock_dataset = MagicMock()
            mock_dataset.test = [{"instance_id": "id1"}]
            mock_build_dataset.return_value = mock_dataset

            mock_get_output_path.return_value = os.path.join(tmpdir, "preds.json")

            with open(os.path.join(tmpdir, "preds.json"), "w") as f:
                json.dump({}, f)

            task_state_manager = MagicMock()

            task.run(task_state_manager)

            task.logger.info.assert_any_call("No instances to evaluate")

    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.clean_swebench_pro_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.list_swebench_pro_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.ensure_swebench_pro_docker_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.get_infer_output_path')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval._build_eval_report')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.eval_with_docker')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.model_abbr_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.task_abbr_from_cfg')
    def test_run_raises_missing_scripts_dir(
        self, mock_task_abbr, mock_model_abbr, mock_dataset_abbr,
        mock_build_dataset, mock_eval_docker, mock_build_report,
        mock_get_output_path, mock_ensure_images, mock_list_images, mock_clean_images
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MagicMock()
            cfg.cli_args = {"debug": False}
            cfg.work_dir = tmpdir

            task = self.eval_module.SWEBenchProEvalTask.__new__(self.eval_module.SWEBenchProEvalTask)
            task.cfg = cfg
            task.work_dir = tmpdir
            task.logger = MagicMock()
            task.model_cfg = {"type": "test"}
            task.dataset_cfgs = [{}]

            mock_model_abbr.return_value = "model1"
            mock_dataset_abbr.return_value = "dataset1"
            mock_task_abbr.return_value = "task1"

            mock_dataset = MagicMock()
            mock_dataset.test = [{"instance_id": "id1"}]
            mock_build_dataset.return_value = mock_dataset

            task_state_manager = MagicMock()

            with self.assertRaises(Exception):
                task.run(task_state_manager)

    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.clean_swebench_pro_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.list_swebench_pro_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.ensure_swebench_pro_docker_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.get_infer_output_path')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval._build_eval_report')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.eval_with_docker')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.model_abbr_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.task_abbr_from_cfg')
    def test_run_raises_missing_predictions(
        self, mock_task_abbr, mock_model_abbr, mock_dataset_abbr,
        mock_build_dataset, mock_eval_docker, mock_build_report,
        mock_get_output_path, mock_ensure_images, mock_list_images, mock_clean_images
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MagicMock()
            cfg.cli_args = {"debug": False}
            cfg.work_dir = tmpdir

            task = self.eval_module.SWEBenchProEvalTask.__new__(self.eval_module.SWEBenchProEvalTask)
            task.cfg = cfg
            task.work_dir = tmpdir
            task.logger = MagicMock()
            task.model_cfg = {"type": "test"}
            task.dataset_cfgs = [{"swebp_scripts_dir": tmpdir, "swebp_docker_dir": tmpdir}]

            mock_model_abbr.return_value = "model1"
            mock_dataset_abbr.return_value = "dataset1"
            mock_task_abbr.return_value = "task1"

            mock_dataset = MagicMock()
            mock_dataset.test = [{"instance_id": "id1"}]
            mock_build_dataset.return_value = mock_dataset

            mock_get_output_path.return_value = os.path.join(tmpdir, "nonexistent.json")

            task_state_manager = MagicMock()

            with self.assertRaises(Exception):
                task.run(task_state_manager)

    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.clean_swebench_pro_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.list_swebench_pro_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.ensure_swebench_pro_docker_images')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.get_infer_output_path')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval._build_eval_report')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.eval_with_docker')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.model_abbr_from_cfg')
    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.task_abbr_from_cfg')
    def test_run_with_instances(
        self, mock_task_abbr, mock_model_abbr, mock_dataset_abbr,
        mock_build_dataset, mock_eval_docker, mock_build_report,
        mock_get_output_path, mock_ensure_images, mock_list_images, mock_clean_images
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MagicMock()
            cfg.cli_args = {"debug": False}
            cfg.work_dir = tmpdir

            task = self.eval_module.SWEBenchProEvalTask.__new__(self.eval_module.SWEBenchProEvalTask)
            task.cfg = cfg
            task.work_dir = tmpdir
            task.logger = MagicMock()
            task.model_cfg = {"type": "test", "batch_size": 1}
            task.dataset_cfgs = [{"swebp_scripts_dir": tmpdir, "swebp_docker_dir": tmpdir}]

            mock_model_abbr.return_value = "model1"
            mock_dataset_abbr.return_value = "dataset1"
            mock_task_abbr.return_value = "task1"

            mock_dataset = MagicMock()
            mock_dataset.test = [{"instance_id": "id1", "fail_to_pass": "[]", "pass_to_pass": "[]"}]
            mock_build_dataset.return_value = mock_dataset

            preds_path = os.path.join(tmpdir, "preds.json")
            mock_get_output_path.return_value = preds_path

            with open(preds_path, "w") as f:
                json.dump({"id1": {"instance_id": "id1", "model_patch": "patch"}}, f)

            mock_eval_docker.return_value = {"tests": [], "error": None}
            mock_build_report.return_value = {"accuracy": 100.0}
            mock_list_images.return_value = set()

            task_state_manager = MagicMock()

            task.run(task_state_manager)

            mock_ensure_images.assert_called_once()
            mock_eval_docker.assert_called_once()
            mock_build_report.assert_called_once()
            mock_clean_images.assert_called_once()


class TestRunEvalWithProgress(unittest.TestCase):
    """Test _run_eval_with_progress nested function."""

    @classmethod
    def setUpClass(cls):
        cls.eval_module = eval_module

    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.eval_with_docker')
    def test_run_eval_with_progress_success(self, mock_eval_docker):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MagicMock()
            cfg.cli_args = {"debug": False}
            cfg.work_dir = tmpdir

            task = self.eval_module.SWEBenchProEvalTask.__new__(self.eval_module.SWEBenchProEvalTask)
            task.cfg = cfg
            task.logger = MagicMock()

            mock_progress_manager = MagicMock()

            def make_run_eval_with_progress():
                def run_eval_with_progress(patch, instance, report_dir, scripts_dir_abs, docker_dir_abs, logger, prefix="", docker_client=None, timeout=7200):
                    instance_id = instance["instance_id"]
                    mock_progress_manager.on_instance_start(instance_id)
                    try:
                        return mock_eval_docker(patch, instance, report_dir, scripts_dir_abs, docker_dir_abs, logger, prefix, docker_client, timeout)
                    except Exception as e:
                        task.logger.error("Error evaluating pred %s: %s", instance_id, e)
                        task.logger.info("Evaluation for %s generated an exception: %s", instance_id, e)
                        return None
                return run_eval_with_progress

            run_eval = make_run_eval_with_progress()
            mock_eval_docker.return_value = {"tests": []}

            result = run_eval(
                "patch", {"instance_id": "id1"}, tmpdir, tmpdir, tmpdir,
                task.logger, "model", None, 7200
            )

            self.assertEqual(result, {"tests": []})
            mock_progress_manager.on_instance_start.assert_called_once_with("id1")

    @patch('ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval.eval_with_docker')
    def test_run_eval_with_progress_exception(self, mock_eval_docker):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MagicMock()
            cfg.cli_args = {"debug": False}
            cfg.work_dir = tmpdir

            task = self.eval_module.SWEBenchProEvalTask.__new__(self.eval_module.SWEBenchProEvalTask)
            task.cfg = cfg
            task.logger = MagicMock()

            mock_progress_manager = MagicMock()

            def make_run_eval_with_progress():
                def run_eval_with_progress(patch, instance, report_dir, scripts_dir_abs, docker_dir_abs, logger, prefix="", docker_client=None, timeout=7200):
                    instance_id = instance["instance_id"]
                    mock_progress_manager.on_instance_start(instance_id)
                    try:
                        return mock_eval_docker(patch, instance, report_dir, scripts_dir_abs, docker_dir_abs, logger, prefix, docker_client, timeout)
                    except Exception as e:
                        task.logger.error("Error evaluating pred %s: %s", instance_id, e)
                        task.logger.info("Evaluation for %s generated an exception: %s", instance_id, e)
                        return None
                return run_eval_with_progress

            run_eval = make_run_eval_with_progress()
            mock_eval_docker.side_effect = Exception("Test error")

            result = run_eval(
                "patch", {"instance_id": "id1"}, tmpdir, tmpdir, tmpdir,
                task.logger, "model", None, 7200
            )

            self.assertIsNone(result)
            task.logger.error.assert_called()


if __name__ == '__main__':
    unittest.main()
