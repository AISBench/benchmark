"""Unit tests for ais_bench.benchmark.summarizers.swebench_pro"""
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

from ais_bench.benchmark.summarizers import swebench_pro as summarizer_module


class TestSWEBenchProSummarizer(unittest.TestCase):
    """Test SWEBenchProSummarizer class."""

    @classmethod
    def setUpClass(cls):
        cls.summarizer_module = summarizer_module

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.work_dir = os.path.join(self.temp_dir, "work_dir")
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_cfg(self, model_abbr, dataset_abbr):
        model_cfg = {
            "type": "MockModel",
            "model_abbr": model_abbr,
            "abbr": model_abbr,
        }

        dataset_cfg = {
            "type": "MockDataset",
            "dataset_abbr": dataset_abbr,
            "path": f"dataset/path/{dataset_abbr}",
            "dataset_name": dataset_abbr,
            "abbr": dataset_abbr,
        }

        return model_cfg, dataset_cfg

    def test_pick_up_results_success(self):
        results_dir = os.path.join(self.work_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        report_data = {
            "total_instances_num": 100,
            "eval_resolved_instances_num": 50,
            "accuracy": 50.0,
        }
        report_path = os.path.join(results_dir, "model1_dataset1_report.json")
        with open(report_path, "w") as f:
            json.dump(report_data, f)

        model_cfg, dataset_cfg = self._create_mock_cfg("model1", "dataset1")

        summarizer = self.summarizer_module.SWEBenchProSummarizer.__new__(
            self.summarizer_module.SWEBenchProSummarizer
        )
        summarizer.work_dir = self.work_dir
        summarizer.model_cfgs = [model_cfg]
        summarizer.dataset_cfgs = [dataset_cfg]
        summarizer.logger = MagicMock()

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertEqual(raw_results["model1"]["dataset1"]["accuracy"], 50.0)
        self.assertEqual(raw_results["model1"]["dataset1"]["correct_count"], 50)
        self.assertEqual(raw_results["model1"]["dataset1"]["total_count"], 100)
        self.assertEqual(parsed_results["model1"]["dataset1"]["accuracy"], 50.0)
        self.assertEqual(dataset_metrics["dataset1"], ["accuracy"])
        self.assertEqual(dataset_eval_mode["dataset1"], "agent")

    def test_pick_up_results_missing_report(self):
        model_cfg, dataset_cfg = self._create_mock_cfg("model1", "dataset1")

        summarizer = self.summarizer_module.SWEBenchProSummarizer.__new__(
            self.summarizer_module.SWEBenchProSummarizer
        )
        summarizer.work_dir = self.work_dir
        summarizer.model_cfgs = [model_cfg]
        summarizer.dataset_cfgs = [dataset_cfg]
        summarizer.logger = MagicMock()

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertEqual(raw_results["model1"], {})
        self.assertEqual(parsed_results["model1"], {})
        self.assertEqual(dataset_eval_mode["dataset1"], "agent")

    def test_pick_up_results_invalid_json(self):
        results_dir = os.path.join(self.work_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        report_path = os.path.join(results_dir, "model1_dataset1_report.json")
        with open(report_path, "w") as f:
            f.write("invalid json {")

        model_cfg, dataset_cfg = self._create_mock_cfg("model1", "dataset1")

        summarizer = self.summarizer_module.SWEBenchProSummarizer.__new__(
            self.summarizer_module.SWEBenchProSummarizer
        )
        summarizer.work_dir = self.work_dir
        summarizer.model_cfgs = [model_cfg]
        summarizer.dataset_cfgs = [dataset_cfg]
        summarizer.logger = MagicMock()

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertEqual(raw_results["model1"], {})
        self.assertEqual(dataset_eval_mode["dataset1"], "agent")

    def test_pick_up_results_non_dict_report(self):
        results_dir = os.path.join(self.work_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        report_path = os.path.join(results_dir, "model1_dataset1_report.json")
        with open(report_path, "w") as f:
            json.dump(["not", "a", "dict"], f)

        model_cfg, dataset_cfg = self._create_mock_cfg("model1", "dataset1")

        summarizer = self.summarizer_module.SWEBenchProSummarizer.__new__(
            self.summarizer_module.SWEBenchProSummarizer
        )
        summarizer.work_dir = self.work_dir
        summarizer.model_cfgs = [model_cfg]
        summarizer.dataset_cfgs = [dataset_cfg]
        summarizer.logger = MagicMock()

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertEqual(raw_results["model1"], {})
        self.assertEqual(dataset_eval_mode["dataset1"], "agent")

    def test_pick_up_results_multiple_models_and_datasets(self):
        results_dir = os.path.join(self.work_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        for model_idx in [1, 2]:
            for dataset_idx in [1, 2]:
                report_data = {
                    "total_instances_num": 100 + model_idx * 10 + dataset_idx,
                    "eval_resolved_instances_num": 50 + model_idx * 5 + dataset_idx,
                    "accuracy": 50.0 + model_idx * 5 + dataset_idx,
                }
                report_path = os.path.join(
                    results_dir, f"model{model_idx}_dataset{dataset_idx}_report.json"
                )
                with open(report_path, "w") as f:
                    json.dump(report_data, f)

        model_cfg1, dataset_cfg1 = self._create_mock_cfg("model1", "dataset1")
        model_cfg2, dataset_cfg2 = self._create_mock_cfg("model2", "dataset2")

        summarizer = self.summarizer_module.SWEBenchProSummarizer.__new__(
            self.summarizer_module.SWEBenchProSummarizer
        )
        summarizer.work_dir = self.work_dir
        summarizer.model_cfgs = [model_cfg1, model_cfg2]
        summarizer.dataset_cfgs = [dataset_cfg1, dataset_cfg2]
        summarizer.logger = MagicMock()

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("model1", raw_results)
        self.assertIn("model2", raw_results)
        self.assertIn("dataset1", raw_results["model1"])
        self.assertIn("dataset2", raw_results["model1"])
        self.assertEqual(dataset_metrics["dataset1"], ["accuracy"])
        self.assertEqual(dataset_metrics["dataset2"], ["accuracy"])
        self.assertEqual(dataset_eval_mode["dataset1"], "agent")
        self.assertEqual(dataset_eval_mode["dataset2"], "agent")

    def test_pick_up_results_rounds_accuracy(self):
        results_dir = os.path.join(self.work_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        report_data = {
            "total_instances_num": 100,
            "eval_resolved_instances_num": 53,
            "accuracy": 53.3333333,
        }
        report_path = os.path.join(results_dir, "model1_dataset1_report.json")
        with open(report_path, "w") as f:
            json.dump(report_data, f)

        model_cfg, dataset_cfg = self._create_mock_cfg("model1", "dataset1")

        summarizer = self.summarizer_module.SWEBenchProSummarizer.__new__(
            self.summarizer_module.SWEBenchProSummarizer
        )
        summarizer.work_dir = self.work_dir
        summarizer.model_cfgs = [model_cfg]
        summarizer.dataset_cfgs = [dataset_cfg]
        summarizer.logger = MagicMock()

        raw_results, parsed_results, _, _ = summarizer._pick_up_results()

        self.assertEqual(raw_results["model1"]["dataset1"]["accuracy"], 53.33)
        self.assertEqual(parsed_results["model1"]["dataset1"]["accuracy"], 53.33)


if __name__ == '__main__':
    unittest.main()
