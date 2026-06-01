import unittest
import os
import tempfile
import json
import shutil
from unittest import mock
from pathlib import Path

from ais_bench.benchmark.summarizers.harbor import HarborSummarizer, METRIC_WHITELIST, METRIC_BLACKLIST
from ais_bench.benchmark.utils.config import ConfigDict


class TestHarborSummarizer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        self.model_cfg = {
            "type": "TestModel",
            "abbr": "test_model",
            "path": "/the/path/to/test_path"
        }
        self.model_cfg2 = {
            "type": "TestModel2",
            "abbr": "test_model2",
            "summarizer_abbr": "custom_abbr"
        }

        self.dataset_cfg = {
            "type": "TestDataset",
            "abbr": "test_dataset",
            "infer_cfg": {
                "inferencer": {"type": "GenInferencer"}
            }
        }
        self.dataset_cfg2 = {
            "type": "TestDataset2",
            "abbr": "test_dataset2",
            "infer_cfg": {
                "inferencer": {"type": "PPLInferencer"}
            }
        }

        self.config = ConfigDict({
            "models": [self.model_cfg, self.model_cfg2],
            "datasets": [self.dataset_cfg, self.dataset_cfg2],
            "work_dir": self.temp_dir,
            "path": "./test_path"
        })

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    def test_init(self, mock_model_abbr, mock_ais_logger):
        """Test HarborSummarizer initialization"""
        mock_model_abbr.return_value = "test_model"
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        summarizer = HarborSummarizer(self.config)

        self.assertEqual(summarizer.cfg, self.config)
        self.assertEqual(summarizer.work_dir, self.temp_dir)

    def test_metric_whitelist_content(self):
        """Test METRIC_WHITELIST contains expected metrics"""
        expected_metrics = ['avg_score', 'score', 'accuracy', 'n_errors', 'n_total_trials']
        for metric in expected_metrics:
            self.assertIn(metric, METRIC_WHITELIST)

    def test_metric_blacklist_content(self):
        """Test METRIC_BLACKLIST contains expected metrics"""
        expected_metrics = ['bp', 'sys_len', 'ref_len', 'type', 'reward_distribution', 'exception_distribution', 'pass_at_k', 'details']
        for metric in expected_metrics:
            self.assertIn(metric, METRIC_BLACKLIST)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    def test_inheritance_from_default_summarizer(self, mock_model_abbr, mock_ais_logger):
        """Test HarborSummarizer inherits from DefaultSummarizer"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger
        mock_model_abbr.return_value = "test_model"

        from ais_bench.benchmark.summarizers.default import DefaultSummarizer
        summarizer = HarborSummarizer(self.config)
        self.assertIsInstance(summarizer, DefaultSummarizer)

    @mock.patch('builtins.print')
    def test_print_harbor_details_normal(self, mock_print):
        """Test _print_harbor_details with normal results"""
        raw_results = {
            "test_model": {
                "test_dataset": {
                    "total_count": 100,
                    "n_errors": 5,
                    "avg_score": 0.75,
                    "reward_distribution": [
                        {"score": 1.0, "count": 50},
                        {"score": 0.5, "count": 30},
                        {"score": 0.0, "count": 20}
                    ],
                    "exception_distribution": [
                        {"exception_type": "AgentTimeoutError", "count": 3},
                        {"exception_type": "VerifierError", "count": 2}
                    ],
                    "pass_at_k": {"pass@1": 0.5, "pass@5": 0.8}
                }
            }
        }

        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = ["test_model"]
        summarizer._print_harbor_details(raw_results)

        self.assertTrue(mock_print.called)

    @mock.patch('builtins.print')
    def test_print_harbor_details_empty(self, mock_print):
        """Test _print_harbor_details with empty results"""
        raw_results = {}

        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = []
        summarizer._print_harbor_details(raw_results)

    @mock.patch('builtins.print')
    def test_print_harbor_details_no_distributions(self, mock_print):
        """Test _print_harbor_details without distributions"""
        raw_results = {
            "test_model": {
                "test_dataset": {
                    "total_count": 10,
                    "avg_score": 0.8
                }
            }
        }

        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = ["test_model"]
        summarizer._print_harbor_details(raw_results)

    @mock.patch('builtins.print')
    def test_print_harbor_details_with_all_fields(self, mock_print):
        """Test _print_harbor_details with all fields present"""
        raw_results = {
            "model1": {
                "dataset1": {
                    "total_count": 50,
                    "n_errors": 2,
                    "avg_score": 0.85,
                    "reward_distribution": [{"score": 1.0, "count": 30}, {"score": 0.0, "count": 20}],
                    "exception_distribution": [{"exception_type": "Timeout", "count": 2}],
                    "pass_at_k": {"pass@1": 0.6, "pass@3": 0.8}
                }
            }
        }

        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = ["model1"]
        summarizer._print_harbor_details(raw_results)

    @mock.patch('builtins.print')
    def test_print_harbor_details_skips_model_not_in_results(self, mock_print):
        """Test _print_harbor_details skips model not in results"""
        raw_results = {
            "model1": {
                "dataset1": {
                    "total_count": 10,
                    "avg_score": 0.8
                }
            }
        }

        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = ["model1", "model2"]
        summarizer._print_harbor_details(raw_results)

    @mock.patch('builtins.print')
    def test_print_harbor_details_empty_distributions(self, mock_print):
        """Test _print_harbor_details skips empty distributions"""
        raw_results = {
            "model1": {
                "dataset1": {
                    "total_count": 10,
                    "avg_score": 0.8,
                    "reward_distribution": [],
                    "exception_distribution": []
                }
            }
        }

        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = ["model1"]
        summarizer._print_harbor_details(raw_results)

    @mock.patch('builtins.print')
    def test_print_harbor_details_only_total_count(self, mock_print):
        """Test _print_harbor_details with only total_count"""
        raw_results = {
            "model1": {
                "dataset1": {
                    "total_count": 10
                }
            }
        }

        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = ["model1"]
        summarizer._print_harbor_details(raw_results)

    @mock.patch('builtins.print')
    def test_print_harbor_details_only_n_errors(self, mock_print):
        """Test _print_harbor_details with only n_errors"""
        raw_results = {
            "model1": {
                "dataset1": {
                    "n_errors": 5
                }
            }
        }

        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = ["model1"]
        summarizer._print_harbor_details(raw_results)

    @mock.patch('builtins.print')
    def test_print_harbor_details_only_avg_score(self, mock_print):
        """Test _print_harbor_details with only avg_score"""
        raw_results = {
            "model1": {
                "dataset1": {
                    "avg_score": 0.75
                }
            }
        }

        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = ["model1"]
        summarizer._print_harbor_details(raw_results)


class TestHarborSummarizerPickUpResults(unittest.TestCase):
    """Test _pick_up_results method comprehensively"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        self.model_cfg = {
            "type": "TestModel",
            "abbr": "test_model",
            "path": "/the/path/to/test_path"
        }

        self.dataset_cfg = {
            "type": "TestDataset",
            "abbr": "test_dataset",
            "infer_cfg": {
                "inferencer": {"type": "GenInferencer"}
            }
        }

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    def test_pick_up_results_basic(self, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test _pick_up_results basic functionality"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "test_dataset.json")
        with open(result_file, 'w') as f:
            json.dump({"avg_score": 0.85, "accuracy": 0.9}, f)

        mock_mmengine_load.return_value = {"avg_score": 0.85, "accuracy": 0.9}

        config = ConfigDict({
            "models": [self.model_cfg],
            "datasets": [self.dataset_cfg],
            "work_dir": self.temp_dir,
            "path": "./test_path"
        })

        summarizer = HarborSummarizer(config)
        summarizer.model_cfgs = [self.model_cfg]
        summarizer.dataset_cfgs = [self.dataset_cfg]

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_model", raw_results)
        self.assertIn("test_dataset", raw_results["test_model"])
        self.assertEqual(raw_results["test_model"]["test_dataset"]["avg_score"], 0.85)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    def test_pick_up_results_skips_blacklisted_metrics(self, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test that _pick_up_results skips blacklisted metrics"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "test_dataset.json")
        with open(result_file, 'w') as f:
            json.dump({
                "avg_score": 0.85,
                "bp": 0.5,
                "details": "some details",
                "reward_distribution": [{"score": 1.0, "count": 5}]
            }, f)

        mock_mmengine_load.return_value = {
            "avg_score": 0.85,
            "bp": 0.5,
            "details": "some details",
            "reward_distribution": [{"score": 1.0, "count": 5}]
        }

        config = ConfigDict({
            "models": [self.model_cfg],
            "datasets": [self.dataset_cfg],
            "work_dir": self.temp_dir,
            "path": "./test_path"
        })

        summarizer = HarborSummarizer(config)
        summarizer.model_cfgs = [self.model_cfg]
        summarizer.dataset_cfgs = [self.dataset_cfg]

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("bp", raw_results["test_model"]["test_dataset"])
        self.assertNotIn("bp", parsed_results["test_model"]["test_dataset"])
        self.assertIn("avg_score", parsed_results["test_model"]["test_dataset"])

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    def test_pick_up_results_all_blacklisted_skips_dataset(self, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test that datasets with only blacklisted metrics are skipped"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "test_dataset.json")
        with open(result_file, 'w') as f:
            json.dump({
                "bp": 0.5,
                "details": "some details",
                "type": "test"
            }, f)

        mock_mmengine_load.return_value = {
            "bp": 0.5,
            "details": "some details",
            "type": "test"
        }

        config = ConfigDict({
            "models": [self.model_cfg],
            "datasets": [self.dataset_cfg],
            "work_dir": self.temp_dir,
            "path": "./test_path"
        })

        summarizer = HarborSummarizer(config)
        summarizer.model_cfgs = [self.model_cfg]
        summarizer.dataset_cfgs = [self.dataset_cfg]

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_dataset", raw_results["test_model"])
        self.assertNotIn("test_dataset", parsed_results["test_model"])
        self.assertNotIn("test_dataset", dataset_metrics)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    def test_pick_up_results_string_values(self, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test _pick_up_results handles string values"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "test_dataset.json")
        with open(result_file, 'w') as f:
            json.dump({"avg_score": "N/A", "status": "completed"}, f)

        mock_mmengine_load.return_value = {"avg_score": "N/A", "status": "completed"}

        config = ConfigDict({
            "models": [self.model_cfg],
            "datasets": [self.dataset_cfg],
            "work_dir": self.temp_dir,
            "path": "./test_path"
        })

        summarizer = HarborSummarizer(config)
        summarizer.model_cfgs = [self.model_cfg]
        summarizer.dataset_cfgs = [self.dataset_cfg]

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertEqual(parsed_results["test_model"]["test_dataset"]["avg_score"], "N/A")
        self.assertEqual(parsed_results["test_model"]["test_dataset"]["status"], "completed")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    def test_pick_up_results_preserves_pass_at_k(self, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test that pass_at_k is preserved in raw_results"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "test_dataset.json")
        with open(result_file, 'w') as f:
            json.dump({"avg_score": 0.75, "pass_at_k": {"pass@1": 0.75, "pass@5": 0.9}}, f)

        mock_mmengine_load.return_value = {"avg_score": 0.75, "pass_at_k": {"pass@1": 0.75, "pass@5": 0.9}}

        config = ConfigDict({
            "models": [self.model_cfg],
            "datasets": [self.dataset_cfg],
            "work_dir": self.temp_dir,
            "path": "./test_path"
        })

        summarizer = HarborSummarizer(config)
        summarizer.model_cfgs = [self.model_cfg]
        summarizer.dataset_cfgs = [self.dataset_cfg]

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("pass_at_k", raw_results["test_model"]["test_dataset"])
        self.assertEqual(raw_results["test_model"]["test_dataset"]["pass_at_k"], {"pass@1": 0.75, "pass@5": 0.9})

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    def test_pick_up_results_file_not_exists(self, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test _pick_up_results skips non-existent files"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        config = ConfigDict({
            "models": [self.model_cfg],
            "datasets": [self.dataset_cfg],
            "work_dir": self.temp_dir,
            "path": "./test_path"
        })

        summarizer = HarborSummarizer(config)
        summarizer.model_cfgs = [self.model_cfg]
        summarizer.dataset_cfgs = [self.dataset_cfg]

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertEqual(raw_results["test_model"], {})
        self.assertEqual(parsed_results["test_model"], {})

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    def test_pick_up_results_multiple_models(self, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test _pick_up_results with multiple models and datasets"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.side_effect = ["model1", "model2"]
        mock_dataset_abbr.side_effect = ["dataset1", "dataset2"]

        for model in ["model1", "model2"]:
            results_dir = os.path.join(self.temp_dir, "results", model)
            os.makedirs(results_dir, exist_ok=True)
            for dataset in ["dataset1", "dataset2"]:
                result_file = os.path.join(results_dir, f"{dataset}.json")
                with open(result_file, 'w') as f:
                    if model == "model1":
                        json.dump({"avg_score": 0.8, "accuracy": 0.9}, f)
                    else:
                        json.dump({"avg_score": 0.6, "accuracy": 0.7}, f)

        def load_side_effect(filepath):
            if "model1" in filepath:
                return {"avg_score": 0.8, "accuracy": 0.9}
            return {"avg_score": 0.6, "accuracy": 0.7}

        mock_mmengine_load.side_effect = load_side_effect

        config = ConfigDict({
            "models": [{"abbr": "model1"}, {"abbr": "model2"}],
            "datasets": [self.dataset_cfg, self.dataset_cfg],
            "work_dir": self.temp_dir,
            "path": "./test_path"
        })

        summarizer = HarborSummarizer(config)
        summarizer.model_cfgs = [{"abbr": "model1"}, {"abbr": "model2"}]
        summarizer.dataset_cfgs = [self.dataset_cfg, self.dataset_cfg]

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("model1", raw_results)
        self.assertIn("model2", raw_results)


class TestHarborSummarizerSummarize(unittest.TestCase):
    """Test summarize method"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        self.model_cfg = {
            "type": "TestModel",
            "abbr": "test_model",
            "path": "/the/path/to/test_path"
        }

        self.dataset_cfg = {
            "type": "TestDataset",
            "abbr": "test_dataset",
            "infer_cfg": {
                "inferencer": {"type": "GenInferencer"}
            }
        }

        self.config = ConfigDict({
            "models": [self.model_cfg],
            "datasets": [self.dataset_cfg],
            "work_dir": self.temp_dir,
            "path": "./test_path"
        })

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_creates_output_files(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test summarize creates summary files"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger
        mock_model_abbr.side_effect = ["test_model", "test_model"]
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "test_dataset.json")
        with open(result_file, 'w') as f:
            json.dump({"avg_score": 0.85, "total_count": 100}, f)

        summary_dir = os.path.join(self.temp_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        mock_file_handle = mock.MagicMock()
        mock_open.return_value = mock_file_handle
        mock_mmengine_load.return_value = {"avg_score": 0.85, "total_count": 100}

        summarizer = HarborSummarizer(self.config)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = ["test_dataset"]
        summarizer._update_dataset_abbrs = mock.MagicMock()

        result = summarizer.summarize(time_str="test_time")

        mock_mkdir.assert_called()

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_with_summary_groups(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test summarize with summary_groups"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger
        mock_model_abbr.side_effect = ["test_model", "test_model"]
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "test_dataset.json")
        with open(result_file, 'w') as f:
            json.dump({"avg_score": 0.8, "total_count": 100}, f)

        summary_dir = os.path.join(self.temp_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        mock_open.return_value = mock.MagicMock()
        mock_mmengine_load.return_value = {"avg_score": 0.8, "total_count": 100}

        summary_groups = [
            {"name": "group1", "subsets": ["test_dataset"], "version": "v1.0"},
            {"name": "group2", "subsets": ["test_dataset"], "metric": "avg_score"}
        ]

        summarizer = HarborSummarizer(self.config, summary_groups=summary_groups)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = ["test_dataset"]
        summarizer._update_dataset_abbrs = mock.MagicMock()

        result = summarizer.summarize(time_str="test_time")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_without_total_count(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test summarize without total_count in metrics"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger
        mock_model_abbr.side_effect = ["test_model", "test_model"]
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "test_dataset.json")
        with open(result_file, 'w') as f:
            json.dump({"avg_score": 0.8}, f)

        summary_dir = os.path.join(self.temp_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        mock_open.return_value = mock.MagicMock()
        mock_mmengine_load.return_value = {"avg_score": 0.8}

        summarizer = HarborSummarizer(self.config)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = ["test_dataset"]
        summarizer._update_dataset_abbrs = mock.MagicMock()

        result = summarizer.summarize(time_str="test_time")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_with_multiple_datasets(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test summarize with multiple datasets"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.side_effect = ["model1", "model1", "model1"]
        mock_dataset_abbr.side_effect = ["dataset1", "dataset2", "dataset1", "dataset2"]

        for dataset in ["dataset1", "dataset2"]:
            results_dir = os.path.join(self.temp_dir, "results", "model1")
            os.makedirs(results_dir, exist_ok=True)
            result_file = os.path.join(results_dir, f"{dataset}.json")
            if dataset == "dataset1":
                with open(result_file, 'w') as f:
                    json.dump({"avg_score": 0.9, "total_count": 100}, f)
            else:
                with open(result_file, 'w') as f:
                    json.dump({"accuracy": 0.85, "total_count": 50}, f)

        summary_dir = os.path.join(self.temp_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        def load_side_effect(filepath):
            if "dataset1" in filepath:
                return {"avg_score": 0.9, "total_count": 100}
            return {"accuracy": 0.85, "total_count": 50}

        mock_open.return_value = mock.MagicMock()
        mock_mmengine_load.side_effect = load_side_effect

        config = ConfigDict({
            "models": [self.model_cfg],
            "datasets": [self.dataset_cfg, self.dataset_cfg],
            "work_dir": self.temp_dir,
            "path": "./test_path"
        })

        summarizer = HarborSummarizer(config)
        summarizer.model_abbrs = ["model1"]
        summarizer.dataset_abbrs = ["dataset1", "dataset2"]
        summarizer._update_dataset_abbrs = mock.MagicMock()

        result = summarizer.summarize(time_str="test_time")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_with_correct_count_and_total_count(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test summarize filters out correct_count and total_count metrics"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger
        mock_model_abbr.side_effect = ["test_model", "test_model"]
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "test_dataset.json")
        with open(result_file, 'w') as f:
            json.dump({"avg_score": 0.8, "correct_count": 80, "total_count": 100}, f)

        summary_dir = os.path.join(self.temp_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        mock_open.return_value = mock.MagicMock()
        mock_mmengine_load.return_value = {"avg_score": 0.8, "correct_count": 80, "total_count": 100}

        summarizer = HarborSummarizer(self.config)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = ["test_dataset"]
        summarizer._update_dataset_abbrs = mock.MagicMock()

        result = summarizer.summarize(time_str="test_time")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_with_dataset_not_in_metrics(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test summarize handles datasets not in metrics"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger
        mock_model_abbr.side_effect = ["test_model", "test_model"]
        mock_dataset_abbr.side_effect = ["dataset1", "dataset2"]

        summary_dir = os.path.join(self.temp_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        mock_open.return_value = mock.MagicMock()

        config = ConfigDict({
            "models": [self.model_cfg],
            "datasets": [self.dataset_cfg, self.dataset_cfg],
            "work_dir": self.temp_dir,
            "path": "./test_path"
        })

        summarizer = HarborSummarizer(config)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = ["dataset1", "dataset2"]
        summarizer._update_dataset_abbrs = mock.MagicMock()

        with mock.patch.object(summarizer, '_pick_up_results', return_value=({}, {}, {}, {})):
            result = summarizer.summarize(time_str="test_time")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_with_multiple_models(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test summarize with multiple models"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.side_effect = ["model1", "model2", "model1", "model2", "model1", "model2"]
        mock_dataset_abbr.return_value = "dataset1"

        for model in ["model1", "model2"]:
            results_dir = os.path.join(self.temp_dir, "results", model)
            os.makedirs(results_dir, exist_ok=True)
            result_file = os.path.join(results_dir, "dataset1.json")
            if model == "model1":
                with open(result_file, 'w') as f:
                    json.dump({"avg_score": 0.9}, f)
            else:
                with open(result_file, 'w') as f:
                    json.dump({"avg_score": 0.7}, f)

        summary_dir = os.path.join(self.temp_dir, "summary")
        os.makedirs(summary_dir, exist_ok=True)

        def load_side_effect(filepath):
            if "model1" in filepath:
                return {"avg_score": 0.9}
            return {"avg_score": 0.7}

        mock_open.return_value = mock.MagicMock()
        mock_mmengine_load.side_effect = load_side_effect

        config = ConfigDict({
            "models": [{"abbr": "model1"}, {"abbr": "model2"}],
            "datasets": [self.dataset_cfg],
            "work_dir": self.temp_dir,
            "path": "./test_path"
        })

        summarizer = HarborSummarizer(config)
        summarizer.model_abbrs = ["model1", "model2"]
        summarizer.dataset_abbrs = ["dataset1"]
        summarizer._update_dataset_abbrs = mock.MagicMock()

        result = summarizer.summarize(time_str="test_time")


if __name__ == '__main__':
    unittest.main()