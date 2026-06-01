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

        self.summary_groups = []

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
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('os.path.exists')
    @mock.patch('mmengine.load')
    def test_pick_up_results_normal(self, mock_mmengine_load, mock_exists, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test _pick_up_results with normal results"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.side_effect = ["test_model", "custom_abbr"]
        mock_dataset_abbr.side_effect = ["test_dataset", "test_dataset2"]

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)

        mock_exists.return_value = True
        mock_result = {
            "avg_score": 0.85,
            "accuracy": 0.9,
            "n_errors": 1,
            "n_total_trials": 10,
            "bp": 0.5,
            "reward_distribution": [{"score": 1.0, "count": 5}],
            "exception_distribution": [{"exception_type": "Timeout", "count": 1}],
            "details": "some details"
        }
        mock_mmengine_load.return_value = mock_result

        summarizer = HarborSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_model", raw_results)
        self.assertIn("test_dataset", raw_results["test_model"])
        self.assertNotIn("bp", raw_results["test_model"]["test_dataset"])
        self.assertNotIn("details", raw_results["test_model"]["test_dataset"])

        self.assertIn("test_dataset", parsed_results["test_model"])
        self.assertIn("avg_score", parsed_results["test_model"]["test_dataset"])
        self.assertEqual(parsed_results["test_model"]["test_dataset"]["avg_score"], 0.85)

        self.assertIn("test_dataset", dataset_metrics)
        self.assertIn("avg_score", dataset_metrics["test_dataset"])

        self.assertEqual(dataset_eval_mode["test_dataset"], "gen")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('os.path.exists')
    def test_pick_up_results_file_not_exists(self, mock_exists, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test _pick_up_results when file doesn't exist"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.side_effect = ["test_model", "custom_abbr"]
        mock_dataset_abbr.side_effect = ["test_dataset", "test_dataset2"]

        mock_exists.return_value = False

        summarizer = HarborSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertEqual(raw_results, {"test_model": {}, "custom_abbr": {}})
        self.assertEqual(parsed_results, {"test_model": {}, "custom_abbr": {}})
        self.assertEqual(dataset_metrics, {})

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('os.path.exists')
    @mock.patch('mmengine.load')
    def test_pick_up_results_with_string_values(self, mock_mmengine_load, mock_exists, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test _pick_up_results with string metric values"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)

        mock_exists.return_value = True
        mock_result = {
            "avg_score": "N/A",
            "status": "completed"
        }
        mock_mmengine_load.return_value = mock_result

        summarizer = HarborSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_dataset", parsed_results["test_model"])
        self.assertEqual(parsed_results["test_model"]["test_dataset"]["avg_score"], "N/A")
        self.assertEqual(parsed_results["test_model"]["test_dataset"]["status"], "completed")

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

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('os.path.exists')
    @mock.patch('mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('mmengine.mkdir_or_exist')
    def test_summarize_normal(self, mock_mkdir, mock_print, mock_print_details, mock_mmengine_load, mock_exists, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test summarize with normal results"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.side_effect = ["test_model", "custom_abbr"]
        mock_dataset_abbr.side_effect = ["test_dataset", "test_dataset2"]

        for model in ["test_model", "custom_abbr"]:
            results_dir = os.path.join(self.temp_dir, "results", model)
            os.makedirs(results_dir, exist_ok=True)
            result_file = os.path.join(results_dir, "test_dataset.json")
            with open(result_file, 'w') as f:
                json.dump({
                    "avg_score": 0.8,
                    "n_errors": 2,
                    "total_count": 10
                }, f)

        mock_exists.return_value = True
        mock_mmengine_load.side_effect = lambda f: {
            os.path.join(self.temp_dir, "results", "test_model", "test_dataset.json"): {"avg_score": 0.8, "n_errors": 2, "total_count": 10},
            os.path.join(self.temp_dir, "results", "test_model", "test_dataset2.json"): {"avg_score": 0.6, "n_errors": 1, "total_count": 5},
            os.path.join(self.temp_dir, "results", "custom_abbr", "test_dataset.json"): {"avg_score": 0.7, "n_errors": 0, "total_count": 10},
            os.path.join(self.temp_dir, "results", "custom_abbr", "test_dataset2.json"): {"avg_score": 0.5, "n_errors": 3, "total_count": 5}
        }.get(f, {})

        summarizer = HarborSummarizer(self.config)
        summarizer.dataset_abbrs = ["test_dataset", "test_dataset2"]
        summarizer.model_abbrs = ["test_model", "custom_abbr"]

        result = summarizer.summarize()

        self.assertIsNotNone(result)
        self.assertIn("test_model", result)
        self.assertIn("custom_abbr", result)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    def test_summarize_with_time_str(self, mock_model_abbr, mock_ais_logger):
        """Test summarize with custom time string"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.return_value = "test_model"

        summarizer = HarborSummarizer(self.config)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = []
        summarizer._update_dataset_abbrs = mock.MagicMock()

        with mock.patch.object(summarizer, '_pick_up_results', return_value=({}, {}, {}, {})), \
             mock.patch.object(summarizer, '_print_harbor_details'), \
             mock.patch('mmengine.mkdir_or_exist'), \
             mock.patch('builtins.print'), \
             mock.patch('builtins.open', mock.mock_open()):
            result = summarizer.summarize(time_str="20230101_120000")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('os.path.exists')
    @mock.patch('mmengine.load')
    def test_summarize_with_summary_groups(self, mock_mmengine_load, mock_exists, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test summarize with summary groups"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "test_dataset.json")
        with open(result_file, 'w') as f:
            json.dump({"avg_score": 0.8}, f)

        mock_exists.return_value = True
        mock_mmengine_load.return_value = {"avg_score": 0.8}

        config_with_groups = ConfigDict({
            "models": [self.model_cfg],
            "datasets": [self.dataset_cfg],
            "work_dir": self.temp_dir,
            "path": "./test_path"
        })

        summary_groups = [
            {"name": "group1", "subsets": ["test_dataset"]},
            {"name": "group2", "subsets": ["test_dataset"], "version": "v1.0"}
        ]

        summarizer = HarborSummarizer(config_with_groups, summary_groups=summary_groups)
        summarizer.dataset_abbrs = ["test_dataset"]
        summarizer.model_abbrs = ["test_model"]

        with mock.patch.object(summarizer, '_print_harbor_details'), \
             mock.patch('mmengine.mkdir_or_exist'), \
             mock.patch('builtins.print'), \
             mock.patch('builtins.open', mock.mock_open()):
            result = summarizer.summarize()

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('os.path.exists')
    def test_pick_up_results_multiple_models(self, mock_exists, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test _pick_up_results with multiple models"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.side_effect = ["model1", "model2", "model3"]
        mock_dataset_abbr.side_effect = ["dataset1", "dataset2"]

        for model in ["model1", "model2", "model3"]:
            for dataset in ["dataset1", "dataset2"]:
                results_dir = os.path.join(self.temp_dir, "results", model)
                os.makedirs(results_dir, exist_ok=True)

        mock_exists.side_effect = [True, True, True, True, False, False]

        summarizer = HarborSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("model1", raw_results)
        self.assertIn("model2", raw_results)
        self.assertIn("model3", raw_results)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('os.path.exists')
    @mock.patch('mmengine.load')
    def test_pick_up_results_with_pass_at_k(self, mock_mmengine_load, mock_exists, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test _pick_up_results includes pass_at_k in raw results"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)

        mock_exists.return_value = True
        mock_result = {
            "avg_score": 0.75,
            "pass_at_k": {"pass@1": 0.75, "pass@5": 0.9}
        }
        mock_mmengine_load.return_value = mock_result

        summarizer = HarborSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("pass_at_k", raw_results["test_model"]["test_dataset"])
        self.assertEqual(raw_results["test_model"]["test_dataset"]["pass_at_k"], {"pass@1": 0.75, "pass@5": 0.9})

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('os.path.exists')
    @mock.patch('mmengine.load')
    def test_pick_up_results_metric_ordering(self, mock_mmengine_load, mock_exists, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test _pick_up_results orders metrics according to whitelist"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)

        mock_exists.return_value = True
        mock_result = {
            "accuracy": 0.9,
            "avg_score": 0.85,
            "n_errors": 2,
            "score": 0.8
        }
        mock_mmengine_load.return_value = mock_result

        summarizer = HarborSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_dataset", dataset_metrics)
        ordered_metrics = dataset_metrics["test_dataset"]
        self.assertEqual(ordered_metrics[0], "avg_score")
        self.assertIn("accuracy", ordered_metrics)
        self.assertIn("score", ordered_metrics)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    def test_summarize_output_files(self, mock_model_abbr, mock_ais_logger):
        """Test summarize creates expected output files"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.return_value = "test_model"

        summarizer = HarborSummarizer(self.config)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = []
        summarizer._update_dataset_abbrs = mock.MagicMock()

        with mock.patch.object(summarizer, '_pick_up_results', return_value=({}, {}, {}, {})), \
             mock.patch.object(summarizer, '_print_harbor_details'), \
             mock.patch('mmengine.mkdir_or_exist') as mock_mkdir, \
             mock.patch('builtins.print') as mock_print, \
             mock.patch('builtins.open', mock.mock_open()) as mock_file:

            with mock.patch('mmengine.utils.TimeStub') as mock_time:
                mock_time_instance = mock.MagicMock()
                mock_time_instance.now.return_value.time_str = "20230101_120000"
                mock_time.now.return_value = mock_time_instance

                summarizer.summarize()

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('os.path.exists')
    @mock.patch('mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('mmengine.mkdir_or_exist')
    def test_summarize_with_n_total_trials(self, mock_mkdir, mock_print, mock_print_details, mock_mmengine_load, mock_exists, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test summarize includes n_total_trials in header when present"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.side_effect = ["test_model"]
        mock_dataset_abbr.side_effect = ["test_dataset"]

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, "test_dataset.json")
        with open(result_file, 'w') as f:
            json.dump({
                "avg_score": 0.8,
                "n_total_trials": 100,
                "total_count": 10
            }, f)

        mock_exists.return_value = True
        mock_mmengine_load.return_value = {"avg_score": 0.8, "n_total_trials": 100, "total_count": 10}

        summarizer = HarborSummarizer(self.config)
        summarizer.dataset_abbrs = ["test_dataset"]
        summarizer.model_abbrs = ["test_model"]

        result = summarizer.summarize()

        self.assertIn("test_model", result)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('os.path.exists')
    @mock.patch('mmengine.load')
    def test_pick_up_results_empty_metrics(self, mock_mmengine_load, mock_exists, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        """Test _pick_up_results with only blacklisted metrics"""
        mock_logger = mock.MagicMock()
        mock_ais_logger.return_value = mock_logger

        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)

        mock_exists.return_value = True
        mock_result = {
            "bp": 0.5,
            "details": "some details",
            "type": "test"
        }
        mock_mmengine_load.return_value = mock_result

        summarizer = HarborSummarizer(self.config)
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertNotIn("test_dataset", dataset_metrics)
        self.assertNotIn("test_dataset", parsed_results.get("test_model", {}))


if __name__ == '__main__':
    unittest.main()