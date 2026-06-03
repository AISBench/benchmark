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
        self.model_cfg = {"type": "TestModel", "abbr": "test_model", "path": "/the/path/to/test_path"}
        self.model_cfg2 = {"type": "TestModel2", "abbr": "test_model2", "summarizer_abbr": "custom_abbr"}
        self.dataset_cfg = {"type": "TestDataset", "abbr": "test_dataset", "infer_cfg": {"inferencer": {"type": "GenInferencer"}}}
        self.dataset_cfg2 = {"type": "TestDataset2", "abbr": "test_dataset2", "infer_cfg": {"inferencer": {"type": "PPLInferencer"}}}
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
        mock_model_abbr.return_value = "test_model"
        mock_ais_logger.return_value = mock.MagicMock()
        summarizer = HarborSummarizer(self.config)
        self.assertEqual(summarizer.cfg, self.config)
        self.assertEqual(summarizer.work_dir, self.temp_dir)

    def test_metric_whitelist_content(self):
        for metric in ['avg_score', 'score', 'accuracy', 'n_errors', 'n_total_trials']:
            self.assertIn(metric, METRIC_WHITELIST)

    def test_metric_blacklist_content(self):
        for metric in ['bp', 'sys_len', 'ref_len', 'type', 'reward_distribution', 'exception_distribution', 'pass_at_k', 'details']:
            self.assertIn(metric, METRIC_BLACKLIST)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    def test_inheritance(self, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.return_value = "test_model"
        from ais_bench.benchmark.summarizers.default import DefaultSummarizer
        summarizer = HarborSummarizer(self.config)
        self.assertIsInstance(summarizer, DefaultSummarizer)

    @mock.patch('builtins.print')
    def test_print_harbor_details_with_distributions(self, mock_print):
        raw_results = {
            "model1": {
                "dataset1": {
                    "total_count": 100, "n_errors": 5, "avg_score": 0.75,
                    "reward_distribution": [{"score": 1.0, "count": 50}, {"score": 0.5, "count": 30}],
                    "exception_distribution": [{"exception_type": "AgentTimeoutError", "count": 3}],
                    "pass_at_k": {"pass@1": 0.5, "pass@5": 0.8}
                }
            }
        }
        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = ["model1"]
        summarizer._print_harbor_details(raw_results)
        self.assertTrue(mock_print.called)

    @mock.patch('builtins.print')
    def test_print_harbor_details_empty(self, mock_print):
        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = []
        summarizer._print_harbor_details({})

    @mock.patch('builtins.print')
    def test_print_harbor_details_no_distributions(self, mock_print):
        raw_results = {"model1": {"dataset1": {"total_count": 10, "avg_score": 0.8}}}
        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = ["model1"]
        summarizer._print_harbor_details(raw_results)

    @mock.patch('builtins.print')
    def test_print_harbor_details_empty_distributions(self, mock_print):
        raw_results = {"model1": {"dataset1": {"total_count": 10, "avg_score": 0.8, "reward_distribution": [], "exception_distribution": []}}}
        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = ["model1"]
        summarizer._print_harbor_details(raw_results)


class TestHarborSummarizerPickUpResults(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_cfg = {"type": "TestModel", "abbr": "test_model", "path": "/the/path/to/test_path"}
        self.dataset_cfg = {"type": "TestDataset", "abbr": "test_dataset", "infer_cfg": {"inferencer": {"type": "GenInferencer"}}}

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    def test_pick_up_results_basic(self, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "test_dataset.json"), 'w') as f:
            json.dump({"avg_score": 0.85, "accuracy": 0.9}, f)

        mock_mmengine_load.return_value = {"avg_score": 0.85, "accuracy": 0.9}

        config = ConfigDict({"models": [self.model_cfg], "datasets": [self.dataset_cfg], "work_dir": self.temp_dir, "path": "./test_path"})
        summarizer = HarborSummarizer(config)
        summarizer.model_cfgs = [self.model_cfg]
        summarizer.dataset_cfgs = [self.dataset_cfg]

        raw_results, parsed_results, _, _ = summarizer._pick_up_results()
        self.assertIn("test_model", raw_results)
        self.assertEqual(raw_results["test_model"]["test_dataset"]["avg_score"], 0.85)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    def test_pick_up_results_skips_blacklisted(self, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "test_dataset.json"), 'w') as f:
            json.dump({"avg_score": 0.85, "bp": 0.5, "details": "some details"}, f)

        mock_mmengine_load.return_value = {"avg_score": 0.85, "bp": 0.5, "details": "some details"}

        config = ConfigDict({"models": [self.model_cfg], "datasets": [self.dataset_cfg], "work_dir": self.temp_dir, "path": "./test_path"})
        summarizer = HarborSummarizer(config)
        summarizer.model_cfgs = [self.model_cfg]
        summarizer.dataset_cfgs = [self.dataset_cfg]

        _, parsed_results, _, _ = summarizer._pick_up_results()
        self.assertNotIn("bp", parsed_results["test_model"]["test_dataset"])
        self.assertIn("avg_score", parsed_results["test_model"]["test_dataset"])

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    def test_pick_up_results_all_blacklisted_skips_dataset(self, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "test_dataset.json"), 'w') as f:
            json.dump({"bp": 0.5, "details": "some details", "type": "test"}, f)

        mock_mmengine_load.return_value = {"bp": 0.5, "details": "some details", "type": "test"}

        config = ConfigDict({"models": [self.model_cfg], "datasets": [self.dataset_cfg], "work_dir": self.temp_dir, "path": "./test_path"})
        summarizer = HarborSummarizer(config)
        summarizer.model_cfgs = [self.model_cfg]
        summarizer.dataset_cfgs = [self.dataset_cfg]

        _, parsed_results, dataset_metrics, _ = summarizer._pick_up_results()
        self.assertNotIn("test_dataset", parsed_results["test_model"])
        self.assertNotIn("test_dataset", dataset_metrics)

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    def test_pick_up_results_string_values(self, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "test_dataset.json"), 'w') as f:
            json.dump({"avg_score": "N/A", "status": "completed"}, f)

        mock_mmengine_load.return_value = {"avg_score": "N/A", "status": "completed"}

        config = ConfigDict({"models": [self.model_cfg], "datasets": [self.dataset_cfg], "work_dir": self.temp_dir, "path": "./test_path"})
        summarizer = HarborSummarizer(config)
        summarizer.model_cfgs = [self.model_cfg]
        summarizer.dataset_cfgs = [self.dataset_cfg]

        _, parsed_results, _, _ = summarizer._pick_up_results()
        self.assertEqual(parsed_results["test_model"]["test_dataset"]["avg_score"], "N/A")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    def test_pick_up_results_preserves_blacklisted(self, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "test_dataset.json"), 'w') as f:
            json.dump({"avg_score": 0.75, "pass_at_k": {"pass@1": 0.75, "pass@5": 0.9}}, f)

        mock_mmengine_load.return_value = {"avg_score": 0.75, "pass_at_k": {"pass@1": 0.75, "pass@5": 0.9}}

        config = ConfigDict({"models": [self.model_cfg], "datasets": [self.dataset_cfg], "work_dir": self.temp_dir, "path": "./test_path"})
        summarizer = HarborSummarizer(config)
        summarizer.model_cfgs = [self.model_cfg]
        summarizer.dataset_cfgs = [self.dataset_cfg]

        raw_results, _, _, _ = summarizer._pick_up_results()
        self.assertIn("pass_at_k", raw_results["test_model"]["test_dataset"])

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    def test_pick_up_results_file_not_exists(self, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"

        config = ConfigDict({"models": [self.model_cfg], "datasets": [self.dataset_cfg], "work_dir": self.temp_dir, "path": "./test_path"})
        summarizer = HarborSummarizer(config)
        summarizer.model_cfgs = [self.model_cfg]
        summarizer.dataset_cfgs = [self.dataset_cfg]

        raw_results, parsed_results, _, _ = summarizer._pick_up_results()
        self.assertEqual(raw_results["test_model"], {})


class TestHarborSummarizerSummarize(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model_cfg = {"type": "TestModel", "abbr": "test_model", "path": "/the/path/to/test_path"}
        self.dataset_cfg = {"type": "TestDataset", "abbr": "test_dataset", "infer_cfg": {"inferencer": {"type": "GenInferencer"}}}
        self.config = ConfigDict({"models": [self.model_cfg], "datasets": [self.dataset_cfg], "work_dir": self.temp_dir, "path": "./test_path"})

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
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.side_effect = ["test_model", "test_model"]
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "test_dataset.json"), 'w') as f:
            json.dump({"avg_score": 0.85, "total_count": 100}, f)

        mock_open.return_value = mock.MagicMock()
        mock_mmengine_load.return_value = {"avg_score": 0.85, "total_count": 100}

        summarizer = HarborSummarizer(self.config)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = ["test_dataset"]
        summarizer._update_dataset_abbrs = mock.MagicMock()
        summarizer.summarize(time_str="test_time")
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
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.side_effect = ["test_model", "test_model"]
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "test_dataset.json"), 'w') as f:
            json.dump({"avg_score": 0.8, "total_count": 100}, f)

        mock_open.return_value = mock.MagicMock()
        mock_mmengine_load.return_value = {"avg_score": 0.8, "total_count": 100}

        summary_groups = [{"name": "group1", "subsets": ["test_dataset"], "version": "v1.0"}]
        summarizer = HarborSummarizer(self.config, summary_groups=summary_groups)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = ["test_dataset"]
        summarizer._update_dataset_abbrs = mock.MagicMock()
        summarizer.summarize(time_str="test_time")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_without_total_count(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.side_effect = ["test_model", "test_model"]
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "test_dataset.json"), 'w') as f:
            json.dump({"avg_score": 0.8}, f)

        mock_open.return_value = mock.MagicMock()
        mock_mmengine_load.return_value = {"avg_score": 0.8}

        summarizer = HarborSummarizer(self.config)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = ["test_dataset"]
        summarizer._update_dataset_abbrs = mock.MagicMock()
        summarizer.summarize(time_str="test_time")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_filters_correct_count(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.side_effect = ["test_model", "test_model"]
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "test_dataset.json"), 'w') as f:
            json.dump({"avg_score": 0.8, "correct_count": 80, "total_count": 100}, f)

        mock_open.return_value = mock.MagicMock()
        mock_mmengine_load.return_value = {"avg_score": 0.8, "correct_count": 80, "total_count": 100}

        summarizer = HarborSummarizer(self.config)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = ["test_dataset"]
        summarizer._update_dataset_abbrs = mock.MagicMock()
        summarizer.summarize(time_str="test_time")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_dataset_not_in_metrics(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.side_effect = ["test_model", "test_model"]
        mock_dataset_abbr.side_effect = ["dataset1", "dataset2"]

        os.makedirs(os.path.join(self.temp_dir, "summary"), exist_ok=True)
        mock_open.return_value = mock.MagicMock()

        config = ConfigDict({"models": [self.model_cfg], "datasets": [self.dataset_cfg, self.dataset_cfg], "work_dir": self.temp_dir, "path": "./test_path"})
        summarizer = HarborSummarizer(config)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = ["dataset1", "dataset2"]
        summarizer._update_dataset_abbrs = mock.MagicMock()

        with mock.patch.object(summarizer, '_pick_up_results', return_value=({}, {}, {}, {})):
            summarizer.summarize(time_str="test_time")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_with_multiple_models(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.side_effect = ["model1", "model2", "model1", "model2", "model1", "model2"]
        mock_dataset_abbr.return_value = "dataset1"

        for model in ["model1", "model2"]:
            results_dir = os.path.join(self.temp_dir, "results", model)
            os.makedirs(results_dir, exist_ok=True)
            score = 0.9 if model == "model1" else 0.7
            with open(os.path.join(results_dir, "dataset1.json"), 'w') as f:
                json.dump({"avg_score": score}, f)

        def load_side_effect(filepath):
            return {"avg_score": 0.9} if "model1" in filepath else {"avg_score": 0.7}

        mock_open.return_value = mock.MagicMock()
        mock_mmengine_load.side_effect = load_side_effect

        config = ConfigDict({"models": [{"abbr": "model1"}, {"abbr": "model2"}], "datasets": [self.dataset_cfg], "work_dir": self.temp_dir, "path": "./test_path"})
        summarizer = HarborSummarizer(config)
        summarizer.model_abbrs = ["model1", "model2"]
        summarizer.dataset_abbrs = ["dataset1"]
        summarizer._update_dataset_abbrs = mock.MagicMock()
        summarizer.summarize(time_str="test_time")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_metric_none_branch(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.side_effect = ["test_model"]
        mock_dataset_abbr.return_value = "test_dataset"

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "test_dataset.json"), 'w') as f:
            json.dump({"score": 0.9, "accuracy": 0.85}, f)

        mock_open.return_value = mock.MagicMock()
        mock_mmengine_load.return_value = {"score": 0.9, "accuracy": 0.85}

        config = ConfigDict({"models": [self.model_cfg], "datasets": [self.dataset_cfg], "work_dir": self.temp_dir, "path": "./test_path"})
        summarizer = HarborSummarizer(config)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = ["test_dataset"]
        summarizer._update_dataset_abbrs = mock.MagicMock()
        summarizer.summarize(time_str="test_time")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_with_multiple_summary_groups(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.side_effect = ["model1", "model2", "model1", "model2"]
        mock_dataset_abbr.side_effect = ["dataset1", "dataset2", "dataset1", "dataset2"]

        for model in ["model1", "model2"]:
            for dataset in ["dataset1", "dataset2"]:
                results_dir = os.path.join(self.temp_dir, "results", model)
                os.makedirs(results_dir, exist_ok=True)
                with open(os.path.join(results_dir, f"{dataset}.json"), 'w') as f:
                    json.dump({"avg_score": 0.85, "accuracy": 0.9}, f)

        mock_open.return_value = mock.MagicMock()
        mock_mmengine_load.return_value = {"avg_score": 0.85, "accuracy": 0.9}

        config = ConfigDict({"models": [{"abbr": "model1"}, {"abbr": "model2"}], "datasets": [self.dataset_cfg, self.dataset_cfg], "work_dir": self.temp_dir, "path": "./test_path"})
        summary_groups = [{"name": "group1", "subsets": ["dataset1", "dataset2"], "version": "v1.0"}, {"name": "group2", "subsets": ["dataset1"], "metric": "avg_score"}]
        summarizer = HarborSummarizer(config, summary_groups=summary_groups)
        summarizer.model_abbrs = ["model1", "model2"]
        summarizer.dataset_abbrs = ["dataset1", "dataset2"]
        summarizer._update_dataset_abbrs = mock.MagicMock()
        summarizer.summarize(time_str="test_time")

    @mock.patch('ais_bench.benchmark.summarizers.harbor.AISLogger')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.model_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.dataset_abbr_from_cfg')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.load')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.mmengine.mkdir_or_exist')
    @mock.patch('ais_bench.benchmark.summarizers.harbor.HarborSummarizer._print_harbor_details')
    @mock.patch('builtins.print')
    @mock.patch('builtins.open', create=True)
    def test_summarize_with_total_count_column(self, mock_open, mock_print, mock_print_details, mock_mkdir, mock_mmengine_load, mock_dataset_abbr, mock_model_abbr, mock_ais_logger):
        mock_ais_logger.return_value = mock.MagicMock()
        mock_model_abbr.side_effect = ["test_model"]
        mock_dataset_abbr.side_effect = ["test_dataset"]

        results_dir = os.path.join(self.temp_dir, "results", "test_model")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "test_dataset.json"), 'w') as f:
            json.dump({"avg_score": 0.85, "total_count": 50, "n_errors": 2}, f)

        mock_open.return_value = mock.MagicMock()
        mock_mmengine_load.return_value = {"avg_score": 0.85, "total_count": 50, "n_errors": 2}

        config = ConfigDict({"models": [self.model_cfg], "datasets": [self.dataset_cfg], "work_dir": self.temp_dir, "path": "./test_path"})
        summarizer = HarborSummarizer(config)
        summarizer.model_abbrs = ["test_model"]
        summarizer.dataset_abbrs = ["test_dataset"]
        summarizer._update_dataset_abbrs = mock.MagicMock()
        summarizer.summarize(time_str="test_time")

    @mock.patch('builtins.print')
    def test_print_harbor_details_shows_total_count(self, mock_print):
        raw_results = {"model1": {"dataset1": {"total_count": 100, "avg_score": 0.85, "reward_distribution": [{"score": 1.0, "count": 50}]}}}
        summarizer = HarborSummarizer.__new__(HarborSummarizer)
        summarizer.model_abbrs = ["model1"]
        summarizer._print_harbor_details(raw_results)
        self.assertTrue(mock_print.called)
        self.assertIn('Total Count:', str(mock_print.call_args_list))


if __name__ == '__main__':
    unittest.main()