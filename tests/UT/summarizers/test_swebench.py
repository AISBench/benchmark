import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from ais_bench.benchmark.summarizers.swebench import SWEBenchSummarizer
from mmengine import ConfigDict


class TestSWEBenchSummarizer(unittest.TestCase):

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

    def _create_summarizer(self, model_cfgs=None, dataset_cfgs=None):
        config = ConfigDict({
            'models': model_cfgs or [self.model_cfg],
            'datasets': dataset_cfgs or [self.dataset_cfg],
            'work_dir': self.temp_dir
        })
        return SWEBenchSummarizer(config=config)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('mmengine.load')
    @patch('ais_bench.benchmark.summarizers.swebench.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.swebench.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.swebench.get_infer_output_path')
    def test_pick_up_results_valid_aggregate(self, mock_get_path, mock_dataset_abbr,
                                              mock_model_abbr, mock_load,
                                              mock_isfile, mock_isdir):
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_get_path.return_value = "aggregate.json"
        mock_load.return_value = {
            "total_instances": 100,
            "resolved_instances": 80
        }

        summarizer = self._create_summarizer()
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_model", raw_results)
        self.assertIn("test_dataset", raw_results["test_model"])
        self.assertEqual(raw_results["test_model"]["test_dataset"]["accuracy"], 80.0)
        self.assertEqual(raw_results["test_model"]["test_dataset"]["correct_count"], 80)
        self.assertEqual(raw_results["test_model"]["test_dataset"]["total_count"], 100)

        self.assertIn("test_model", parsed_results)
        self.assertIn("test_dataset", parsed_results["test_model"])
        self.assertEqual(parsed_results["test_model"]["test_dataset"]["accuracy"], 80.0)

        self.assertIn("test_dataset", dataset_metrics)
        self.assertEqual(dataset_metrics["test_dataset"], ["accuracy"])

        self.assertIn("test_dataset", dataset_eval_mode)
        self.assertEqual(dataset_eval_mode["test_dataset"], "agent")

    @patch('os.path.isdir')
    @patch('ais_bench.benchmark.summarizers.swebench.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.swebench.dataset_abbr_from_cfg')
    def test_pick_up_results_no_dir(self, mock_dataset_abbr, mock_model_abbr, mock_isdir):
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"
        mock_isdir.return_value = False

        summarizer = self._create_summarizer()
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_model", raw_results)
        self.assertEqual(len(raw_results["test_model"]), 0)
        self.assertIn("test_model", parsed_results)
        self.assertEqual(len(parsed_results["test_model"]), 0)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('ais_bench.benchmark.summarizers.swebench.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.swebench.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.swebench.get_infer_output_path')
    def test_pick_up_results_no_aggregate_file(self, mock_get_path, mock_dataset_abbr,
                                                mock_model_abbr, mock_isfile, mock_isdir):
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"
        mock_isdir.return_value = True
        mock_isfile.return_value = False
        mock_get_path.return_value = "aggregate.json"

        summarizer = self._create_summarizer()
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_model", raw_results)
        self.assertEqual(len(raw_results["test_model"]), 0)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('mmengine.load')
    @patch('ais_bench.benchmark.summarizers.swebench.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.swebench.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.swebench.get_infer_output_path')
    def test_pick_up_results_zero_total_instances(self, mock_get_path, mock_dataset_abbr,
                                                   mock_model_abbr, mock_load,
                                                   mock_isfile, mock_isdir):
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_get_path.return_value = "aggregate.json"
        mock_load.return_value = {
            "total_instances": 0,
            "resolved_instances": 0
        }

        summarizer = self._create_summarizer()
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_model", raw_results)
        self.assertEqual(raw_results["test_model"]["test_dataset"]["accuracy"], 0.0)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('mmengine.load')
    @patch('ais_bench.benchmark.summarizers.swebench.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.swebench.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.swebench.get_infer_output_path')
    def test_pick_up_results_invalid_data(self, mock_get_path, mock_dataset_abbr,
                                           mock_model_abbr, mock_load,
                                           mock_isfile, mock_isdir):
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_get_path.return_value = "aggregate.json"
        mock_load.return_value = "invalid_data"

        summarizer = self._create_summarizer()
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_model", raw_results)
        self.assertEqual(len(raw_results["test_model"]), 0)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('mmengine.load')
    @patch('ais_bench.benchmark.summarizers.swebench.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.swebench.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.swebench.get_infer_output_path')
    def test_pick_up_results_missing_fields(self, mock_get_path, mock_dataset_abbr,
                                             mock_model_abbr, mock_load,
                                             mock_isfile, mock_isdir):
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_get_path.return_value = "aggregate.json"
        mock_load.return_value = {
            "total_instances": 100
        }

        summarizer = self._create_summarizer()
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_model", raw_results)
        self.assertEqual(len(raw_results["test_model"]), 0)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('mmengine.load')
    @patch('ais_bench.benchmark.summarizers.swebench.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.swebench.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.swebench.get_infer_output_path')
    def test_pick_up_results_negative_total(self, mock_get_path, mock_dataset_abbr,
                                             mock_model_abbr, mock_load,
                                             mock_isfile, mock_isdir):
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_get_path.return_value = "aggregate.json"
        mock_load.return_value = {
            "total_instances": -1,
            "resolved_instances": 0
        }

        summarizer = self._create_summarizer()
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_model", raw_results)
        self.assertEqual(len(raw_results["test_model"]), 0)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('mmengine.load')
    @patch('ais_bench.benchmark.summarizers.swebench.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.swebench.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.swebench.get_infer_output_path')
    def test_pick_up_results_load_exception(self, mock_get_path, mock_dataset_abbr,
                                             mock_model_abbr, mock_load,
                                             mock_isfile, mock_isdir):
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_get_path.return_value = "aggregate.json"
        mock_load.side_effect = Exception("Load error")

        summarizer = self._create_summarizer()
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_model", raw_results)
        self.assertEqual(len(raw_results["test_model"]), 0)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('mmengine.load')
    @patch('ais_bench.benchmark.summarizers.swebench.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.swebench.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.swebench.get_infer_output_path')
    def test_pick_up_results_100_percent(self, mock_get_path, mock_dataset_abbr,
                                         mock_model_abbr, mock_load,
                                         mock_isfile, mock_isdir):
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_get_path.return_value = "aggregate.json"
        mock_load.return_value = {
            "total_instances": 50,
            "resolved_instances": 50
        }

        summarizer = self._create_summarizer()
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertEqual(raw_results["test_model"]["test_dataset"]["accuracy"], 100.0)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('mmengine.load')
    @patch('ais_bench.benchmark.summarizers.swebench.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.swebench.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.swebench.get_infer_output_path')
    def test_pick_up_results_string_total_instances(self, mock_get_path, mock_dataset_abbr,
                                                     mock_model_abbr, mock_load,
                                                     mock_isfile, mock_isdir):
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_get_path.return_value = "aggregate.json"
        mock_load.return_value = {
            "total_instances": "100",
            "resolved_instances": 80
        }

        summarizer = self._create_summarizer()
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_model", raw_results)
        self.assertEqual(len(raw_results["test_model"]), 0)

    def test_pick_up_results_empty_configs(self):
        config = ConfigDict({
            'models': [],
            'datasets': [],
            'work_dir': self.temp_dir
        })
        summarizer = SWEBenchSummarizer(config=config)

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertEqual(len(raw_results), 0)
        self.assertEqual(len(parsed_results), 0)
        self.assertEqual(len(dataset_metrics), 0)
        self.assertEqual(len(dataset_eval_mode), 0)

    @patch('os.path.isdir')
    @patch('os.path.isfile')
    @patch('mmengine.load')
    @patch('ais_bench.benchmark.summarizers.swebench.model_abbr_from_cfg_used_in_summarizer')
    @patch('ais_bench.benchmark.summarizers.swebench.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.summarizers.swebench.get_infer_output_path')
    def test_pick_up_results_resolved_greater_than_total(self, mock_get_path, mock_dataset_abbr,
                                                          mock_model_abbr, mock_load,
                                                          mock_isfile, mock_isdir):
        mock_model_abbr.return_value = "test_model"
        mock_dataset_abbr.return_value = "test_dataset"
        mock_isdir.return_value = True
        mock_isfile.return_value = True
        mock_get_path.return_value = "aggregate.json"
        mock_load.return_value = {
            "total_instances": 100,
            "resolved_instances": 150
        }

        summarizer = self._create_summarizer()
        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = summarizer._pick_up_results()

        self.assertIn("test_model", raw_results)
        self.assertEqual(raw_results["test_model"]["test_dataset"]["accuracy"], 150.0)


if __name__ == '__main__':
    unittest.main()