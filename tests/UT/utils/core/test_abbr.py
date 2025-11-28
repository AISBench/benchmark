import unittest
from unittest.mock import patch
import os.path as osp

from mmengine.config import ConfigDict

from ais_bench.benchmark.utils.core.abbr import (
    model_abbr_from_cfg,
    dataset_abbr_from_cfg,
    task_abbr_from_cfg,
    get_infer_output_path,
    deal_with_judge_model_abbr,
)
from ais_bench.benchmark.utils.logging.exceptions import AISBenchRuntimeError


class TestAbbrUtilities(unittest.TestCase):
    """Test suite for abbreviation utilities"""

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    def test_model_abbr_from_cfg_with_abbr(self, mock_logger):
        """Test model_abbr_from_cfg when abbr is provided"""
        cfg = ConfigDict({"abbr": "my-model", "type": "GPT4"})
        result = model_abbr_from_cfg(cfg)
        
        self.assertEqual(result, "my-model")
        # Should not generate from path
        mock_logger.debug.assert_not_called()

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    @patch("os.path.realpath")
    def test_model_abbr_from_cfg_without_abbr(self, mock_realpath, mock_logger):
        """Test model_abbr_from_cfg generates abbr from path"""
        mock_realpath.return_value = "/home/models/gpt/v4"
        cfg = ConfigDict({"type": "OpenAI.GPT4", "path": "/models/gpt/v4"})
        
        result = model_abbr_from_cfg(cfg)
        
        self.assertEqual(result, "OpenAI.GPT4_gpt_v4")
        mock_logger.debug.assert_called_once()

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    def test_model_abbr_from_cfg_list(self, mock_logger):
        """Test model_abbr_from_cfg with list of configs"""
        cfgs = [
            ConfigDict({"abbr": "model-a"}),
            ConfigDict({"abbr": "model-b"}),
        ]
        
        result = model_abbr_from_cfg(cfgs)
        
        self.assertEqual(result, "model-a_model-b")

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    def test_dataset_abbr_from_cfg_with_abbr(self, mock_logger):
        """Test dataset_abbr_from_cfg when abbr is provided"""
        cfg = ConfigDict({"abbr": "gsm8k", "path": "datasets/gsm8k"})
        
        result = dataset_abbr_from_cfg(cfg)
        
        self.assertEqual(result, "gsm8k")
        mock_logger.debug.assert_not_called()

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    def test_dataset_abbr_from_cfg_without_abbr(self, mock_logger):
        """Test dataset_abbr_from_cfg generates abbr from path"""
        cfg = ConfigDict({"path": "datasets/math/problems"})
        
        result = dataset_abbr_from_cfg(cfg)
        
        self.assertEqual(result, "datasets_math_problems")
        mock_logger.debug.assert_called_once()

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    def test_dataset_abbr_from_cfg_with_name(self, mock_logger):
        """Test dataset_abbr_from_cfg with name field"""
        cfg = ConfigDict({"path": "datasets/gsm8k", "name": "train"})
        
        result = dataset_abbr_from_cfg(cfg)
        
        self.assertEqual(result, "datasets_gsm8k_train")

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    def test_task_abbr_from_cfg_single_dataset(self, mock_logger):
        """Test task_abbr_from_cfg with single dataset"""
        task = {
            "models": [{"abbr": "gpt-4"}],
            "datasets": [[{"abbr": "gsm8k"}]]
        }
        
        result = task_abbr_from_cfg(task)
        
        self.assertEqual(result, "gpt-4/gsm8k")
        mock_logger.debug.assert_called_once()

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    def test_task_abbr_from_cfg_merged_datasets(self, mock_logger):
        """Test task_abbr_from_cfg with merged datasets"""
        task = {
            "models": [{"abbr": "gpt-4"}],
            "datasets": [[
                {"abbr": "gsm8k", "type": "ais_bench.datasets.GSM8KDataset"},
                {"abbr": "math", "type": "ais_bench.datasets.MATHDataset"}
            ]]
        }
        
        result = task_abbr_from_cfg(task)
        
        self.assertEqual(result, "gpt-4/gsm8kdataset")

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    def test_get_infer_output_path_single_dataset(self, mock_logger):
        """Test get_infer_output_path with single dataset"""
        model_cfg = ConfigDict({"abbr": "gpt-4"})
        dataset_cfg = ConfigDict({"abbr": "gsm8k"})
        
        result = get_infer_output_path(model_cfg, dataset_cfg, "/outputs")
        
        expected = osp.join("/outputs", "gpt-4", "gsm8k.json")
        self.assertEqual(result, expected)
        mock_logger.debug.assert_called()

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    def test_get_infer_output_path_custom_extension(self, mock_logger):
        """Test get_infer_output_path with custom file extension"""
        model_cfg = ConfigDict({"abbr": "gpt-4"})
        dataset_cfg = ConfigDict({"abbr": "gsm8k"})
        
        result = get_infer_output_path(
            model_cfg, dataset_cfg, "/outputs", file_extension="jsonl"
        )
        
        expected = osp.join("/outputs", "gpt-4", "gsm8k.jsonl")
        self.assertEqual(result, expected)

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    def test_get_infer_output_path_no_root_path(self, mock_logger):
        """Test get_infer_output_path raises error without root_path"""
        model_cfg = ConfigDict({"abbr": "gpt-4"})
        dataset_cfg = ConfigDict({"abbr": "gsm8k"})
        
        with self.assertRaises(AISBenchRuntimeError):
            get_infer_output_path(model_cfg, dataset_cfg)
        
        mock_logger.warning.assert_not_called()

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    def test_deal_with_judge_model_abbr_new_judge(self, mock_logger):
        """Test deal_with_judge_model_abbr adds judge suffix"""
        model_cfg = ConfigDict({"abbr": "gpt-4"})
        judge_cfg = ConfigDict({"abbr": "judge-model"})
        
        result = deal_with_judge_model_abbr(model_cfg, judge_cfg, meta=False)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["abbr"], "judged-by--judge-model")
        mock_logger.debug.assert_called()

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    def test_deal_with_judge_model_abbr_existing_judge(self, mock_logger):
        """Test deal_with_judge_model_abbr doesn't duplicate judge suffix"""
        model_cfg = (
            ConfigDict({"abbr": "gpt-4"}),
            ConfigDict({"abbr": "judged-by--existing-judge"}),
        )
        judge_cfg = ConfigDict({"abbr": "new-judge"})
        
        result = deal_with_judge_model_abbr(model_cfg, judge_cfg, meta=False)
        
        # Should not add another judge
        self.assertEqual(len(result), 2)
        mock_logger.debug.assert_not_called()

    @patch("ais_bench.benchmark.utils.core.abbr.logger")
    def test_deal_with_judge_model_abbr_meta(self, mock_logger):
        """Test deal_with_judge_model_abbr with meta=True"""
        model_cfg = ConfigDict({"abbr": "gpt-4"})
        judge_cfg = ConfigDict({"abbr": "summarizer"})
        
        result = deal_with_judge_model_abbr(model_cfg, judge_cfg, meta=True)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["abbr"], "summarized-by--summarizer")


if __name__ == "__main__":
    unittest.main()