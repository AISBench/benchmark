import unittest
import json
import numpy as np
import pandas as pd
from unittest.mock import patch
from datasets import Dataset

from ais_bench.benchmark.datasets.hle import (
    HLEDataset,
    HLEJGDataset,
    HLEJudgeEvaluator,
    parse_hle_item,
    parse_predictions,
    calib_err,
    dump_metrics,
)


class TestParseHLEItem(unittest.TestCase):
    """Tests for parse_hle_item function."""

    def test_parse_with_image(self):
        """Test parsing item with image URL."""
        item = {
            "question": "What is in the picture?",
            "image": "http://example.com/image.jpg",
            "answer": "A cat"
        }
        result = parse_hle_item(item)
        
        self.assertIn("content", result)
        self.assertIn("answer", result)
        self.assertEqual(result["answer"], "A cat")
        self.assertIn("What is in the picture?", result["content"])

    def test_parse_without_image(self):
        """Test parsing item without image."""
        item = {
            "question": "What is 2+2?",
            "answer": "4"
        }
        result = parse_hle_item(item)
        
        self.assertIn("content", result)
        self.assertEqual(result["answer"], "4")

    def test_parse_empty_question(self):
        """Test parsing item with empty question."""
        item = {
            "question": "",
            "answer": "A"
        }
        result = parse_hle_item(item)
        
        self.assertIn("content", result)
        self.assertEqual(result["answer"], "A")

    def test_parse_empty_answer(self):
        """Test parsing item with empty answer."""
        item = {
            "question": "Q?",
            "answer": ""
        }
        result = parse_hle_item(item)
        
        self.assertEqual(result["answer"], "")

    def test_parse_missing_fields(self):
        """Test parsing item with missing fields."""
        item = {}
        result = parse_hle_item(item)
        
        self.assertIn("content", result)
        self.assertEqual(result["answer"], "")


class TestHLEDataset(unittest.TestCase):
    """Tests for HLEDataset class."""

    @patch("ais_bench.benchmark.datasets.hle.get_data_path", return_value="/test.parquet")
    @patch("os.path.exists", return_value=True)
    @patch("pandas.read_parquet")
    def test_load_success(self, mock_read_parquet, mock_exists, mock_get_path):
        """Test successful dataset loading."""
        # Create mock data
        mock_data = pd.DataFrame([
            {"question": "Q1", "image": "data:image/webp;base64,UklGRlaZAABXRUJQV", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
        ])
        mock_read_parquet.return_value = mock_data
        
        result = HLEDataset.load("/test.parquet")
        
        self.assertIsInstance(result, Dataset)
        self.assertEqual(len(result), 2)

    @patch("ais_bench.benchmark.datasets.hle.get_data_path", return_value="/nonexistent.parquet")
    @patch("os.path.exists", return_value=False)
    def test_load_file_not_found(self, mock_exists, mock_get_path):
        """Test loading when file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            HLEDataset.load("/nonexistent.parquet")

    @patch("ais_bench.benchmark.datasets.hle.get_data_path", return_value="/test.parquet")
    @patch("os.path.exists", return_value=True)
    @patch("pandas.read_parquet")
    def test_load_empty_dataset(self, mock_read_parquet, mock_exists, mock_get_path):
        """Test loading empty dataset."""
        mock_data = pd.DataFrame([])
        mock_read_parquet.return_value = mock_data
        
        result = HLEDataset.load("/test.parquet")
        
        self.assertIsInstance(result, Dataset)
        self.assertEqual(len(result), 0)


class TestHLEJGDataset(unittest.TestCase):
    """Tests for HLEJGDataset class."""

    def test_get_dataset_class(self):
        """Test _get_dataset_class method returns HLEDataset."""
        dataset = HLEJGDataset.__new__(HLEJGDataset)
        dataset_class = dataset._get_dataset_class()
        
        self.assertEqual(dataset_class, HLEDataset)


class TestParsePredictions(unittest.TestCase):
    """Tests for parse_predictions function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON predictions."""
        pred = json.dumps({
            "extracted_final_answer": "A",
            "reasoning": "Because...",
            "correct": "yes",
            "confidence": 90
        })
        result = parse_predictions([pred])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["model_answer"], "A")
        self.assertEqual(result[0]["correct"], "yes")
        self.assertEqual(result[0]["confidence"], 90)

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON should skip and return empty."""
        pred = "invalid json {{{"
        result = parse_predictions([pred])
        
        self.assertEqual(len(result), 0)

    def test_parse_mixed_valid_invalid(self):
        """Test parsing mixed valid and invalid predictions."""
        valid_pred = json.dumps({"correct": "yes", "confidence": 80})
        invalid_pred = "invalid json"
        result = parse_predictions([valid_pred, invalid_pred, valid_pred])
        
        self.assertEqual(len(result), 2)

    def test_parse_missing_fields(self):
        """Test parsing JSON with missing fields."""
        pred = json.dumps({"reasoning": "Missing some fields"})
        result = parse_predictions([pred])
        
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]["model_answer"])
        self.assertIsNone(result[0]["correct"])

    def test_parse_with_newlines(self):
        """Test parsing predictions with newline characters."""
        pred = '''{
            "extracted_final_answer": "A",
            "correct": "yes"
        }'''
        result = parse_predictions([pred])
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["correct"], "yes")


class TestCalibErr(unittest.TestCase):
    """Tests for calib_err function."""

    def test_basic_l2_calibration(self):
        """Test L2 calibration error calculation."""
        confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        correct = np.array([1, 1, 0, 0, 0])
        result = calib_err(confidence, correct, p="2", beta=5)
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        
    def test_multiple_bins(self):
        """Test calibration with multiple bins (to cover more code)."""
        # Generate enough data points to create multiple bins
        num_samples = 200
        confidence = np.linspace(0.0, 1.0, num_samples)
        correct = (np.random.rand(num_samples) < confidence).astype(float)
        
        # Test with beta=50 so we get 4 bins (200/50=4)
        result_l2 = calib_err(confidence, correct, p="2", beta=50)
        result_l1 = calib_err(confidence, correct, p="1", beta=50)
        result_max = calib_err(confidence, correct, p="infty", beta=50)
        
        self.assertIsInstance(result_l2, float)
        self.assertIsInstance(result_l1, float)
        self.assertIsInstance(result_max, float)

    def test_too_few_samples(self):
        """Test edge case with too few samples for binning."""
        confidence = np.array([0.9])
        correct = np.array([1])
        result = calib_err(confidence, correct, beta=100)
        
        # Should return overall error
        self.assertIsInstance(result, float)

    def test_zero_samples(self):
        """Test with zero samples (edge case)."""
        confidence = np.array([])
        correct = np.array([])
        result = calib_err(confidence, correct, beta=10)
        
        # Should handle this gracefully
        self.assertIsInstance(result, float)

    def test_all_correct(self):
        """Test all predictions are correct."""
        confidence = np.array([0.95, 0.90, 0.85])
        correct = np.array([1, 1, 1])
        result = calib_err(confidence, correct, beta=3)
        
        self.assertIsInstance(result, float)

    def test_nan_values(self):
        """Test handling of NaN values."""
        confidence = np.array([0.9, np.nan, 0.7])
        correct = np.array([1, 0, 0])
        result = calib_err(confidence, correct, beta=3)
        
        self.assertIsInstance(result, float)


class TestDumpMetrics(unittest.TestCase):
    """Tests for dump_metrics function."""

    def test_all_correct(self):
        """Test metrics when all predictions are correct."""
        judge_results = [
            {"correct": "yes", "confidence": 90},
            {"correct": "yes", "confidence": 85},
        ]
        result = dump_metrics(judge_results, n=2)
        
        self.assertIn("accuracy", result)
        self.assertIn("calibration_error", result)
        self.assertEqual(result["sample_num"], 2)
        self.assertIn("100.0%", result["accuracy"])

    def test_all_incorrect(self):
        """Test metrics when all predictions are incorrect."""
        judge_results = [
            {"correct": "no", "confidence": 70},
            {"correct": "no", "confidence": 60},
        ]
        result = dump_metrics(judge_results, n=2)
        
        self.assertIn("accuracy", result)
        self.assertIn("0.0%", result["accuracy"])

    def test_mixed_results(self):
        """Test metrics with mixed correct/incorrect."""
        judge_results = [
            {"correct": "yes", "confidence": 90},
            {"correct": "no", "confidence": 70},
            {"correct": "yes", "confidence": 80},
        ]
        result = dump_metrics(judge_results, n=3)
        
        self.assertIn("accuracy", result)
        self.assertIn("calibration_error", result)

    def test_empty_results(self):
        """Test empty judge results."""
        result = dump_metrics([], n=0)
        
        self.assertEqual(result["accuracy"], "0%")
        self.assertEqual(result["sample_num"], 0)

    def test_partial_results(self):
        """Test when we have fewer results than expected n."""
        judge_results = [{"correct": "yes", "confidence": 80}]
        result = dump_metrics(judge_results, n=5)
        
        self.assertEqual(result["sample_num"], 5)
        self.assertIn("20.0%", result["accuracy"])

    def test_missing_fields(self):
        """Test judge results with missing fields."""
        judge_results = [
            {"correct": "yes"},  # Missing confidence
            {"confidence": 70},  # Missing correct
        ]
        result = dump_metrics(judge_results, n=2)
        
        self.assertIn("accuracy", result)
        self.assertIn("calibration_error", result)

    def test_confidence_edge_cases(self):
        """Test edge cases for confidence values."""
        judge_results = [
            {"correct": "yes", "confidence": 100},
            {"correct": "no", "confidence": 0},
        ]
        result = dump_metrics(judge_results, n=2)
        
        self.assertIsInstance(result["calibration_error"], float)


class TestHLEJudgeEvaluator(unittest.TestCase):
    """Tests for HLEJudgeEvaluator class."""

    def setUp(self):
        """Set up evaluator instance."""
        self.evaluator = HLEJudgeEvaluator()

    def test_score_basic(self):
        """Test basic scoring functionality."""
        predictions = [
            json.dumps({"correct": "yes", "confidence": 90}),
            json.dumps({"correct": "no", "confidence": 70}),
        ]
        references = ["A", "B"]
        
        result = self.evaluator.score(predictions, references)
        
        self.assertIn("accuracy", result)
        self.assertIn("calibration_error", result)

    def test_score_length_mismatch(self):
        """Test error when predictions and references have different lengths."""
        predictions = [json.dumps({"correct": "yes"})]
        references = ["A", "B"]
        
        result = self.evaluator.score(predictions, references)
        
        self.assertIn("error", result)

    def test_score_empty_predictions(self):
        """Test scoring with empty predictions."""
        predictions = []
        references = []
        
        result = self.evaluator.score(predictions, references)
        
        self.assertIn("accuracy", result)

    def test_score_all_correct(self):
        """Test scoring when all are correct."""
        predictions = [
            json.dumps({"correct": "yes", "confidence": 95}),
            json.dumps({"correct": "yes", "confidence": 85}),
            json.dumps({"correct": "yes", "confidence": 75}),
        ]
        references = ["A", "B", "C"]
        
        result = self.evaluator.score(predictions, references)
        
        self.assertIn("accuracy", result)
        self.assertIn("100.0%", result["accuracy"])


if __name__ == '__main__':
    unittest.main()
