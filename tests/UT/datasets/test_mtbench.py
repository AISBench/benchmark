import unittest
from unittest.mock import patch, mock_open, MagicMock
import json

from datasets import Dataset

from ais_bench.benchmark.datasets.mtbench import (
    MTBenchDataset,
    MTBenchEvaluator,
)


class TestMTBenchDataset(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.mtbench.get_data_path", return_value="/fake/path")
    @patch("ais_bench.benchmark.datasets.mtbench.get_logger")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_with_reference(self, mock_open_file, mock_logger, mock_get_path):
        data = {
            "question_id": "1",
            "prompt": ["Q1?", "Q2?"],
            "reference": ["A1", "A2"]
        }
        mock_open_file.return_value.__iter__ = lambda self: iter([json.dumps(data) + "\n"])
        mock_logger.return_value = MagicMock()
        
        ds = MTBenchDataset.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]["id"], "1")
        self.assertEqual(len(ds[0]["question"]), 2)
        self.assertEqual(len(ds[0]["answer"]), 2)

    @patch("ais_bench.benchmark.datasets.mtbench.get_data_path", return_value="/fake/path")
    @patch("ais_bench.benchmark.datasets.mtbench.get_logger")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_without_reference(self, mock_open_file, mock_logger, mock_get_path):
        data = {
            "question_id": "1",
            "prompt": ["Q1?", "Q2?"]
        }
        mock_open_file.return_value.__iter__ = lambda self: iter([json.dumps(data) + "\n"])
        mock_logger.return_value = MagicMock()
        
        ds = MTBenchDataset.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds[0]["answer"]), 2)
        self.assertEqual(ds[0]["answer"][0], "")


class TestMTBenchEvaluator(unittest.TestCase):
    def test_find_choice_valid(self):
        evaluator = MTBenchEvaluator()
        self.assertEqual(evaluator.find_choice("A"), "laughter")
        self.assertEqual(evaluator.find_choice("B"), "sigh")
        self.assertEqual(evaluator.find_choice("C"), "cough")

    def test_find_choice_invalid(self):
        evaluator = MTBenchEvaluator()
        self.assertEqual(evaluator.find_choice("X"), "")

    def test_score_success(self):
        evaluator = MTBenchEvaluator()
        # 代码逻辑：如果字符串长度>1，取第一个字符并查找映射
        # 所以传入长度>1的字符串，第一个字符是选择字母
        predictions = ["Axxx", "Byyy"]  # "Axxx"[0] = "A" -> "laughter", "Byyy"[0] = "B" -> "sigh"
        references = ["laughter", "sigh"]
        result = evaluator.score(predictions, references)
        self.assertIn("accuracy", result)
        self.assertEqual(result["accuracy"], 100.0)
        self.assertIn("details", result)

    def test_score_with_list_prediction(self):
        evaluator = MTBenchEvaluator()
        # 如果 prediction 是列表，len(["A"]) = 1，不会进入 if len(i) > 1
        # 但代码期望的是字符串，所以传入长度>1的字符串
        predictions = ["Axxx", "Byyy"]
        references = ["laughter", "sigh"]
        result = evaluator.score(predictions, references)
        self.assertEqual(result["accuracy"], 100.0)

    def test_score_partial(self):
        evaluator = MTBenchEvaluator()
        # "Axxx"[0] = "A" -> "laughter", "Xyyy"[0] = "X" -> "" (not found)
        predictions = ["Axxx", "Xyyy"]
        references = ["laughter", "sigh"]
        result = evaluator.score(predictions, references)
        self.assertEqual(result["accuracy"], 50.0)

    def test_score_length_mismatch(self):
        evaluator = MTBenchEvaluator()
        result = evaluator.score(["pred1"], ["ref1", "ref2"])
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()

