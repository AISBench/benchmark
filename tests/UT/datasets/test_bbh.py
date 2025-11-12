import unittest
from unittest.mock import patch, mock_open, MagicMock
import json

from datasets import Dataset

from ais_bench.benchmark.datasets.bbh import (
    BBHDataset,
    bbh_mcq_postprocess,
    bbh_freeform_postprocess,
    BBHEvaluator,
    BBHEvaluator_mcq,
)


class TestBBHDataset(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.bbh.get_data_path", return_value="/fake/path")
    @patch("ais_bench.benchmark.datasets.bbh.environ.get", return_value=None)
    @patch("builtins.open", new_callable=mock_open)
    def test_load_local(self, mock_open_file, mock_environ_get, mock_get_path):
        data = {"examples": [{"question": "Q?", "answer": "A"}]}
        mock_open_file.return_value.read.return_value = json.dumps(data)
        ds = BBHDataset.load("/any", "test_name")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)

    @patch("ais_bench.benchmark.datasets.bbh.get_data_path", return_value="repo")
    @patch("ais_bench.benchmark.datasets.bbh.environ.get", return_value="ModelScope")
    def test_load_modelscope(self, mock_environ_get, mock_get_path):
        from datasets import Dataset
        
        mock_ms_dataset = MagicMock()
        mock_item = {"question": "Q?", "answer": "A"}
        mock_ms_dataset.__iter__ = lambda self: iter([mock_item])
        
        # 将 mock_ms_dataset 转换为 Dataset
        dataset_from_list = Dataset.from_list([mock_item])
        
        with patch.dict('sys.modules', {'modelscope': MagicMock()}):
            import sys
            sys.modules['modelscope'].MsDataset = MagicMock()
            sys.modules['modelscope'].MsDataset.load.return_value = dataset_from_list
            
            ds = BBHDataset.load("repo", "test_name")
            self.assertIsInstance(ds, Dataset)


class TestBBHPostprocess(unittest.TestCase):
    def test_bbh_mcq_postprocess_with_answer_is(self):
        text = "The answer is (A) correct"
        result = bbh_mcq_postprocess(text)
        self.assertEqual(result, "A")

    def test_bbh_mcq_postprocess_with_capital_letter(self):
        text = "The answer is B correct"
        result = bbh_mcq_postprocess(text)
        self.assertEqual(result, "B")

    def test_bbh_mcq_postprocess_no_match(self):
        text = "The answer is correct"
        result = bbh_mcq_postprocess(text)
        # 如果没有匹配到大写字母，返回处理后的文本（去掉 "answer is " 前缀）
        self.assertEqual(result, "correct")

    def test_bbh_freeform_postprocess_with_answer_is(self):
        text = "The answer is **correct answer**"
        result = bbh_freeform_postprocess(text)
        self.assertEqual(result, "correct answer")

    def test_bbh_freeform_postprocess_with_dot(self):
        text = "The answer is correct answer."
        result = bbh_freeform_postprocess(text)
        self.assertEqual(result, "correct answer")

    def test_bbh_freeform_postprocess_with_newline(self):
        text = "The answer is correct answer\nmore text"
        result = bbh_freeform_postprocess(text)
        self.assertEqual(result, "correct answer")

    def test_bbh_freeform_postprocess_no_answer_is(self):
        text = "correct answer"
        result = bbh_freeform_postprocess(text)
        self.assertEqual(result, "correct answer")


class TestBBHEvaluator(unittest.TestCase):
    def test_score_success(self):
        evaluator = BBHEvaluator()
        predictions = ["The answer is **correct**", "wrong"]
        references = ["correct", "wrong"]
        result = evaluator.score(predictions, references)
        self.assertIn("score", result)
        self.assertEqual(result["score"], 100.0)
        self.assertIn("details", result)

    def test_score_length_mismatch(self):
        evaluator = BBHEvaluator()
        result = evaluator.score(["pred1"], ["ref1", "ref2"])
        self.assertIn("error", result)


class TestBBHEvaluatorMCQ(unittest.TestCase):
    def test_score_success(self):
        evaluator = BBHEvaluator_mcq()
        predictions = ["A", "B"]
        references = ["A", "B"]
        result = evaluator.score(predictions, references)
        self.assertIn("score", result)
        self.assertEqual(result["score"], 100.0)
        self.assertIn("details", result)

    def test_score_partial(self):
        evaluator = BBHEvaluator_mcq()
        predictions = ["A", "B"]
        references = ["A", "C"]
        result = evaluator.score(predictions, references)
        self.assertEqual(result["score"], 50.0)

    def test_score_length_mismatch(self):
        evaluator = BBHEvaluator_mcq()
        result = evaluator.score(["pred1"], ["ref1", "ref2"])
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()

