import unittest
from unittest.mock import patch, mock_open

from datasets import DatasetDict

from ais_bench.benchmark.datasets.drop_simple_eval import (
    DropOpenAIDataset,
    DropOpenAIEvaluator,
    normalize,
    fuzzy_match,
)


class TestDropSimpleEval(unittest.TestCase):
    def test_normalize_and_fuzzy(self):
        self.assertEqual(normalize('The, apple!'), 'apple')
        self.assertTrue(fuzzy_match('Answer', 'answer is correct'))
        self.assertTrue(fuzzy_match('', ''))  # 覆盖空字符串分支

    @patch("ais_bench.benchmark.datasets.drop_simple_eval.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_load(self, mock_open_file, mock_get_path):
        line = '{"context": "ctx", "ref_text": "ans"}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = DropOpenAIDataset.load("/any")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("validation", ds)
        self.assertEqual(len(ds["validation"]), 1)

    def test_evaluator(self):
        eva = DropOpenAIEvaluator()
        out = eva.score(["Answer: Foo"], ["foo|bar"])
        self.assertIn("accuracy", out)
        # 长度不一致
        out2 = eva.score(["a"], ["foo", "bar"])
        self.assertIn("error", out2)


if __name__ == '__main__':
    unittest.main()
