import unittest
from unittest.mock import patch, mock_open

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.triviaqa import (
    TriviaQADataset,
    TriviaQADatasetV2,
    TriviaQADatasetV3,
    TriviaQAEvaluator,
)


class TestTriviaQADatasets(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.triviaqa.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_v1(self, mock_open_file, mock_get_path):
        # tab分隔，两列
        content = "q\t['a','b']\n"
        m = mock_open(read_data=content)
        # dev 与 test 两次打开
        mock_open_file.side_effect = [m.return_value, m.return_value]
        ds = TriviaQADataset.load("/any")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("dev", ds)
        self.assertIn("test", ds)

    @patch("ais_bench.benchmark.datasets.triviaqa.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_v2(self, mock_open_file, mock_get_path):
        line = '{"field": 1}'
        m = mock_open(read_data=line + "\n")
        # validation 与 train 两次打开
        mock_open_file.side_effect = [m.return_value, m.return_value]
        ds = TriviaQADatasetV2.load("/any")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("validation", ds)
        self.assertIn("train", ds)

    @patch("ais_bench.benchmark.datasets.triviaqa.get_data_path", return_value="/fake/path/file.jsonl")
    @patch("builtins.open")
    def test_v3(self, mock_open_file, mock_get_path):
        line = '{"field": 1}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = TriviaQADatasetV3.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertGreaterEqual(len(ds), 1)


class TestTriviaQAEvaluator(unittest.TestCase):
    def test_score(self):
        eva = TriviaQAEvaluator()
        preds = [" The answer is apple \n extra"]
        refs = [["apple", "orange"]]
        out = eva.score(preds, refs)
        self.assertIn("score", out)
        self.assertIn("details", out)
        # 长度不一致
        out2 = eva.score(["a"], [["a"], ["b"]])
        self.assertIn("error", out2)


if __name__ == "__main__":
    unittest.main()
