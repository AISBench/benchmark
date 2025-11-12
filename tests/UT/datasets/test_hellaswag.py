import unittest
from unittest.mock import patch, mock_open

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.hellaswag import (
    HellaswagDataset,
    HellaswagDataset_V2,
    HellaswagDataset_V3,
    HellaswagDatasetwithICE,
)


class TestHellaswag(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.hellaswag.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_v1(self, mock_open_file, mock_get_path):
        line = '{"query": "ctx: hello", "choices": ["A","B","C","D"], "gold": "B"}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = HellaswagDataset.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)

    @patch("ais_bench.benchmark.datasets.hellaswag.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_v2(self, mock_open_file, mock_get_path):
        line = '{"query": "ctx: hello", "choices": ["A","B","C","D"], "gold": 1}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = HellaswagDataset_V2.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)

    @patch("ais_bench.benchmark.datasets.hellaswag.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_v3(self, mock_open_file, mock_get_path):
        line = '{"query": "question?", "choices": ["A","B","C","D"], "gold": 2}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = HellaswagDataset_V3.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)

    @patch("ais_bench.benchmark.datasets.hellaswag.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_with_ice(self, mock_open_file, mock_get_path):
        line = '{"query": "ctx: hello", "choices": ["A","B","C","D"], "gold": 1}'
        m = mock_open(read_data=line + "\n")
        # 依次打开两个不同文件
        mock_open_file.side_effect = [m.return_value, m.return_value]
        ds = HellaswagDatasetwithICE.load("/any")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("train", ds)
        self.assertIn("val", ds)


if __name__ == "__main__":
    unittest.main()
