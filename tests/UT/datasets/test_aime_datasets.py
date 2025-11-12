import unittest
from unittest.mock import patch, mock_open

from datasets import Dataset

from ais_bench.benchmark.datasets.aime2024 import Aime2024Dataset
from ais_bench.benchmark.datasets.aime2025 import Aime2025Dataset


class TestAimeDatasets(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.aime2024.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_aime2024_load(self, mock_open_file, mock_get_path):
        line = '{"origin_prompt": "What?", "gold_answer": "42"}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = Aime2024Dataset.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)
        row = ds[0]
        self.assertIn("question", row)
        self.assertIn("answer", row)

    @patch("ais_bench.benchmark.datasets.aime2025.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_aime2025_load(self, mock_open_file, mock_get_path):
        line = '{"field": "value"}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = Aime2025Dataset.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)


if __name__ == "__main__":
    unittest.main()
