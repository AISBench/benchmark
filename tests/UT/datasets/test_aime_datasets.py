import unittest
from unittest.mock import patch, mock_open

from datasets import Dataset

from ais_bench.benchmark.datasets.aime2024 import Aime2024Dataset
from ais_bench.benchmark.datasets.aime2025 import Aime2025Dataset, Aime2025JDGDataset


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

    @patch("ais_bench.benchmark.datasets.aime2024.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_aime2024_multiple_records(self, mock_open_file, mock_get_path):
        line1 = '{"origin_prompt": "Q1?", "gold_answer": "1"}'
        line2 = '{"origin_prompt": "Q2?", "gold_answer": "2"}'
        m = mock_open(read_data=line1 + "\n" + line2 + "\n")
        mock_open_file.return_value = m.return_value
        ds = Aime2024Dataset.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 2)
        self.assertIn("question", ds[0])
        self.assertIn("question", ds[1])

    @patch("ais_bench.benchmark.datasets.aime2024.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_aime2024_fields_mapping(self, mock_open_file, mock_get_path):
        line = '{"origin_prompt": "What is 1+1?", "gold_answer": "2"}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = Aime2024Dataset.load("/any")
        row = ds[0]
        self.assertEqual(row["question"], "What is 1+1?")
        self.assertEqual(row["answer"], "2")

    @patch("ais_bench.benchmark.datasets.aime2025.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_aime2025_multiple_records(self, mock_open_file, mock_get_path):
        line1 = '{"field": "value1"}'
        line2 = '{"field": "value2"}'
        m = mock_open(read_data=line1 + "\n" + line2 + "\n")
        mock_open_file.return_value = m.return_value
        ds = Aime2025Dataset.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 2)

    def test_aime2025_jdg_dataset_class(self):
        instance = Aime2025JDGDataset.__new__(Aime2025JDGDataset)
        result = instance._get_dataset_class()
        self.assertIs(result, Aime2025Dataset)


if __name__ == "__main__":
    unittest.main()
