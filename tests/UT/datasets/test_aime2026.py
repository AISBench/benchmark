import unittest
from unittest.mock import patch, mock_open

from datasets import Dataset

from ais_bench.benchmark.datasets.aime2026 import Aime2026Dataset


class TestAime2026Dataset(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.aime2026.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_aime2026_load(self, mock_open_file, mock_get_path):
        line = '{"question": "What is 1+1?", "answer": "2"}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = Aime2026Dataset.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)
        row = ds[0]
        self.assertEqual(row["question"], "What is 1+1?")
        self.assertEqual(row["answer"], "2")

    @patch("ais_bench.benchmark.datasets.aime2026.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_aime2026_load_multiple_records(self, mock_open_file, mock_get_path):
        lines = '{"id": 1, "value": "a"}\n{"id": 2, "value": "b"}\n{"id": 3, "value": "c"}\n'
        m = mock_open(read_data=lines)
        mock_open_file.return_value = m.return_value
        ds = Aime2026Dataset.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0]["id"], 1)
        self.assertEqual(ds[1]["value"], "b")
        self.assertEqual(ds[2]["value"], "c")

    @patch("ais_bench.benchmark.datasets.aime2026.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_aime2026_load_strips_whitespace(self, mock_open_file, mock_get_path):
        line = '  {"key": "value"}  '
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = Aime2026Dataset.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]["key"], "value")

    @patch("ais_bench.benchmark.datasets.aime2026.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_aime2026_load_fields_preserved(self, mock_open_file, mock_get_path):
        line = '{"origin_prompt": "Solve this", "gold_answer": "42", "id": 101, "subject": "math"}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = Aime2026Dataset.load("/any")
        self.assertIsInstance(ds, Dataset)
        row = ds[0]
        self.assertEqual(row["origin_prompt"], "Solve this")
        self.assertEqual(row["gold_answer"], "42")
        self.assertEqual(row["id"], 101)
        self.assertEqual(row["subject"], "math")


if __name__ == "__main__":
    unittest.main()
