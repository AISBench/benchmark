import unittest
from unittest.mock import patch, mock_open

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.boolq import BoolQDataset, BoolQDatasetV2, BoolQDatasetV3


class TestBoolQ(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.boolq.load_dataset")
    def test_boolq_load(self, mock_load):
        ds = Dataset.from_list([
            {"label": "true"},
            {"label": "false"},
        ])
        mock_load.return_value = DatasetDict({"train": ds, "test": ds})
        out = BoolQDataset.load(path="ignored")
        self.assertIn("train", out)
        # 映射后应有answer字段
        self.assertIn("answer", out["train"][0])

    @patch("ais_bench.benchmark.datasets.boolq.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_boolq_v2_load(self, mock_open_file, mock_get_path):
        line = '{"label": "true"}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        out = BoolQDatasetV2.load("/any")
        self.assertIsInstance(out, Dataset)
        self.assertEqual(out[0]["label"], "A")

    @patch("ais_bench.benchmark.datasets.boolq.get_data_path", return_value="/fake/path.jsonl")
    @patch("builtins.open")
    def test_boolq_v3_load(self, mock_open_file, mock_get_path):
        line = '{"passage": "a -- b -- c", "question": "who?"}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        out = BoolQDatasetV3.load("/any")
        self.assertIsInstance(out, Dataset)
        row = out[0]
        self.assertTrue(row["passage"].startswith("b"))
        self.assertTrue(row["question"][0].isupper())


if __name__ == "__main__":
    unittest.main()
