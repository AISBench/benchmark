import unittest
from unittest.mock import patch, mock_open

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.winogrande import (
    WinograndeDataset,
    WinograndeDatasetV2,
    WinograndeDatasetV3,
)


class TestWinogrande(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.winogrande.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_v1(self, mock_open_file, mock_get_path):
        line = '{"sentence": "x _ y", "option1": "A", "option2": "B", "answer": "A"}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = WinograndeDataset.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertGreaterEqual(len(ds), 1)
        self.assertIn("prompt", ds[0])
        self.assertIn("opt1", ds[0])
        self.assertIn("opt2", ds[0])

    @patch("ais_bench.benchmark.datasets.winogrande.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_v2(self, mock_open_file, mock_get_path):
        line = '{"sentence": "x _ y", "option1": "A", "option2": "B", "answer": "1"}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = WinograndeDatasetV2.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertGreaterEqual(len(ds), 1)
        # answer "1" maps to ' AB'[1] which is 'B' (space at index 0, 'A' at 1, 'B' at 2)
        # But wait, ' AB'[1] is 'A', ' AB'[2] is 'B'
        # So answer "1" should be ' AB'[1] = 'A', but we expect "B"?
        # Let me check: if answer is "1", int("1") = 1, ' AB'[1] = 'A'
        # But the test expects "B", so maybe answer "1" means option 1 (which is index 1, which is option2/B)?
        # Actually, looking at the code: answer = ' AB'[int(answer)]
        # If answer is "1", int("1") = 1, ' AB'[1] = 'A' (space at 0, A at 1)
        # But if answer is "2", int("2") = 2, ' AB'[2] = 'B'
        # So "1" should map to "A", not "B". Let me fix the test:
        self.assertEqual(ds[0]["answer"], "A")

    @patch("ais_bench.benchmark.datasets.winogrande.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_v2_empty_answer(self, mock_open_file, mock_get_path):
        line = '{"sentence": "x _ y", "option1": "A", "option2": "B", "answer": ""}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = WinograndeDatasetV2.load("/any")
        self.assertEqual(ds[0]["answer"], "NULL")

    @patch("ais_bench.benchmark.datasets.winogrande.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_v3(self, mock_open_file, mock_get_path):
        line = '{"sentence": "x _ y", "option1": "A", "option2": "B", "answer": "1"}'
        m = mock_open(read_data=line + "\n")
        # train_xs 与 dev 两次打开
        mock_open_file.side_effect = [m.return_value, m.return_value]
        ds = WinograndeDatasetV3.load("/any")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("train_xs", ds)
        self.assertIn("dev", ds)


if __name__ == "__main__":
    unittest.main()
