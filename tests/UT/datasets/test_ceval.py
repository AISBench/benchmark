import unittest
from unittest.mock import patch, mock_open

from datasets import DatasetDict

from ais_bench.benchmark.datasets.ceval import CEvalDataset


class TestCEval(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.ceval.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_load(self, mock_open_file, mock_get_path):
        # dev/val/test 三个split，包含表头和一行数据
        csv_content = "q,explanation,answer\nwhat,why,42\n"
        # 为每次open提供全新的句柄，避免文件指针耗尽
        mock_open_file.side_effect = [
            mock_open(read_data=csv_content).return_value,
            mock_open(read_data=csv_content).return_value,
            mock_open(read_data=csv_content).return_value,
        ]
        ds = CEvalDataset.load("/any", name="name")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("dev", ds)
        self.assertIn("val", ds)
        self.assertIn("test", ds)


if __name__ == "__main__":
    unittest.main()
