import unittest
from unittest.mock import patch, mock_open

from datasets import DatasetDict

from ais_bench.benchmark.datasets.cmmlu import CMMLUDataset


class TestCMMLU(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.cmmlu.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_load(self, mock_open_file, mock_get_path):
        # 第一行表头，第二行数据，共7列
        csv_content = "h1,h2,h3,h4,h5,h6,h7\n,question,A,B,C,D,answer\n"
        mock_open_file.side_effect = [
            mock_open(read_data=csv_content).return_value,
            mock_open(read_data=csv_content).return_value,
        ]
        ds = CMMLUDataset.load("/any", name="name")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("dev", ds)
        self.assertIn("test", ds)


if __name__ == "__main__":
    unittest.main()
