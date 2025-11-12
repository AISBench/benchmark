import unittest
from unittest.mock import patch, mock_open

from datasets import DatasetDict

from ais_bench.benchmark.datasets.mmlu import MMLUDataset


class TestMMLU(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.mmlu.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_load(self, mock_open_file, mock_get_path):
        # csv.reader逐行返回，包含6列
        csv_content = "input,A,B,C,D,target\n"
        m = mock_open(read_data=csv_content)
        # dev 和 test 两次打开
        mock_open_file.side_effect = [m.return_value, m.return_value]
        ds = MMLUDataset.load("/any", name="name")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("dev", ds)
        self.assertIn("test", ds)


if __name__ == "__main__":
    unittest.main()
