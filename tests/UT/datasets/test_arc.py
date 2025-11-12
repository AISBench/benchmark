import unittest
from unittest.mock import patch, mock_open

from datasets import DatasetDict

from ais_bench.benchmark.datasets.arc import ARCDataset


class TestARCDataset(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.arc.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_load(self, mock_open_file, mock_get_path):
        # 模拟Dev/Test两个文件各一行，且choices长度为4
        line = (
            '{"question": {"stem": "Q?", "choices": '
            '[{"label": "A", "text": "ta"}, '
            '{"label": "B", "text": "tb"}, '
            '{"label": "C", "text": "tc"}, '
            '{"label": "D", "text": "td"}]}, '
            '"answerKey": "B"}'
        )
        m = mock_open(read_data=line + "\n")
        mock_open_file.side_effect = [m.return_value, m.return_value]

        ds = ARCDataset.load(path="/any", name="ARC")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("Dev", ds)
        self.assertIn("Test", ds)
        self.assertGreater(len(ds["Dev"]), 0)
        row = ds["Dev"][0]
        for key in ["question", "answerKey", "textA", "textB", "textC", "textD"]:
            self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
