import unittest
from unittest.mock import patch, mock_open

from datasets import Dataset

from ais_bench.benchmark.datasets.xsum import XsumDataset, Xsum_postprocess


class TestXsum(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.xsum.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    @patch("ais_bench.benchmark.datasets.xsum.environ.get", return_value=None)
    def test_xsum_load_local(self, mock_env, mock_open_file, mock_get_path):
        # dev.jsonl
        line = '{"dialogue": "D", "summary": "S"}'
        m = mock_open(read_data=line + "\n")
        # open(path/dev.jsonl)
        mock_open_file.return_value = m.return_value
        out = XsumDataset.load("/any")
        self.assertIsInstance(out, Dataset)
        self.assertEqual(len(out), 1)

    def test_postprocess(self):
        self.assertEqual(Xsum_postprocess(" a\n b "), "a")


if __name__ == "__main__":
    unittest.main()
