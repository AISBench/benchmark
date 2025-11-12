import unittest
from unittest.mock import patch, mock_open
import types

from datasets import Dataset

from ais_bench.benchmark.datasets.lcsts import LCSTSDataset, lcsts_postprocess


class TestLCSTS(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.lcsts.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    @patch("ais_bench.benchmark.datasets.lcsts.environ.get", return_value=None)
    def test_local_branch(self, mock_env, mock_open_file, mock_get_path):
        src = "text1\n"
        tgt = "summary1\n"
        m_src = mock_open(read_data=src)
        m_tgt = mock_open(read_data=tgt)
        mock_open_file.side_effect = [m_src.return_value, m_tgt.return_value]
        ds = LCSTSDataset.load("/any")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(ds[0]['abst'], 'summary1')

    @patch("ais_bench.benchmark.datasets.lcsts.get_data_path", return_value="repo")
    @patch("ais_bench.benchmark.datasets.lcsts.environ.get", return_value="ModelScope")
    def test_modelscope_branch(self, mock_env, mock_get_path):
        class DummyMsDataset:
            @staticmethod
            def load(path, split=None):
                return [{'text': 'c', 'summary': 's'}]
        fake_module = types.SimpleNamespace(MsDataset=DummyMsDataset)
        with patch.dict('sys.modules', {'modelscope': fake_module}):
            ds = LCSTSDataset.load("/any")
            self.assertIsInstance(ds, Dataset)

    def test_postprocess(self):
        self.assertEqual(lcsts_postprocess('1. “Hello，”'), 'Hello')


if __name__ == '__main__':
    unittest.main()
