import unittest
from unittest.mock import patch, mock_open, MagicMock
import importlib
import types

from datasets import DatasetDict, Dataset

from ais_bench.benchmark.datasets.siqa import siqaDataset_V2


class TestSIQA(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.siqa.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    @patch("ais_bench.benchmark.datasets.siqa.environ.get", return_value=None)
    def test_local_branch(self, mock_env, mock_open_file, mock_get_path):
        data_line = '{"answerA": "a", "answerB": "b", "answerC": "c"}'
        label_line = '2'
        m_data = mock_open(read_data=data_line + "\n")
        m_label = mock_open(read_data=label_line + "\n")
        # train.jsonl, train-labels.lst, dev.jsonl, dev-labels.lst
        mock_open_file.side_effect = [
            m_data.return_value,
            m_label.return_value,
            m_data.return_value,
            m_label.return_value,
        ]
        ds = siqaDataset_V2.load("/any")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("train", ds)
        self.assertIn("validation", ds)

    @patch("ais_bench.benchmark.datasets.siqa.get_data_path", return_value="repo")
    @patch("ais_bench.benchmark.datasets.siqa.environ.get", return_value="ModelScope")
    def test_modelscope_branch(self, mock_env, mock_get_path):
        class DummyMsDataset:
            @staticmethod
            def load(path, split=None):
                return [{"answerA": "a", "answerB": "b", "answerC": "c", "label": "2"}]
        fake_module = types.SimpleNamespace(MsDataset=DummyMsDataset)
        with patch.dict('sys.modules', {'modelscope': fake_module}):
            ds = siqaDataset_V2.load("/any")
            self.assertIsInstance(ds, DatasetDict)
            self.assertIn("train", ds)


if __name__ == "__main__":
    unittest.main()
