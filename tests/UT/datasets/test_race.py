import unittest
from unittest.mock import patch, mock_open, MagicMock
import importlib

from datasets import DatasetDict

from ais_bench.benchmark.datasets.race import RaceDataset


class TestRaceDataset(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.race.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    @patch("ais_bench.benchmark.datasets.race.environ.get", return_value=None)
    def test_local_branch(self, mock_env, mock_open_file, mock_get_path):
        line = '{"article": "a", "question": "q", "options": ["A","B","C","D"], "answer": "A"}'
        m = mock_open(read_data=line + "\n")
        # 两个split都会被打开
        mock_open_file.side_effect = [m.return_value, m.return_value]
        ds = RaceDataset.load("/any", name="name")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("validation", ds)
        self.assertIn("test", ds)

    @patch("ais_bench.benchmark.datasets.race.get_data_path", return_value="repo")
    @patch("ais_bench.benchmark.datasets.race.environ.get", return_value="ModelScope")
    def test_modelscope_branch(self, mock_env, mock_get_path):
        with patch.dict('sys.modules', {'modelscope': MagicMock()}):
            ms = importlib.import_module('modelscope')
            ms.MsDataset = MagicMock()
            ms.MsDataset.load.return_value = [
                {"article": "a", "question": "q", "options": ["A","B","C","D"], "answer": "B"}
            ]
            ds = RaceDataset.load("/any", name="name")
            self.assertIsInstance(ds, DatasetDict)


if __name__ == "__main__":
    unittest.main()
