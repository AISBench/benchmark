import unittest
from unittest.mock import patch, MagicMock
import importlib

from datasets import Dataset

from ais_bench.benchmark.datasets.xsum import XsumDataset


class TestXsumModelScope(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.xsum.get_data_path", return_value="repo_id")
    @patch("ais_bench.benchmark.datasets.xsum.environ.get", return_value="ModelScope")
    def test_models_scope_branch(self, mock_env, mock_get_path):
        with patch.dict('sys.modules', {'modelscope': MagicMock()}):
            ms = importlib.import_module('modelscope')
            ms.MsDataset = MagicMock()
            ms.MsDataset.load.return_value = [
                {"document": "doc1", "summary": "sum1"},
                {"document": "doc2", "summary": "sum2"},
            ]
            out = XsumDataset.load("anything")
            self.assertIsInstance(out, Dataset)
            self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
