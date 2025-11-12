import unittest
from unittest.mock import patch, mock_open

from datasets import DatasetDict

from ais_bench.benchmark.datasets.lambada import lambadaDataset, LambadaEvaluator


class TestLambadaDataset(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.lambada.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    @patch("ais_bench.benchmark.datasets.lambada.environ.get", return_value=None)
    def test_load_local(self, mock_environ_get, mock_open_file, mock_get_path):
        line = '{"text": "hello world"}'
        m = mock_open(read_data=line + "\n")
        mock_open_file.return_value = m.return_value
        ds = lambadaDataset.load("/any")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("test", ds)
        self.assertEqual(len(ds["test"]), 1)


class TestLambadaEvaluator(unittest.TestCase):
    def test_score_basic(self):
        evaluator = LambadaEvaluator()
        predictions = ["Hello!"]
        references = ["hello"]
        out = evaluator.score(predictions, references)
        self.assertIn("accuracy", out)
        self.assertGreaterEqual(out["accuracy"], 0.0)

    def test_score_len_mismatch(self):
        evaluator = LambadaEvaluator()
        out = evaluator.score(["a"], ["a", "b"])  # 长度不一致
        self.assertIn("error", out)


if __name__ == "__main__":
    unittest.main()
