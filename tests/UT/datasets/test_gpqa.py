import unittest
from unittest.mock import patch, mock_open

from datasets import Dataset

from ais_bench.benchmark.datasets.gpqa import (
    GPQADataset,
    GPQASimpleEvalDataset,
    GPQAEvaluator,
    GPQA_Simple_Eval_postprocess,
)


class TestGPQA(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.gpqa.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_dataset(self, mock_open_file, mock_get_path):
        # CSV 头 + 一行数据；索引7是Question，8-11为选项
        content = (
            "h0,h1,h2,h3,h4,h5,h6,Question,A,B,C,D\n"
            ",,,,,,,Q,oa,ob,oc,od\n"
        )
        m = mock_open(read_data=content)
        mock_open_file.return_value = m.return_value
        ds = GPQADataset.load("/any", name="file.csv")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)

    @patch("ais_bench.benchmark.datasets.gpqa.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_simple_eval_dataset(self, mock_open_file, mock_get_path):
        content = (
            "h0,h1,h2,h3,h4,h5,h6,Question,A,B,C,D\n"
            ",,,,,,,Q,oa,ob,oc,od\n"
        )
        m = mock_open(read_data=content)
        mock_open_file.return_value = m.return_value
        ds = GPQASimpleEvalDataset.load("/any", name="file.csv")
        self.assertIsInstance(ds, Dataset)
        self.assertGreaterEqual(len(ds), 1)

    def test_evaluator_and_postprocess(self):
        eva = GPQAEvaluator()
        out = eva.score(["A"], ["A"])
        self.assertIn("accuracy", out)
        self.assertEqual(GPQA_Simple_Eval_postprocess("Answer: B"), "B")


if __name__ == "__main__":
    unittest.main()
