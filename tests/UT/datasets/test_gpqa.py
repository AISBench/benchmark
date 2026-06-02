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

    def test_evaluator_score_length_mismatch(self):
        eva = GPQAEvaluator()
        result = eva.score(['A', 'B'], ['A'])
        self.assertIn('error', result)

    def test_evaluator_score_all_correct(self):
        eva = GPQAEvaluator()
        result = eva.score(['A', 'B', 'C'], ['A', 'B', 'C'])
        self.assertEqual(result['accuracy'], 100.0)
        self.assertEqual(len(result['details']), 3)

    def test_evaluator_score_all_wrong(self):
        eva = GPQAEvaluator()
        result = eva.score(['A', 'B'], ['C', 'D'])
        self.assertEqual(result['accuracy'], 0.0)

    def test_evaluator_score_details_structure(self):
        eva = GPQAEvaluator()
        result = eva.score(['A'], ['B'])
        detail = result['details'][0]
        self.assertIn('pred', detail)
        self.assertIn('answer', detail)
        self.assertIn('correct', detail)

    def test_postprocess_no_match(self):
        result = GPQA_Simple_Eval_postprocess("This has no answer pattern")
        self.assertIsNone(result)

    def test_postprocess_case_insensitive(self):
        result = GPQA_Simple_Eval_postprocess("answer: B")
        self.assertEqual(result, 'B')

    @patch("ais_bench.benchmark.datasets.gpqa.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_dataset_multiple_rows(self, mock_open_file, mock_get_path):
        content = (
            "h0,h1,h2,h3,h4,h5,h6,Question,A,B,C,D\n"
            ",,,,,,,Q1,opt1_a,opt1_b,opt1_c,opt1_d\n"
            ",,,,,,,Q2,opt2_a,opt2_b,opt2_c,opt2_d\n"
        )
        m = mock_open(read_data=content)
        mock_open_file.return_value = m.return_value
        ds = GPQADataset.load("/any", name="file.csv")
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0]['answer'], 'D')
        self.assertEqual(ds[0]['D'], 'opt1_a')
        self.assertEqual(ds[1]['answer'], 'C')
        self.assertEqual(ds[1]['C'], 'opt2_a')


if __name__ == "__main__":
    unittest.main()
