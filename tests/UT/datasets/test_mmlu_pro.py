import unittest
from unittest.mock import patch, MagicMock

from datasets import Dataset

from ais_bench.benchmark.datasets.mmlu_pro import MMLUProDataset, MMLUProBaseEvaluator


class DummyDataset:
    def __init__(self, data):
        self.data = data
    def filter(self, fn):
        filtered = [x for x in self.data if fn(x)]
        return DummyDataset(filtered)
    def map(self, fn):
        mapped = [fn(dict(item)) for item in self.data]
        return {'train': mapped}


class TestMMLUPro(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.mmlu_pro.get_data_path", return_value="repo")
    @patch("ais_bench.benchmark.datasets.mmlu_pro.load_dataset")
    def test_dataset(self, mock_load, mock_get_path):
        mock_load.return_value = DummyDataset([
            {'category': 'math', 'options': ['opt1', 'opt2'], 'answer': 'A', 'cot_content': "A: Let's think step by step."}
        ])
        ds = MMLUProDataset.load("/any", category='math')
        self.assertIn('train', ds)
        self.assertTrue(ds['train'][0]['options_str'])

    def test_evaluator(self):
        eva = MMLUProBaseEvaluator()
        out = eva.score(["A"], ["A. opt1" ])
        self.assertIn('accuracy', out)
        out2 = eva.score(["B"], ["A. opt1"])
        self.assertIn('accuracy', out2)
        out3 = eva.score(["A"], ["A. opt1", "B. opt2"])
        self.assertIn('error', out3)


if __name__ == '__main__':
    unittest.main()
