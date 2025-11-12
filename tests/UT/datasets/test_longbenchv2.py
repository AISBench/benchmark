import unittest
from unittest.mock import patch

from datasets import Dataset

from ais_bench.benchmark.datasets.longbenchv2 import LongBenchv2Dataset, LongBenchv2Evaluator


class DummyDatasetSplit:
    def __init__(self, data):
        self._data = data
    def __len__(self):
        return len(next(iter(self._data.values())))
    def __getitem__(self, key):
        return self._data[key]


class DummyHFDS(dict):
    def __init__(self, data):
        super().__init__()
        self._train = DummyDatasetSplit(data)
    def __getitem__(self, key):
        if key == 'train':
            return self._train
        return super().__getitem__(key)
    def __setitem__(self, key, value):
        super().__setitem__(key, value)


class TestLongBenchv2(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.longbenchv2.get_data_path", return_value="data.json")
    @patch("ais_bench.benchmark.datasets.longbenchv2.load_dataset")
    def test_dataset(self, mock_load, mock_get_path):
        data = {
            'question': ['q'],
            'context': ['c'],
            'answer': ['A'],
            'choice_A': ['optA'],
            'choice_B': ['optB'],
            'choice_C': ['optC'],
            'choice_D': ['optD'],
            'difficulty': ['easy'],
            'length': ['short']
        }
        mock_load.return_value = DummyHFDS(data)
        ds = LongBenchv2Dataset.load("/any")
        self.assertIsInstance(ds['test'], Dataset)
        self.assertEqual(len(ds['test']), 1)

    def test_evaluator(self):
        eva = LongBenchv2Evaluator()
        samples = [{'difficulty': 'easy', 'length': 'short'}]
        out = eva.score(['A'], ['A'], samples)
        self.assertIn('accuracy', out)
        with self.assertRaises(ValueError):
            eva.score(['A'], ['A'], [])


if __name__ == '__main__':
    unittest.main()
