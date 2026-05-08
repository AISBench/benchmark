import unittest

from datasets import Dataset

from ais_bench.benchmark.datasets.vbench import VBenchDataset


class TestVBenchDataset(unittest.TestCase):

    def test_load_returns_placeholder(self):
        ds = VBenchDataset.load(path='/any')
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]['dummy'], 0)


if __name__ == '__main__':
    unittest.main()
