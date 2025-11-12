import unittest
from datasets import Dataset

from ais_bench.benchmark.openicl.icl_retriever.icl_random_retriever import RandomRetriever


class DummyDataset:
    def __init__(self):
        self.reader = type("R", (), {"output_column": "label"})()
        self.train = Dataset.from_dict({"text": ["t0", "t1", "t2", "t3"], "label": [0, 1, 0, 1]})
        self.test = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 0]})
        self.abbr = "dummy"


class TestRandomRetriever(unittest.TestCase):
    def test_retrieve_deterministic_seed(self):
        """测试RandomRetriever在相同种子下返回确定性结果"""
        ds = DummyDataset()
        r1 = RandomRetriever(ds, ice_num=2, seed=123)
        r2 = RandomRetriever(ds, ice_num=2, seed=123)
        out1 = r1.retrieve()
        out2 = r2.retrieve()
        self.assertEqual(out1, out2)
        self.assertEqual(len(out1), len(ds.test))
        self.assertEqual(len(out1[0]), 2)


if __name__ == '__main__':
    unittest.main()


