import unittest
from datasets import Dataset

from ais_bench.benchmark.openicl.icl_retriever.icl_zero_retriever import ZeroRetriever


class DummyDataset:
    def __init__(self):
        self.reader = type("R", (), {"output_column": "label"})()
        self.train = Dataset.from_dict({"text": ["t0"], "label": [0]})
        self.test = Dataset.from_dict({"text": ["a", "b", "c"], "label": [0, 1, 0]})
        self.abbr = "dummy"


class TestZeroRetriever(unittest.TestCase):
    def test_retrieve_empty_lists(self):
        """测试ZeroRetriever返回空列表列表"""
        ds = DummyDataset()
        r = ZeroRetriever(ds)
        out = r.retrieve()
        self.assertEqual(out, [[] for _ in range(len(ds.test))])


if __name__ == '__main__':
    unittest.main()


