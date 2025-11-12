import unittest
from datasets import Dataset

from ais_bench.benchmark.openicl.icl_retriever.icl_fix_k_retriever import FixKRetriever
from ais_bench.benchmark.utils.logging.exceptions import AISBenchValueError


class DummyDataset:
    def __init__(self):
        self.reader = type("R", (), {"output_column": "label"})()
        self.train = Dataset.from_dict({"text": ["t0", "t1", "t2"], "label": [0, 1, 0]})
        self.test = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
        self.abbr = "dummy"


class TestFixKRetriever(unittest.TestCase):
    def test_retrieve_success(self):
        """测试FixKRetriever成功返回固定的索引列表"""
        ds = DummyDataset()
        r = FixKRetriever(ds, fix_id_list=[0, 2])
        out = r.retrieve()
        self.assertEqual(len(out), len(ds.test))
        self.assertTrue(all(o == [0, 2] for o in out))

    def test_retrieve_index_out_of_range(self):
        """测试FixKRetriever在索引超出范围时抛出异常"""
        ds = DummyDataset()
        with self.assertRaises(AISBenchValueError):
            FixKRetriever(ds, fix_id_list=[0, 99]).retrieve()


if __name__ == '__main__':
    unittest.main()


