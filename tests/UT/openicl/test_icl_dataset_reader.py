import unittest
from datasets import Dataset, DatasetDict

from ais_bench.benchmark.openicl.icl_dataset_reader import DatasetReader, load_partial_dataset


class TestLoadPartialDataset(unittest.TestCase):
    def setUp(self):
        self.ds = Dataset.from_dict({"x": list(range(10))})

    def test_size_none_returns_all(self):
        """测试当size为None时返回完整数据集"""
        out = load_partial_dataset(self.ds, None)
        self.assertEqual(len(out), 10)

    def test_size_int_less_than_total(self):
        """测试当size为整数且小于总数时返回指定数量的数据"""
        out = load_partial_dataset(self.ds, 3)
        self.assertEqual(len(out), 3)

    def test_size_float_fraction(self):
        """测试当size为浮点数时按比例返回数据"""
        out = load_partial_dataset(self.ds, 0.3)
        self.assertEqual(len(out), 3)

    def test_size_str_slice(self):
        """测试当size为字符串切片时返回切片后的数据"""
        out = load_partial_dataset(self.ds, "[:4]")
        self.assertEqual(len(out), 4)

    def test_size_str_slice_error(self):
        """测试当size为无效字符串切片时返回完整数据集"""
        out = load_partial_dataset(self.ds, "[invalid]")
        self.assertEqual(len(out), 10)

    def test_size_invalid_type(self):
        """测试当size为无效类型时返回完整数据集"""
        out = load_partial_dataset(self.ds, [1, 2, 3])
        self.assertEqual(len(out), 10)

    def test_size_out_of_range_int(self):
        """测试当size为超出范围的整数时返回完整数据集"""
        out = load_partial_dataset(self.ds, 100)
        self.assertEqual(len(out), 10)


class TestDatasetReader(unittest.TestCase):
    def test_init_and_normalization(self):
        """测试DatasetReader初始化和数据集分割名称规范化"""
        train = Dataset.from_dict({"a": [1, 2], "b": [3, 4]})
        test = Dataset.from_dict({"a": [5, 6], "b": [7, 8]})
        dsd = DatasetDict({"train": train, "validation": test})
        reader = DatasetReader(dataset=dsd, input_columns=["a"], output_column="b", train_split="train", test_split="validation")
        self.assertIn("train", reader.dataset)
        self.assertIn("test", reader.dataset)
        self.assertGreaterEqual(len(reader), 2)
        _ = reader["train"]
        self.assertIn("DatasetReader({", repr(reader))

    def test_init_with_str_input_columns(self):
        """测试使用字符串格式的input_columns初始化DatasetReader"""
        ds = Dataset.from_dict({"x": [1, 2], "y": [3, 4]})
        reader = DatasetReader(dataset=ds, input_columns="x y", output_column="y")
        self.assertEqual(reader.input_columns, ["x", "y"])

    def test_init_with_max_tokens_column(self):
        """测试使用max_tokens_column初始化DatasetReader"""
        ds = Dataset.from_dict({"x": [1, 2], "max_tokens": [10, 20]})
        reader = DatasetReader(dataset=ds, input_columns=["x"], output_column=None, max_tokens_column="max_tokens")
        self.assertEqual(reader.max_tokens_column, "max_tokens")

    def test_get_max_out_len_none(self):
        """测试当max_out_len不在input_columns或features中时返回None"""
        ds = Dataset.from_dict({"x": [1, 2]})
        reader = DatasetReader(dataset=ds, input_columns=["x"], output_column=None)
        self.assertIsNone(reader.get_max_out_len())

    def test_get_max_out_len(self):
        """测试从input_columns中获取max_out_len值"""
        ds = Dataset.from_dict({"max_out_len": [10, 20]})
        reader = DatasetReader(dataset=ds, input_columns=["max_out_len"], output_column=None)
        self.assertEqual(reader.get_max_out_len(), [10, 20])


if __name__ == '__main__':
    unittest.main()


