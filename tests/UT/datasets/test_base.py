import unittest
from unittest.mock import patch, MagicMock

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.base import BaseDataset


class DummyDataset(BaseDataset):
    @staticmethod
    def load(**kwargs):
        # 返回一个简单的Dataset
        return Dataset.from_list([
            {"text": "a"},
            {"text": "b"},
            {"text": "c"},
        ])


class DummyDatasetDict(BaseDataset):
    @staticmethod
    def load(**kwargs):
        # 返回一个DatasetDict
        return DatasetDict({
            'train': Dataset.from_list([{"text": "train1"}, {"text": "train2"}]),
            'test': Dataset.from_list([{"text": "test1"}]),
        })


class DummyDatasetSingle(BaseDataset):
    """返回单个Dataset，用于测试Dataset类型的处理路径"""
    @staticmethod
    def load(**kwargs):
        return Dataset.from_list([
            {"text": "a"},
            {"text": "b"},
        ])
    
    def _init_reader(self, **kwargs):
        # 先正常初始化reader
        from ais_bench.benchmark.openicl.icl_dataset_reader import DatasetReader
        self.reader = DatasetReader(self.dataset, **kwargs)
        # 手动将dataset设置为Dataset类型，以测试Dataset类型的处理路径（覆盖46-62行）
        # 注意：正常情况下DatasetReader会将Dataset转换为DatasetDict，这里是为了测试覆盖
        self.reader.dataset = Dataset.from_list([
            {"text": "a"},
            {"text": "b"},
        ])


class TestBaseDataset(unittest.TestCase):
    def test_repeated_dataset_and_metadata(self):
        # n=2 确保重复采样，提供必需的reader_cfg参数
        ds = DummyDataset(
            reader_cfg={'input_columns': ['text'], 'output_column': None},
            k=1,
            n=2
        )
        # DatasetReader会将Dataset转换为DatasetDict，包含train和test
        # 验证是DatasetDict类型
        self.assertIsInstance(ds.dataset, DatasetDict)
        # 验证每个split的长度加倍（原始3条 * 2 = 6条）
        self.assertEqual(len(ds.dataset['train']), 6)
        self.assertEqual(len(ds.dataset['test']), 6)
        # 验证添加的元数据字段存在
        first = ds.dataset['test'][0]
        self.assertIn("subdivision", first)
        self.assertIn("idx", first)
    
    def test_repeated_dataset_with_dataset_type(self):
        """测试当reader.dataset是Dataset类型时的处理（覆盖46-62行）"""
        # 创建一个返回Dataset的类，并手动设置reader.dataset为Dataset
        ds = DummyDatasetSingle(
            reader_cfg={'input_columns': ['text'], 'output_column': None},
            k=1,
            n=3
        )
        # 验证dataset是Dataset类型
        self.assertIsInstance(ds.dataset, Dataset)
        # 验证长度是原始长度的n倍（2条 * 3 = 6条）
        self.assertEqual(len(ds.dataset), 6)
        # 验证添加了元数据字段
        first = ds.dataset[0]
        self.assertIn("subdivision", first)
        self.assertIn("idx", first)
    
    def test_train_property(self):
        """测试train属性（覆盖92行）"""
        ds = DummyDataset(
            reader_cfg={'input_columns': ['text'], 'output_column': None},
            k=1,
            n=1
        )
        train = ds.train
        self.assertIsInstance(train, Dataset)
        self.assertGreater(len(train), 0)
    
    def test_test_property(self):
        """测试test属性（覆盖96行）"""
        ds = DummyDataset(
            reader_cfg={'input_columns': ['text'], 'output_column': None},
            k=1,
            n=1
        )
        test = ds.test
        self.assertIsInstance(test, Dataset)
        self.assertGreater(len(test), 0)
    
    def test_repeated_dataset_with_large_batch_size(self):
        """测试大批量数据的批处理逻辑（覆盖批处理相关代码）"""
        # 创建一个较大的数据集来触发批处理逻辑
        large_data = [{"text": f"item_{i}"} for i in range(15000)]
        
        class LargeDataset(BaseDataset):
            @staticmethod
            def load(**kwargs):
                return Dataset.from_list(large_data)
        
        ds = LargeDataset(
            reader_cfg={'input_columns': ['text'], 'output_column': None},
            k=1,
            n=2
        )
        # 验证数据被正确处理
        self.assertIsInstance(ds.dataset, DatasetDict)
        # 验证长度正确（15000 * 2 = 30000）
        self.assertEqual(len(ds.dataset['train']), 30000)
        # 验证元数据存在
        first = ds.dataset['train'][0]
        self.assertIn("subdivision", first)
        self.assertIn("idx", first)


if __name__ == "__main__":
    unittest.main()
