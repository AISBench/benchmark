import unittest
from unittest.mock import patch, MagicMock
import pytest

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.base import BaseDataset, BaseJDGDataset


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

    def test_init_with_task_state_manager(self):
        """测试使用task_state_manager初始化"""
        mock_task_state_manager = MagicMock()
        ds = DummyDataset(
            reader_cfg={'input_columns': ['text'], 'output_column': None},
            k=1,
            n=1,
            task_state_manager=mock_task_state_manager
        )
        self.assertEqual(ds.task_state_manager, mock_task_state_manager)

    def test_update_task_state_with_manager(self):
        """测试使用task_state_manager更新状态"""
        mock_task_state_manager = MagicMock()
        ds = DummyDataset(
            reader_cfg={'input_columns': ['text'], 'output_column': None},
            k=1,
            n=1,
            task_state_manager=mock_task_state_manager
        )
        state = {'status': 'processing'}
        ds.update_task_state(state)
        mock_task_state_manager.update_task_state.assert_called_once_with(state)

    def test_update_task_state_without_manager(self):
        """测试没有task_state_manager时更新状态"""
        ds = DummyDataset(
            reader_cfg={'input_columns': ['text'], 'output_column': None},
            k=1,
            n=1
        )
        # 不应抛出异常
        ds.update_task_state({'status': 'processing'})

    def test_init_with_abbr(self):
        """测试使用abbr参数初始化"""
        ds = DummyDataset(
            reader_cfg={'input_columns': ['text'], 'output_column': None},
            k=1,
            n=1,
            abbr='custom_abbr'
        )
        self.assertEqual(ds.abbr, 'custom_abbr')

    def test_init_k_greater_than_n_raises_error(self):
        """测试k > n时抛出异常"""
        from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError
        with self.assertRaises(ParameterValueError):
            DummyDataset(
                reader_cfg={'input_columns': ['text'], 'output_column': None},
                k=5,
                n=3
            )

    def test_init_k_list_greater_than_n_raises_error(self):
        """测试k为列表且最大值 > n时抛出异常"""
        from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError
        with self.assertRaises(ParameterValueError):
            DummyDataset(
                reader_cfg={'input_columns': ['text'], 'output_column': None},
                k=[1, 2, 5],
                n=3
            )

    def test_repeated_dataset_with_dataset_dict(self):
        """测试DatasetDict类型的repeated_dataset处理"""
        ds = DummyDatasetDict(
            reader_cfg={'input_columns': ['text'], 'output_column': None},
            k=1,
            n=2
        )
        self.assertIsInstance(ds.dataset, DatasetDict)
        # 验证每个split都被正确处理
        self.assertEqual(len(ds.dataset['train']), 4)  # 2 * 2
        self.assertEqual(len(ds.dataset['test']), 2)   # 1 * 2
        # 验证元数据
        first_train = ds.dataset['train'][0]
        self.assertIn("subdivision", first_train)
        self.assertIn("idx", first_train)


class TestBaseJDGDataset(unittest.TestCase):
    def test_init_org_datasets_instance(self):
        """测试_init_org_datasets_instance方法"""
        class DummyJDGDataset(BaseJDGDataset):
            def _get_dataset_class(self):
                return DummyDataset
            def _load_from_predictions(self, prediction_path):
                return []

        with patch.object(DummyJDGDataset, 'load') as mock_load:
            mock_load.return_value = Dataset.from_list([{"text": "a"}])
            ds = DummyJDGDataset(
                reader_cfg={'input_columns': ['text'], 'output_column': None},
                k=1,
                n=1
            )
            self.assertIsNotNone(ds.dataset_instance)

    def test_process_single_item(self):
        """测试_process_single_item方法"""
        class DummyJDGDataset(BaseJDGDataset):
            def _get_dataset_class(self):
                return DummyDataset
            def _load_from_predictions(self, prediction_path):
                return []

        dataset_content = Dataset.from_list([
            {"text": "question1", "answer": "A"},
            {"text": "question2", "answer": "B"}
        ])
        pred_item = {"id": 0, "prediction": "predicted_answer", "uuid": "test_uuid"}

        with patch.object(DummyJDGDataset, 'load') as mock_load:
            mock_load.return_value = Dataset.from_list([{"text": "a"}])
            ds = DummyJDGDataset(
                reader_cfg={'input_columns': ['text'], 'output_column': None},
                k=1,
                n=1
            )
            result = ds._process_single_item(dataset_content, pred_item)
            self.assertEqual(result["model_answer"], "predicted_answer")
            self.assertEqual(result["model_pred_uuid"], "test_uuid")

    def test_modify_dataset_item(self):
        """测试_modify_dataset_item方法"""
        class DummyJDGDataset(BaseJDGDataset):
            def _get_dataset_class(self):
                return DummyDataset
            def _load_from_predictions(self, prediction_path):
                return []

        with patch.object(DummyJDGDataset, 'load') as mock_load:
            mock_load.return_value = Dataset.from_list([{"text": "a"}])
            ds = DummyJDGDataset(
                reader_cfg={'input_columns': ['text'], 'output_column': None},
                k=1,
                n=1
            )
            dataset_item = {"text": "question", "answer": "A"}
            pred_item = {"prediction": "predicted_answer"}
            ds._modify_dataset_item(dataset_item, pred_item)
            self.assertEqual(dataset_item["model_answer"], "predicted_answer")

    def test_load_with_predictions(self):
        """测试load方法处理predictions"""
        class DummyJDGDataset(BaseJDGDataset):
            def _get_dataset_class(self):
                return DummyDataset
            def _load_from_predictions(self, prediction_path):
                return [{"id": 0, "prediction": "pred1", "uuid": "uuid1"}]

        with patch.object(DummyJDGDataset, '_process_predictions') as mock_process:
            mock_process.return_value = Dataset.from_list([{"text": "result"}])
            
            with patch.object(DummyJDGDataset, '__init__', lambda self, *args, **kwargs: None):
                ds = DummyJDGDataset.__new__(DummyJDGDataset)
                ds.dataset_instance = MagicMock()
                ds.dataset_instance.dataset = {"test": Dataset.from_list([{"text": "test"}])}
                ds.task_state_manager = None
                ds.logger = MagicMock()
                
                result = ds.load(predictions_path="/test/predictions.jsonl")
                self.assertIsInstance(result, Dataset)


if __name__ == "__main__":
    unittest.main()
