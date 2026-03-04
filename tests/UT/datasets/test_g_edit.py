import sys
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
from io import BytesIO
from PIL import Image
import base64

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from datasets import Dataset

from ais_bench.benchmark.datasets.g_edit import (
    GEditDataset,
    GEditSCJDGDataset,
    GEditPQJDGDataset,
    GEditEvaluator
)


class TestGEditEvaluator:
    def test_score(self):
        """测试score方法"""
        evaluator = GEditEvaluator()
        predictions = ["pred1", "pred2", "pred3"]
        references = ["ref1", "ref2", "ref3"]
        
        result = evaluator.score(predictions, references)
        
        assert "accuracy" in result
        assert "details" in result
        assert result["accuracy"] == 100.0
        assert len(result["details"]) == 3

    def test_score_empty(self):
        """测试score方法，空输入"""
        evaluator = GEditEvaluator()
        predictions = []
        references = []
        
        result = evaluator.score(predictions, references)
        
        assert "accuracy" in result
        assert "details" in result


class TestGEditDataset:
    def test_load_basic(self):
        """测试基本load方法"""
        mock_dataset = Dataset.from_list([
            {
                "input_image": Image.new('RGB', (100, 100), color='red'),
                "input_image_raw": Image.new('RGB', (100, 100), color='blue'),
                "instruction": "test instruction"
            }
        ])

        with patch('ais_bench.benchmark.datasets.g_edit.load_from_disk') as mock_load:
            with patch('ais_bench.benchmark.datasets.g_edit.get_data_path') as mock_get_path:
                mock_get_path.return_value = '/test/path'
                mock_load.return_value = mock_dataset

                ds = GEditDataset.__new__(GEditDataset)
                ds.task_state_manager = None
                ds.logger = MagicMock()
                ds.update_task_state = MagicMock()
                
                result = ds.load(path='/test/path')
                
                assert isinstance(result, Dataset)

    def test_load_with_split(self):
        """测试带数据集切分的load方法"""
        mock_dataset = Dataset.from_list([
            {
                "input_image": Image.new('RGB', (100, 100), color='red'),
                "input_image_raw": Image.new('RGB', (100, 100), color='blue'),
                "instruction": "test instruction"
            }
        ] * 10)

        with patch('ais_bench.benchmark.datasets.g_edit.load_from_disk') as mock_load:
            with patch('ais_bench.benchmark.datasets.g_edit.get_data_path') as mock_get_path:
                mock_get_path.return_value = '/test/path'
                mock_load.return_value = mock_dataset

                ds = GEditDataset.__new__(GEditDataset)
                ds.task_state_manager = None
                ds.logger = MagicMock()
                ds.update_task_state = MagicMock()
                
                result = ds.load(path='/test/path', split_count=2, split_index=0)
                
                assert isinstance(result, Dataset)

    def test_load_use_raw(self):
        """测试使用原始图片的load方法"""
        mock_dataset = Dataset.from_list([
            {
                "input_image": Image.new('RGB', (100, 100), color='red'),
                "input_image_raw": Image.new('RGB', (100, 100), color='blue'),
                "instruction": "test instruction"
            }
        ])

        with patch('ais_bench.benchmark.datasets.g_edit.load_from_disk') as mock_load:
            with patch('ais_bench.benchmark.datasets.g_edit.get_data_path') as mock_get_path:
                mock_get_path.return_value = '/test/path'
                mock_load.return_value = mock_dataset

                ds = GEditDataset.__new__(GEditDataset)
                ds.task_state_manager = None
                ds.logger = MagicMock()
                ds.update_task_state = MagicMock()
                
                result = ds.load(path='/test/path', use_raw=True)
                
                assert isinstance(result, Dataset)


class TestGEditSCJDGDataset:
    def test_get_dataset_class(self):
        """测试_get_dataset_class方法"""
        ds = GEditSCJDGDataset.__new__(GEditSCJDGDataset)
        result = ds._get_dataset_class()
        assert result == GEditDataset


class TestGEditPQJDGDataset:
    def test_get_dataset_class(self):
        """测试_get_dataset_class方法"""
        ds = GEditPQJDGDataset.__new__(GEditPQJDGDataset)
        result = ds._get_dataset_class()
        assert result == GEditDataset
