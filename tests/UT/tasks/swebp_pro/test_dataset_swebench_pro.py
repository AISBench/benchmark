"""Unit tests for ais_bench.benchmark.datasets.swebench_pro"""
import unittest
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ais_bench.benchmark.datasets import swebench_pro as dataset_module


class TestParquetShardsForSplit(unittest.TestCase):
    """Test _parquet_shards_for_split function."""

    @classmethod
    def setUpClass(cls):
        cls.dataset_module = dataset_module

    def test_finds_shards_for_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "test-00000-of-00002.parquet").write_text("")
            (root / "test-00001-of-00002.parquet").write_text("")
            result = self.dataset_module._parquet_shards_for_split(root, "test")
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 2)

    def test_returns_none_for_missing_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = self.dataset_module._parquet_shards_for_split(root, "nonexistent")
            self.assertIsNone(result)

    def test_finds_shards_in_data_subdir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data").mkdir()
            (root / "data" / "test-00000-of-00002.parquet").write_text("")
            result = self.dataset_module._parquet_shards_for_split(root, "test")
            self.assertIsNotNone(result)
            self.assertEqual(len(result), 1)


class TestParquetDataFilesForRoot(unittest.TestCase):
    """Test _parquet_data_files_for_root function."""

    @classmethod
    def setUpClass(cls):
        cls.dataset_module = dataset_module

    @patch.object(Path, 'is_file', return_value=True)
    def test_returns_file_dict_for_file_path(self, mock_is_file):
        test_path = Path("/data/test.parquet")
        result = self.dataset_module._parquet_data_files_for_root(test_path, "test")
        self.assertEqual(result, {"test": str(test_path)})

    @patch.object(Path, 'is_file', return_value=False)
    def test_returns_none_when_no_shards(self, mock_is_file):
        with patch.object(self.dataset_module, '_parquet_shards_for_split', return_value=None):
            result = self.dataset_module._parquet_data_files_for_root(Path("/data"), "test")
            self.assertIsNone(result)

    @patch.object(Path, 'is_file', return_value=False)
    def test_returns_shards_when_found(self, mock_is_file):
        with patch.object(self.dataset_module, '_parquet_shards_for_split', return_value=["shard1.parquet", "shard2.parquet"]):
            result = self.dataset_module._parquet_data_files_for_root(Path("/data"), "test")
            self.assertEqual(result["test"], ["shard1.parquet", "shard2.parquet"])


class TestFilterInstancesMethod(unittest.TestCase):
    """Test SWEBenchProDataset.filter_instances method."""

    @classmethod
    def setUpClass(cls):
        cls.dataset_module = dataset_module

    def test_filters_by_instance_id_pattern(self):
        dataset = self.dataset_module.SWEBenchProDataset.__new__(self.dataset_module.SWEBenchProDataset)
        dataset.logger = MagicMock()
        instances = [
            {"instance_id": "id1"},
            {"instance_id": "id2"},
            {"instance_id": "other"},
        ]
        result = dataset.filter_instances(instances, filter_spec="^id")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["instance_id"], "id1")
        self.assertEqual(result[1]["instance_id"], "id2")

    def test_returns_all_when_no_filter(self):
        dataset = self.dataset_module.SWEBenchProDataset.__new__(self.dataset_module.SWEBenchProDataset)
        dataset.logger = MagicMock()
        instances = [
            {"instance_id": "id1"},
            {"instance_id": "id2"},
        ]
        result = dataset.filter_instances(instances, filter_spec="")
        self.assertEqual(len(result), 2)

    def test_shuffles_when_enabled(self):
        dataset = self.dataset_module.SWEBenchProDataset.__new__(self.dataset_module.SWEBenchProDataset)
        dataset.logger = MagicMock()
        instances = [
            {"instance_id": "id1"},
            {"instance_id": "id2"},
            {"instance_id": "id3"},
        ]
        result = dataset.filter_instances(instances, filter_spec="", shuffle=True)
        self.assertEqual(len(result), 3)


class TestDatasetLoad(unittest.TestCase):
    """Test SWEBenchProDataset.load method."""

    @classmethod
    def setUpClass(cls):
        cls.dataset_module = dataset_module

    def test_raises_for_invalid_dataset_name(self):
        dataset = self.dataset_module.SWEBenchProDataset.__new__(self.dataset_module.SWEBenchProDataset)
        dataset.logger = MagicMock()

        with self.assertRaises(Exception):
            dataset.load("invalid_name", "test", "", False)

    @patch('ais_bench.benchmark.datasets.swebench_pro.load_dataset')
    def test_raises_when_hf_load_fails(self, mock_load_dataset):
        dataset = self.dataset_module.SWEBenchProDataset.__new__(self.dataset_module.SWEBenchProDataset)
        dataset.logger = MagicMock()

        mock_load_dataset.side_effect = Exception("HF Error")

        with self.assertRaises(Exception):
            dataset.load("mini", path="", split="test", filter_spec="", shuffle=False)

    @patch('ais_bench.benchmark.datasets.swebench_pro.get_data_path')
    @patch('ais_bench.benchmark.datasets.swebench_pro.load_dataset')
    def test_raises_when_local_path_resolve_fails(self, mock_load_dataset, mock_get_data):
        dataset = self.dataset_module.SWEBenchProDataset.__new__(self.dataset_module.SWEBenchProDataset)
        dataset.logger = MagicMock()

        mock_get_data.side_effect = Exception("Path Error")

        with self.assertRaises(Exception):
            dataset.load("mini", "test", "/local/path", False)

    @patch('ais_bench.benchmark.datasets.swebench_pro.get_data_path')
    def test_raises_when_no_parquet_found(self, mock_get_data):
        dataset = self.dataset_module.SWEBenchProDataset.__new__(self.dataset_module.SWEBenchProDataset)
        dataset.logger = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_data.return_value = tmpdir

            with self.assertRaises(Exception):
                dataset.load("mini", "test", tmpdir, False)

    @patch('ais_bench.benchmark.datasets.swebench_pro.get_data_path')
    @patch('ais_bench.benchmark.datasets.swebench_pro.load_dataset')
    def test_raises_when_local_parquet_load_fails(self, mock_load_dataset, mock_get_data):
        dataset = self.dataset_module.SWEBenchProDataset.__new__(self.dataset_module.SWEBenchProDataset)
        dataset.logger = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_get_data.return_value = tmpdir
            Path(tmpdir) / "test-00000-of-00001.parquet"
            (Path(tmpdir) / "test-00000-of-00001.parquet").write_text("")
            mock_load_dataset.side_effect = Exception("Load Error")

            with self.assertRaises(Exception):
                dataset.load("mini", "test", tmpdir, False)


if __name__ == '__main__':
    unittest.main()
