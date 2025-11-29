import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from mmengine.config import Config, ConfigDict

from ais_bench.benchmark.partitioners.naive import NaivePartitioner


class TestNaivePartitioner(unittest.TestCase):
    """Tests for NaivePartitioner class."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.out_dir, exist_ok=True)
        self.work_dir = os.path.join(self.temp_dir, "work")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_naive_partitioner_init(self):
        """Test NaivePartitioner initialization."""
        partitioner = NaivePartitioner(out_dir=self.out_dir, n=2)

        self.assertEqual(partitioner.out_dir, self.out_dir)
        self.assertEqual(partitioner.n, 2)

    def test_naive_partitioner_init_default_n(self):
        """Test NaivePartitioner initialization with default n."""
        partitioner = NaivePartitioner(out_dir=self.out_dir)

        self.assertEqual(partitioner.n, 1)

    @patch('ais_bench.benchmark.partitioners.naive.get_infer_output_path')
    @patch('ais_bench.benchmark.partitioners.naive.model_abbr_from_cfg')
    @patch('ais_bench.benchmark.partitioners.naive.dataset_abbr_from_cfg')
    def test_naive_partitioner_partition_new_tasks(self, mock_dataset_abbr, mock_model_abbr,
                                                    mock_get_path):
        """Test NaivePartitioner partition with new tasks."""
        partitioner = NaivePartitioner(out_dir=self.out_dir, n=1)

        model1 = ConfigDict({'abbr': 'model1', 'type': 'Model1'})
        dataset1 = ConfigDict({'abbr': 'dataset1', 'type': 'Dataset1'})

        mock_model_abbr.return_value = 'model1'
        mock_dataset_abbr.return_value = 'dataset1'
        mock_get_path.return_value = os.path.join(self.out_dir, 'model1_dataset1.json')

        model_dataset_combinations = [
            {
                'models': [model1],
                'datasets': [dataset1]
            }
        ]

        tasks = partitioner.partition(
            model_dataset_combinations=model_dataset_combinations,
            work_dir=self.work_dir,
            out_dir=self.out_dir
        )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(tasks[0]['models']), 1)
        self.assertEqual(len(tasks[0]['datasets'][0]), 1)

    @patch('ais_bench.benchmark.partitioners.naive.get_infer_output_path')
    @patch('ais_bench.benchmark.partitioners.naive.model_abbr_from_cfg')
    @patch('ais_bench.benchmark.partitioners.naive.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.partitioners.naive.os.path.exists')
    @patch('ais_bench.benchmark.partitioners.naive.os.stat')
    @patch('ais_bench.benchmark.partitioners.naive.os.getuid')
    def test_naive_partitioner_partition_skip_existing(self, mock_getuid, mock_stat,
                                                       mock_exists, mock_dataset_abbr,
                                                       mock_model_abbr, mock_get_path):
        """Test NaivePartitioner skips existing tasks."""
        partitioner = NaivePartitioner(out_dir=self.out_dir, n=1)

        model1 = ConfigDict({'abbr': 'model1', 'type': 'Model1'})
        dataset1 = ConfigDict({'abbr': 'dataset1', 'type': 'Dataset1'})

        mock_model_abbr.return_value = 'model1'
        mock_dataset_abbr.return_value = 'dataset1'
        output_file = os.path.join(self.out_dir, 'predictions', 'model1_dataset1.json')
        mock_get_path.return_value = output_file

        # File exists and is in predictions mode
        mock_exists.side_effect = lambda path: path == output_file
        mock_stat.return_value = MagicMock(st_uid=os.getuid())
        mock_getuid.return_value = os.getuid()

        model_dataset_combinations = [
            {
                'models': [model1],
                'datasets': [dataset1]
            }
        ]

        tasks = partitioner.partition(
            model_dataset_combinations=model_dataset_combinations,
            work_dir=self.work_dir,
            out_dir=os.path.join(self.out_dir, 'predictions')
        )

        # Task should be skipped
        self.assertEqual(len(tasks), 0)

    @patch('ais_bench.benchmark.partitioners.naive.get_infer_output_path')
    @patch('ais_bench.benchmark.partitioners.naive.model_abbr_from_cfg')
    @patch('ais_bench.benchmark.partitioners.naive.dataset_abbr_from_cfg')
    def test_naive_partitioner_partition_multiple_datasets(self, mock_dataset_abbr,
                                                            mock_model_abbr, mock_get_path):
        """Test NaivePartitioner with multiple datasets and n=2."""
        partitioner = NaivePartitioner(out_dir=self.out_dir, n=2)

        model1 = ConfigDict({'abbr': 'model1', 'type': 'Model1'})
        dataset1 = ConfigDict({'abbr': 'dataset1', 'type': 'Dataset1'})
        dataset2 = ConfigDict({'abbr': 'dataset2', 'type': 'Dataset2'})
        dataset3 = ConfigDict({'abbr': 'dataset3', 'type': 'Dataset3'})

        mock_model_abbr.return_value = 'model1'
        mock_dataset_abbr.side_effect = ['dataset1', 'dataset2', 'dataset3']
        mock_get_path.side_effect = [
            os.path.join(self.out_dir, 'model1_dataset1.json'),
            os.path.join(self.out_dir, 'model1_dataset2.json'),
            os.path.join(self.out_dir, 'model1_dataset3.json')
        ]

        model_dataset_combinations = [
            {
                'models': [model1],
                'datasets': [dataset1, dataset2, dataset3]
            }
        ]

        tasks = partitioner.partition(
            model_dataset_combinations=model_dataset_combinations,
            work_dir=self.work_dir,
            out_dir=self.out_dir
        )

        # With n=2, 3 datasets should create 2 tasks (2 datasets + 1 dataset)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(len(tasks[0]['datasets'][0]), 2)
        self.assertEqual(len(tasks[1]['datasets'][0]), 1)

    @patch('ais_bench.benchmark.partitioners.naive.get_infer_output_path')
    @patch('ais_bench.benchmark.partitioners.naive.model_abbr_from_cfg')
    @patch('ais_bench.benchmark.partitioners.naive.dataset_abbr_from_cfg')
    @patch('ais_bench.benchmark.partitioners.naive.os.path.exists')
    def test_naive_partitioner_partition_with_add_cfg(self, mock_exists, mock_dataset_abbr,
                                                      mock_model_abbr, mock_get_path):
        """Test NaivePartitioner with add_cfg."""
        partitioner = NaivePartitioner(out_dir=self.out_dir, n=1)

        model1 = ConfigDict({'abbr': 'model1', 'type': 'Model1'})
        dataset1 = ConfigDict({'abbr': 'dataset1', 'type': 'Dataset1'})

        mock_model_abbr.return_value = 'model1'
        mock_dataset_abbr.return_value = 'dataset1'
        mock_get_path.return_value = os.path.join(self.out_dir, 'model1_dataset1.json')
        mock_exists.return_value = False

        add_cfg = {'key': 'value'}

        model_dataset_combinations = [
            {
                'models': [model1],
                'datasets': [dataset1]
            }
        ]

        tasks = partitioner.partition(
            model_dataset_combinations=model_dataset_combinations,
            work_dir=self.work_dir,
            out_dir=self.out_dir,
            add_cfg=add_cfg
        )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]['key'], 'value')


if __name__ == "__main__":
    unittest.main()

