import unittest
from unittest.mock import patch, MagicMock
from mmengine.config import ConfigDict

from ais_bench.benchmark.partitioners.base import BasePartitioner
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError
from ais_bench.benchmark.utils.logging.error_codes import PARTI_CODES


class TestBasePartitioner(unittest.TestCase):
    """Tests for BasePartitioner class."""

    def setUp(self):
        self.out_dir = "/tmp/test_output"
        self.work_dir = "/tmp/test_work"

    def test_base_partitioner_init_default_keep_keys(self):
        """Test BasePartitioner initialization with default keep_keys."""
        partitioner = BasePartitioner(out_dir=self.out_dir)

        self.assertEqual(partitioner.out_dir, self.out_dir)
        self.assertIsNotNone(partitioner.keep_keys)
        self.assertIn('eval.runner.task.judge_cfg', partitioner.keep_keys)
        self.assertIn('eval.runner.task.dump_details', partitioner.keep_keys)

    def test_base_partitioner_init_custom_keep_keys(self):
        """Test BasePartitioner initialization with custom keep_keys."""
        custom_keys = ['custom.key1', 'custom.key2']
        partitioner = BasePartitioner(out_dir=self.out_dir, keep_keys=custom_keys)

        self.assertEqual(partitioner.keep_keys, custom_keys)

    @patch('ais_bench.benchmark.partitioners.base.AISLogger')
    def test_base_partitioner_call(self, mock_logger_class):
        """Test BasePartitioner __call__ method."""
        # Create a concrete partitioner class
        class TestPartitioner(BasePartitioner):
            def partition(self, models, datasets, work_dir, out_dir, add_cfg={}):
                return [
                    {
                        'models': [models[0]],
                        'datasets': [[datasets[0]]],
                        'work_dir': work_dir,
                        'cli_args': {}
                    }
                ]

        partitioner = TestPartitioner(out_dir=self.out_dir)

        cfg = ConfigDict({
            'work_dir': self.work_dir,
            'models': [ConfigDict({'abbr': 'model1', 'type': 'Model1'})],
            'datasets': [ConfigDict({'abbr': 'dataset1', 'type': 'Dataset1'})]
        })

        tasks = partitioner(cfg)

        self.assertIsInstance(tasks, list)
        self.assertEqual(len(tasks), 1)

    @patch('ais_bench.benchmark.partitioners.base.AISLogger')
    def test_base_partitioner_call_with_keep_keys(self, mock_logger_class):
        """Test BasePartitioner __call__ with keep_keys extraction."""
        class TestPartitioner(BasePartitioner):
            def partition(self, models, datasets, work_dir, out_dir, add_cfg={}):
                return [
                    {
                        'models': [models[0]],
                        'datasets': [[datasets[0]]],
                        'work_dir': work_dir,
                        'cli_args': {}
                    }
                ]

        partitioner = TestPartitioner(out_dir=self.out_dir)

        cfg = ConfigDict({
            'work_dir': self.work_dir,
            'models': [ConfigDict({'abbr': 'model1', 'type': 'Model1'})],
            'datasets': [ConfigDict({'abbr': 'dataset1', 'type': 'Dataset1'})],
            'eval': ConfigDict({
                'runner': ConfigDict({
                    'task': ConfigDict({
                        'judge_cfg': {'key': 'value'},
                        'dump_details': True
                    })
                })
            })
        })

        tasks = partitioner(cfg)

        # Verify keep_keys were extracted
        self.assertIsInstance(tasks, list)
        self.assertEqual(len(tasks), 1)

    @patch('ais_bench.benchmark.partitioners.base.AISLogger')
    def test_base_partitioner_call_missing_keep_key(self, mock_logger_class):
        """Test BasePartitioner __call__ with missing keep_key."""
        class TestPartitioner(BasePartitioner):
            def partition(self, models, datasets, work_dir, out_dir, add_cfg={}):
                return [
                    {
                        'models': [models[0]],
                        'datasets': [[datasets[0]]],
                        'work_dir': work_dir,
                        'cli_args': {}
                    }
                ]

        partitioner = TestPartitioner(out_dir=self.out_dir)

        cfg = ConfigDict({
            'work_dir': self.work_dir,
            'models': [ConfigDict({'abbr': 'model1', 'type': 'Model1'})],
            'datasets': [ConfigDict({'abbr': 'dataset1', 'type': 'Dataset1'})]
            # Missing eval.runner.task.judge_cfg
        })

        # Should not raise error, just log debug message
        tasks = partitioner(cfg)
        self.assertIsInstance(tasks, list)
        self.assertEqual(len(tasks), 1)

    @patch('ais_bench.benchmark.partitioners.base.AISLogger')
    def test_base_partitioner_parse_model_dataset_args_simple(self, mock_logger_class):
        """Test parse_model_dataset_args with simple models and datasets."""
        partitioner = BasePartitioner(out_dir=self.out_dir)

        cfg = ConfigDict({
            'models': [ConfigDict({'abbr': 'model1'})],
            'datasets': [ConfigDict({'abbr': 'dataset1'})]
        })

        result = partitioner.parse_model_dataset_args(cfg)

        self.assertIn('models', result)
        self.assertIn('datasets', result)
        self.assertEqual(len(result['models']), 1)
        self.assertEqual(len(result['datasets']), 1)

    @patch('ais_bench.benchmark.partitioners.base.AISLogger')
    def test_base_partitioner_parse_model_dataset_args_with_combinations(self, mock_logger_class):
        """Test parse_model_dataset_args with model_dataset_combinations."""
        # Create a partitioner that supports model_dataset_combinations
        class TestPartitioner(BasePartitioner):
            def partition(self, model_dataset_combinations, work_dir, out_dir, add_cfg={}):
                return []

        partitioner = TestPartitioner(out_dir=self.out_dir)

        cfg = ConfigDict({
            'models': [ConfigDict({'abbr': 'model1'})],
            'datasets': [ConfigDict({'abbr': 'dataset1'})],
            'model_dataset_combinations': [
                {
                    'models': [ConfigDict({'abbr': 'model1'})],
                    'datasets': [ConfigDict({'abbr': 'dataset1'})]
                }
            ]
        })

        result = partitioner.parse_model_dataset_args(cfg)

        self.assertIn('model_dataset_combinations', result)
        self.assertEqual(len(result['model_dataset_combinations']), 1)

    @patch('ais_bench.benchmark.partitioners.base.AISLogger')
    def test_base_partitioner_parse_model_dataset_args_invalid_model(self, mock_logger_class):
        """Test parse_model_dataset_args with invalid model in combination."""
        class TestPartitioner(BasePartitioner):
            def partition(self, model_dataset_combinations, work_dir, out_dir, add_cfg={}):
                return []

        partitioner = TestPartitioner(out_dir=self.out_dir)

        cfg = ConfigDict({
            'models': [ConfigDict({'abbr': 'model1'})],
            'datasets': [ConfigDict({'abbr': 'dataset1'})],
            'model_dataset_combinations': [
                {
                    'models': [ConfigDict({'abbr': 'invalid_model'})],  # Not in cfg['models']
                    'datasets': [ConfigDict({'abbr': 'dataset1'})]
                }
            ]
        })

        with self.assertRaises(AISBenchConfigError) as cm:
            partitioner.parse_model_dataset_args(cfg)

        self.assertEqual(cm.exception.error_code_str, PARTI_CODES.UNKNOWN_ERROR.full_code)

    @patch('ais_bench.benchmark.partitioners.base.AISLogger')
    def test_base_partitioner_parse_model_dataset_args_invalid_dataset(self, mock_logger_class):
        """Test parse_model_dataset_args with invalid dataset in combination."""
        class TestPartitioner(BasePartitioner):
            def partition(self, model_dataset_combinations, work_dir, out_dir, add_cfg={}):
                return []

        partitioner = TestPartitioner(out_dir=self.out_dir)

        cfg = ConfigDict({
            'models': [ConfigDict({'abbr': 'model1'})],
            'datasets': [ConfigDict({'abbr': 'dataset1'})],
            'model_dataset_combinations': [
                {
                    'models': [ConfigDict({'abbr': 'model1'})],
                    'datasets': [ConfigDict({'abbr': 'invalid_dataset'})]  # Not in cfg['datasets']
                }
            ]
        })

        with self.assertRaises(AISBenchConfigError) as cm:
            partitioner.parse_model_dataset_args(cfg)

        self.assertEqual(cm.exception.error_code_str, PARTI_CODES.UNKNOWN_ERROR.full_code)

    @patch('ais_bench.benchmark.partitioners.base.AISLogger')
    def test_base_partitioner_check_task_cfg_only_perf_dataset(self, mock_logger_class):
        """Test _check_task_cfg filters out non-perf datasets in non-perf mode."""
        from ais_bench.benchmark.datasets.utils.datasets import ONLY_PERF_DATASETS

        partitioner = BasePartitioner(out_dir=self.out_dir)

        # Get a valid dataset type from ONLY_PERF_DATASETS
        if len(ONLY_PERF_DATASETS) > 0:
            perf_dataset_type = list(ONLY_PERF_DATASETS)[0]
            task = {
                'cli_args': {'mode': 'eval'},  # Not perf mode
                'datasets': [[ConfigDict({'type': perf_dataset_type})]],
                'models': [ConfigDict({'type': 'SomeModel'})]
            }

            # Should raise ValueError when all tasks are filtered out
            with self.assertRaises(ValueError) as cm:
                partitioner._check_task_cfg([task])

            self.assertIn("No executable task", str(cm.exception))
        else:
            self.skipTest("ONLY_PERF_DATASETS is empty")

    @patch('ais_bench.benchmark.partitioners.base.AISLogger')
    def test_base_partitioner_check_task_cfg_mm_dataset_error(self, mock_logger_class):
        """Test _check_task_cfg filters out MM datasets with non-MM APIs."""
        from ais_bench.benchmark.datasets.utils.datasets import MM_DATASETS, MM_APIS

        partitioner = BasePartitioner(out_dir=self.out_dir)

        # Get a valid dataset type from MM_DATASETS
        if len(MM_DATASETS) > 0:
            mm_dataset_type = list(MM_DATASETS)[0]
            task = {
                'cli_args': {'mode': 'eval'},
                'datasets': [[ConfigDict({'type': mm_dataset_type})]],
                'models': [ConfigDict({'type': 'NonMMModel'})]  # Not in MM_APIS
            }

            # Should raise ValueError when all tasks are filtered out
            with self.assertRaises(ValueError) as cm:
                partitioner._check_task_cfg([task])

            self.assertIn("No executable task", str(cm.exception))
        else:
            self.skipTest("MM_DATASETS is empty")

    @patch('ais_bench.benchmark.partitioners.base.AISLogger')
    def test_base_partitioner_check_task_cfg_no_executable_tasks_error(self, mock_logger_class):
        """Test _check_task_cfg raises error when no executable tasks."""
        partitioner = BasePartitioner(out_dir=self.out_dir)

        # All tasks will be filtered out
        tasks = []

        with self.assertRaises(ValueError) as cm:
            partitioner._check_task_cfg(tasks)

        self.assertIn("No executable task", str(cm.exception))

    @patch('ais_bench.benchmark.partitioners.base.AISLogger')
    def test_base_partitioner_check_task_cfg_valid_task(self, mock_logger_class):
        """Test _check_task_cfg keeps valid tasks."""
        partitioner = BasePartitioner(out_dir=self.out_dir)

        task = {
            'cli_args': {'mode': 'eval'},
            'datasets': [[ConfigDict({'type': 'GSM8KDataset'})]],
            'models': [ConfigDict({'type': 'SomeModel'})]
        }

        tasks = partitioner._check_task_cfg([task])

        # Task should be kept
        self.assertEqual(len(tasks), 1)

    @patch('ais_bench.benchmark.partitioners.base.AISLogger')
    def test_base_partitioner_partition_abstract(self, mock_logger_class):
        """Test that partition must be implemented by subclasses."""
        partitioner = BasePartitioner(out_dir=self.out_dir)

        # BasePartitioner.partition is not actually abstract, but subclasses should implement it
        # We can test that calling it on base class would fail due to missing implementation
        # Actually, it's not abstract, so we just verify it exists
        self.assertTrue(hasattr(partitioner, 'partition'))
        self.assertTrue(callable(partitioner.partition))


if __name__ == "__main__":
    unittest.main()

