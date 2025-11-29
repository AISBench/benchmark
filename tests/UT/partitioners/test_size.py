import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from mmengine.config import ConfigDict

from ais_bench.benchmark.partitioners.size import SizePartitioner


class TestSizePartitioner(unittest.TestCase):
    """Tests for SizePartitioner class."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.out_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.out_dir, exist_ok=True)
        self.work_dir = os.path.join(self.temp_dir, "work")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_size_partitioner_init(self):
        """Test SizePartitioner initialization."""
        partitioner = SizePartitioner(
            out_dir=self.out_dir,
            max_task_size=1000,
            gen_task_coef=10,
            strategy='heuristic'
        )

        self.assertEqual(partitioner.max_task_size, 1000)
        self.assertEqual(partitioner.gen_task_coef, 10)
        self.assertEqual(partitioner.strategy, 'heuristic')

    def test_size_partitioner_init_invalid_strategy(self):
        """Test SizePartitioner initialization with invalid strategy."""
        with self.assertRaises(AssertionError):
            SizePartitioner(
                out_dir=self.out_dir,
                strategy='invalid'
            )

    @patch('ais_bench.benchmark.partitioners.size.get_infer_output_path')
    @patch('ais_bench.benchmark.partitioners.size.dataset_abbr_from_cfg')
    def test_size_partitioner_get_cost_cached(self, mock_dataset_abbr, mock_get_path):
        """Test get_cost with cached dataset size."""
        partitioner = SizePartitioner(out_dir=self.out_dir)
        partitioner._dataset_size = {'test_dataset': 100}

        dataset = ConfigDict({
            'abbr': 'test_dataset',
            'reader_cfg': {'test_range': ''},
            'infer_cfg': ConfigDict({
                'prompt_template': ConfigDict({
                    'template': 'Gen template'
                })
            })
        })

        mock_dataset_abbr.return_value = 'test_dataset'

        cost = partitioner.get_cost(dataset)

        # Should use cached size
        self.assertIsInstance(cost, int)
        self.assertGreater(cost, 0)

    @patch('ais_bench.benchmark.partitioners.size.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.partitioners.size.mmengine')
    @patch('ais_bench.benchmark.partitioners.size.dataset_abbr_from_cfg')
    def test_size_partitioner_get_cost_new_dataset(self, mock_dataset_abbr, mock_mmengine,
                                                    mock_build_dataset):
        """Test get_cost with new dataset (not cached)."""
        partitioner = SizePartitioner(out_dir=self.out_dir)

        dataset = ConfigDict({
            'abbr': 'new_dataset',
            'reader_cfg': {'test_range': ''},
            'infer_cfg': ConfigDict({
                'prompt_template': ConfigDict({
                    'template': 'Gen template'
                })
            })
        })

        mock_dataset_abbr.return_value = 'new_dataset'

        # Mock dataset object
        mock_dataset = MagicMock()
        mock_dataset.test = MagicMock()
        mock_dataset.test.__len__ = MagicMock(return_value=50)
        mock_build_dataset.return_value = mock_dataset

        mock_mmengine.mkdir_or_exist = MagicMock()
        mock_mmengine.dump = MagicMock()

        cost = partitioner.get_cost(dataset)

        # Should calculate and cache the size
        self.assertIsInstance(cost, int)
        self.assertIn('new_dataset', partitioner.dataset_size)

    def test_size_partitioner_get_factor_gen_template(self):
        """Test get_factor with generation template."""
        partitioner = SizePartitioner(out_dir=self.out_dir)

        dataset = ConfigDict({
            'infer_cfg': ConfigDict({
                'prompt_template': ConfigDict({
                    'template': 'This is a gen template'
                })
            }),
            'abbr': 'test_dataset'
        })

        factor = partitioner.get_factor(dataset)

        # Should return gen_task_coef for gen template
        self.assertEqual(factor, partitioner.gen_task_coef)

    def test_size_partitioner_get_factor_ppl_template(self):
        """Test get_factor with PPL template."""
        partitioner = SizePartitioner(out_dir=self.out_dir)

        dataset = ConfigDict({
            'infer_cfg': ConfigDict({
                'prompt_template': ConfigDict({
                    'template': {
                        'label1': 'template1',
                        'label2': 'template2',
                        'label3': 'template3'
                    }
                })
            }),
            'abbr': 'test_dataset'
        })

        factor = partitioner.get_factor(dataset)

        # Should return number of labels for PPL template
        self.assertEqual(factor, 3)

    @patch('ais_bench.benchmark.partitioners.size.dataset_abbr_from_cfg')
    def test_size_partitioner_get_factor_special_datasets(self, mock_dataset_abbr):
        """Test get_factor with special datasets (bbh, gsm8k, etc.)."""
        partitioner = SizePartitioner(out_dir=self.out_dir)

        dataset = ConfigDict({
            'infer_cfg': ConfigDict({
                'prompt_template': ConfigDict({
                    'template': 'Gen template'
                })
            }),
            'abbr': 'bbh_test'
        })

        mock_dataset_abbr.return_value = 'bbh_test'

        factor = partitioner.get_factor(dataset)

        # Should multiply by 10 for special datasets
        self.assertEqual(factor, partitioner.gen_task_coef * 10)

    @patch('ais_bench.benchmark.partitioners.size.get_infer_output_path')
    @patch('ais_bench.benchmark.partitioners.size.dataset_abbr_from_cfg')
    def test_size_partitioner_split_dataset(self, mock_dataset_abbr, mock_get_path):
        """Test split_dataset method."""
        partitioner = SizePartitioner(out_dir=self.out_dir, max_task_size=100)

        dataset = ConfigDict({
            'abbr': 'large_dataset',
            'reader_cfg': {'test_range': ''},
            'infer_cfg': ConfigDict({
                'prompt_template': ConfigDict({
                    'template': 'Gen template'
                })
            })
        })

        mock_dataset_abbr.return_value = 'large_dataset'
        partitioner._dataset_size = {'large_dataset': 250}

        # Mock get_cost to return tuple (size, factor) when get_raw_factors=True
        def mock_get_cost(dataset_cfg, get_raw_factors=False):
            if get_raw_factors:
                return 250, 1  # (size, factor)
            return 250  # Just size

        partitioner.get_cost = mock_get_cost

        splits = partitioner.split_dataset(dataset)

        # Should split into multiple parts
        self.assertGreater(len(splits), 1)
        for split in splits:
            self.assertIn('abbr', split)
            self.assertIn('reader_cfg', split)


if __name__ == "__main__":
    unittest.main()

