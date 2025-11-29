import unittest
from unittest.mock import patch, MagicMock
from mmengine.config import ConfigDict

from ais_bench.benchmark.partitioners.sub_size import SubjectiveSizePartitioner


class TestSubjectiveSizePartitioner(unittest.TestCase):
    """Tests for SubjectiveSizePartitioner class."""

    def setUp(self):
        self.out_dir = "/tmp/test_output"
        self.work_dir = "/tmp/test_work"

    def test_subjective_size_partitioner_init(self):
        """Test SubjectiveSizePartitioner initialization."""
        partitioner = SubjectiveSizePartitioner(
            out_dir=self.out_dir,
            max_task_size=1000,
            gen_task_coef=10,
            strategy='heuristic'
        )

        self.assertEqual(partitioner.max_task_size, 1000)
        self.assertEqual(partitioner.gen_task_coef, 10)
        self.assertEqual(partitioner.strategy, 'heuristic')

    def test_subjective_size_partitioner_init_invalid_strategy(self):
        """Test SubjectiveSizePartitioner initialization with invalid strategy."""
        with self.assertRaises(AssertionError):
            SubjectiveSizePartitioner(
                out_dir=self.out_dir,
                strategy='invalid'
            )

    @patch('ais_bench.benchmark.partitioners.sub_size.get_infer_output_path')
    @patch('ais_bench.benchmark.partitioners.sub_size.dataset_abbr_from_cfg')
    def test_subjective_size_partitioner_get_cost(self, mock_dataset_abbr, mock_get_path):
        """Test get_cost method."""
        partitioner = SubjectiveSizePartitioner(out_dir=self.out_dir)
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

        self.assertIsInstance(cost, int)

    def test_subjective_size_partitioner_get_factor(self):
        """Test get_factor method."""
        partitioner = SubjectiveSizePartitioner(out_dir=self.out_dir)

        dataset = ConfigDict({
            'infer_cfg': ConfigDict({
                'prompt_template': ConfigDict({
                    'template': 'Gen template'
                })
            }),
            'abbr': 'test_dataset'
        })

        factor = partitioner.get_factor(dataset)

        self.assertEqual(factor, partitioner.gen_task_coef)

    @patch('ais_bench.benchmark.partitioners.sub_size.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.partitioners.sub_size.mmengine')
    @patch('ais_bench.benchmark.partitioners.sub_size.dataset_abbr_from_cfg')
    def test_subjective_size_partitioner_split_dataset(self, mock_dataset_abbr, mock_mmengine,
                                                        mock_build_dataset):
        """Test split_dataset method."""
        partitioner = SubjectiveSizePartitioner(out_dir=self.out_dir, max_task_size=100)

        dataset = ConfigDict({
            'abbr': 'large_dataset',
            'reader_cfg': {'test_range': ''}
        })

        mock_dataset_abbr.return_value = 'large_dataset'
        partitioner._dataset_size = {'large_dataset': 250}
        partitioner.get_cost = MagicMock(return_value=(250, 1))

        splits = partitioner.split_dataset(dataset)

        self.assertGreater(len(splits), 1)


if __name__ == "__main__":
    unittest.main()

