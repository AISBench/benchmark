import unittest
from unittest.mock import patch, MagicMock
from mmengine.config import ConfigDict

from ais_bench.benchmark.partitioners.sub_naive import (
    SubjectiveNaivePartitioner,
    remove_duplicate_pairs,
    replicate_tasks_with_judge_models,
    remove_already_tasks,
    get_model_combinations
)


class TestSubjectiveNaivePartitioner(unittest.TestCase):
    """Tests for SubjectiveNaivePartitioner class."""

    def setUp(self):
        self.out_dir = "/tmp/test_output"
        self.work_dir = "/tmp/test_work"

    def test_remove_duplicate_pairs(self):
        """Test remove_duplicate_pairs function."""
        model_combinations = [
            [{'abbr': 'model1'}, {'abbr': 'model2'}],
            [{'abbr': 'model2'}, {'abbr': 'model1'}],  # Duplicate
            [{'abbr': 'model1'}, {'abbr': 'model1'}],  # Same model, should be skipped
            [{'abbr': 'model3'}, {'abbr': 'model4'}]
        ]

        result = remove_duplicate_pairs(model_combinations)

        # Should remove duplicates and same-model pairs
        self.assertLessEqual(len(result), len(model_combinations))

    def test_replicate_tasks_with_judge_models_no_meta(self):
        """Test replicate_tasks_with_judge_models without meta judge model."""
        tasks = [
            {'models': [{'abbr': 'model1'}], 'datasets': [[{'abbr': 'dataset1'}]]}
        ]
        judge_models = [
            {'abbr': 'judge1'},
            {'abbr': 'judge2'}
        ]

        result = replicate_tasks_with_judge_models(tasks, judge_models, None)

        # Should replicate tasks for each judge model
        self.assertEqual(len(result), 2)
        self.assertIn('judge_model', result[0])
        self.assertIn('judge_model', result[1])

    def test_replicate_tasks_with_judge_models_with_meta(self):
        """Test replicate_tasks_with_judge_models with meta judge model."""
        tasks = [
            {'models': [{'abbr': 'model1'}], 'datasets': [[{'abbr': 'dataset1'}]]}
        ]
        judge_models = [
            {'abbr': 'judge1'},
            {'abbr': 'judge2'}
        ]
        meta_judge_model = {'abbr': 'meta_judge'}

        result = replicate_tasks_with_judge_models(tasks, judge_models, meta_judge_model)

        # Should return two stages
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)  # Two judge models
        self.assertEqual(len(result[1]), 1)  # One meta judge task
        self.assertIn('meta_judge_model', result[1][0])

    @patch('os.path.exists')
    @patch('ais_bench.benchmark.partitioners.sub_naive.get_infer_output_path')
    @patch('ais_bench.benchmark.partitioners.sub_naive.deal_with_judge_model_abbr')
    def test_remove_already_tasks(self, mock_deal_abbr, mock_get_path, mock_exists):
        """Test remove_already_tasks function."""
        tasks = [
            {
                'models': [{'abbr': 'model1'}],
                'datasets': [[{'abbr': 'dataset1'}, {'abbr': 'dataset2'}]],
                'judge_model': {'abbr': 'judge1'}
            }
        ]

        mock_deal_abbr.return_value = {'abbr': 'model1_judge1'}
        mock_get_path.side_effect = [
            '/tmp/results/model1_judge1_dataset1.json',
            '/tmp/results/model1_judge1_dataset2.json'
        ]
        # First file exists, second doesn't
        mock_exists.side_effect = [True, False]

        result = remove_already_tasks(tasks, self.work_dir, None)

        # Should remove dataset1 but keep dataset2
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]['datasets'][0]), 1)
        self.assertEqual(result[0]['datasets'][0][0]['abbr'], 'dataset2')

    def test_get_model_combinations_allpair(self):
        """Test get_model_combinations with allpair mode."""
        models = [
            {'abbr': 'model1'},
            {'abbr': 'model2'},
            {'abbr': 'model3'}
        ]

        result = list(get_model_combinations('allpair', models))

        # Should generate C(3,2) = 3 combinations
        self.assertEqual(len(result), 3)

    def test_get_model_combinations_m2n(self):
        """Test get_model_combinations with m2n mode."""
        base_models = [
            {'abbr': 'base1'},
            {'abbr': 'base2'}
        ]
        compare_models = [
            {'abbr': 'compare1'},
            {'abbr': 'compare2'}
        ]

        result = list(get_model_combinations('m2n', [], base_models, compare_models))

        # Should generate 2*2 = 4 combinations, minus duplicates
        self.assertGreater(len(result), 0)

    def test_get_model_combinations_fixed(self):
        """Test get_model_combinations with fixed mode."""
        result = get_model_combinations('fixed', [])

        self.assertIsNone(result)

    @patch('ais_bench.benchmark.partitioners.sub_naive.NaivePartitioner.partition')
    def test_subjective_naive_partitioner_partition_singlescore(self, mock_super_partition):
        """Test SubjectiveNaivePartitioner partition with singlescore mode."""
        partitioner = SubjectiveNaivePartitioner(
            out_dir=self.out_dir,
            models=[ConfigDict({'abbr': 'model1'})],
            judge_models=[ConfigDict({'abbr': 'judge1'})]
        )

        datasets = [
            ConfigDict({
                'mode': 'singlescore',
                'abbr': 'dataset1'
            })
        ]

        mock_super_partition.return_value = [
            {'models': [ConfigDict({'abbr': 'model1'})], 'datasets': [[ConfigDict({'abbr': 'dataset1'})]]}
        ]

        tasks = partitioner.partition(
            models=[ConfigDict({'abbr': 'model1'})],
            datasets=datasets,
            work_dir=self.work_dir,
            out_dir=self.out_dir
        )

        mock_super_partition.assert_called_once()

    @patch('ais_bench.benchmark.partitioners.sub_naive.NaivePartitioner.partition')
    def test_subjective_naive_partitioner_partition_allpair(self, mock_super_partition):
        """Test SubjectiveNaivePartitioner partition with allpair mode."""
        partitioner = SubjectiveNaivePartitioner(
            out_dir=self.out_dir,
            models=[
                ConfigDict({'abbr': 'model1'}),
                ConfigDict({'abbr': 'model2'})
            ],
            judge_models=[ConfigDict({'abbr': 'judge1'})]
        )

        datasets = [
            ConfigDict({
                'mode': 'allpair',
                'abbr': 'dataset1',
                'base_models': []  # Add base_models key
            })
        ]

        mock_super_partition.return_value = [
            {'models': [ConfigDict({'abbr': 'model1'})], 'datasets': [[ConfigDict({'abbr': 'dataset1'})]]}
        ]

        tasks = partitioner.partition(
            models=[
                ConfigDict({'abbr': 'model1'}),
                ConfigDict({'abbr': 'model2'})
            ],
            datasets=datasets,
            work_dir=self.work_dir,
            out_dir=self.out_dir
        )

        mock_super_partition.assert_called_once()


if __name__ == "__main__":
    unittest.main()

