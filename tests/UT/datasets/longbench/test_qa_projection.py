import unittest
from importlib import import_module
from unittest.mock import patch

from datasets import Dataset, DatasetDict


QA_ADAPTERS = (
    ('2wikim_qa', 'LongBench2wikimqaDataset'),
    ('dureader', 'LongBenchdureaderDataset'),
    ('hotpot_qa', 'LongBenchhotpotqaDataset'),
    ('multifieldqa_en', 'LongBenchmultifieldqa_enDataset'),
    ('multifieldqa_zh', 'LongBenchmultifieldqa_zhDataset'),
    ('musique', 'LongBenchmusiqueDataset'),
    ('narrative_qa', 'LongBenchnarrativeqaDataset'),
    ('qasper', 'LongBenchqasperDataset'),
)


def adapter_module(suffix):
    return import_module(f'ais_bench.benchmark.datasets.longbench.longbench_{suffix}')


class TestLongBenchQAProjection(unittest.TestCase):
    def test_preserves_load_arguments_and_projects_only_test_split(self):
        expected = [
            {'input': 'Second?', 'context': 'Context B', 'answers': ['B', 'b']},
            {'input': 'First?', 'context': 'Context A', 'answers': []},
        ]
        train = Dataset.from_list([{'untouched': 'training example'}])
        for name, class_name in QA_ADAPTERS:
            with self.subTest(adapter=name):
                module = adapter_module(name)
                original_test = Dataset.from_list([
                    dict(row, metadata={'id': i}) for i, row in enumerate(expected)
                ])
                splits = DatasetDict(train=train, test=original_test)
                with patch.object(module, 'get_data_path', return_value='/resolved') as resolve:
                    with patch.object(module, 'load_dataset', return_value=splits) as load:
                        result = getattr(module, class_name).load('/requested', name)

                resolve.assert_called_once_with('/requested')
                load.assert_called_once_with(
                    path='/resolved', name=name, data_dir='/resolved', trust_remote_code=True
                )
                self.assertIs(result, splits)
                self.assertIs(result['train'], train)
                self.assertIsNot(result['test'], original_test)
                self.assertEqual(result['test'].column_names, ['input', 'context', 'answers'])
                self.assertEqual(result['test'].to_list(), expected)
                self.assertIn('metadata', original_test.column_names)

    def test_empty_test_split_is_rebuilt_without_columns(self):
        module = adapter_module('qasper')
        empty = Dataset.from_dict({'input': [], 'context': [], 'answers': []})
        splits = DatasetDict(test=empty)
        with patch.object(module, 'get_data_path', return_value='/resolved'):
            with patch.object(module, 'load_dataset', return_value=splits):
                result = module.LongBenchqasperDataset.load('/requested', 'qasper')

        self.assertIsNot(result['test'], empty)
        self.assertEqual(len(result['test']), 0)
        self.assertEqual(result['test'].column_names, [])

    def test_missing_required_column_does_not_replace_test_split(self):
        module = adapter_module('hotpot_qa')
        incomplete = Dataset.from_dict({'input': ['Question?'], 'context': ['Context']})
        splits = DatasetDict(test=incomplete)
        with patch.object(module, 'get_data_path', return_value='/resolved'):
            with patch.object(module, 'load_dataset', return_value=splits):
                # datasets versions use either KeyError or ValueError for an absent column.
                with self.assertRaisesRegex((KeyError, ValueError), 'answers'):
                    module.LongBenchhotpotqaDataset.load('/requested', 'hotpot_qa')

        self.assertIs(splits['test'], incomplete)
