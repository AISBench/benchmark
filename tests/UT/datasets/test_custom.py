import unittest
from unittest.mock import patch, mock_open, MagicMock

import numpy as np

from datasets import Dataset

from ais_bench.benchmark.datasets.custom import (
    OptionSimAccEvaluator,
    CustomDataset,
    stringfy_types,
    make_mcq_gen_config,
    make_qa_gen_config,
    parse_example_dataset,
    make_custom_dataset_config,
    get_max_token_list_from_meta_json_file,
)


class DummyCls:
    pass


class DummyL:
    @staticmethod
    def distance(a, b):
        # 简化的编辑距离：只考虑长度差
        # 为了让测试可预测，我们返回一个基于字符串内容的固定值
        distances = {
            ('closest', 'A'): 7,
            ('closest', 'alpha'): 5,
            ('closest', 'A. alpha'): 6,
            ('closest', 'B'): 7,
            ('closest', 'beta'): 6,
            ('closest', 'B. beta'): 7,
        }
        return distances.get((a, b), abs(len(a) - len(b)))


class TestOptionSimAccEvaluator(unittest.TestCase):
    def test_init_valid(self):
        evaluator = OptionSimAccEvaluator(['A', 'B'])
        self.assertEqual(evaluator.options, ['A', 'B'])

    def test_init_invalid(self):
        with self.assertRaises(ValueError):
            OptionSimAccEvaluator(['AA'])

    def test_match_any_label_branches(self):
        evaluator = OptionSimAccEvaluator(['A', 'B'])
        item = {'A': 'apple', 'B': 'banana'}
        # direct match
        self.assertEqual(evaluator.match_any_label(' A ', item), 'A')
        # first_option_postprocess branch
        with patch('ais_bench.benchmark.utils.postprocess.text_postprocessors.first_option_postprocess', return_value='B'), \
             patch('rapidfuzz.distance.Levenshtein', DummyL):
            self.assertEqual(evaluator.match_any_label('something', item), 'B')
        # substring branch
        with patch('ais_bench.benchmark.utils.postprocess.text_postprocessors.first_option_postprocess', return_value=''), \
             patch('rapidfuzz.distance.Levenshtein', DummyL):
            self.assertEqual(evaluator.match_any_label('this has banana inside', item), 'B')
        # distance branch
        with patch('ais_bench.benchmark.utils.postprocess.text_postprocessors.first_option_postprocess', return_value=''), \
             patch('rapidfuzz.distance.Levenshtein', DummyL):
            self.assertEqual(evaluator.match_any_label('closest', {'A': 'alpha', 'B': 'beta'}), 'A')

    def test_score_with_patch(self):
        evaluator = OptionSimAccEvaluator(['A'])
        with patch.object(evaluator, 'match_any_label', return_value='A'):
            out = evaluator.score(['pred'], ['A'], [{'A': 'opt'}])
            self.assertEqual(out['accuracy'], 100)


class TestCustomDataset(unittest.TestCase):
    @patch('ais_bench.benchmark.datasets.custom.get_meta_json', return_value={'output_config': {"method": "uniform", "params": {"min_value": 1, "max_value": 1}}, 'sampling_mode': 'default', 'request_count': 0})
    @patch('ais_bench.benchmark.datasets.custom.get_data_path', return_value='/root')
    @patch('builtins.open')
    def test_load_jsonl_with_filename_and_output_config(self, mock_open_file, mock_get_path, mock_meta):
        content = '{"question": "q", "answer": "a"}\n'
        mock_open_file.return_value = mock_open(read_data=content).return_value
        with patch('ais_bench.benchmark.datasets.custom.check_output_config_from_meta_json', return_value=True), \
             patch('ais_bench.benchmark.datasets.custom.check_meta_json_dict', return_value={'sampling_mode': 'default', 'request_count': 0}), \
             patch('ais_bench.benchmark.datasets.custom.get_max_token_list_from_meta_json_file', return_value=[5]):
            ds = CustomDataset.load('/input', file_name='data.jsonl', meta_path='meta.json')
            self.assertEqual(ds[0]['max_out_len'], 5)

    @patch('ais_bench.benchmark.datasets.custom.get_meta_json', return_value={})
    @patch('ais_bench.benchmark.datasets.custom.get_data_path', return_value='/fake/data.csv')
    @patch('builtins.open')
    def test_load_csv(self, mock_open_file, mock_get_path, mock_meta):
        content = 'question,answer\nq,a\n'
        mock_open_file.return_value = mock_open(read_data=content).return_value
        ds = CustomDataset.load('/input')
        self.assertEqual(ds[0]['answer'], 'a')

    @patch('ais_bench.benchmark.datasets.custom.get_data_path', return_value='/fake/data.txt')
    def test_load_unsupported(self, mock_get_path):
        with self.assertRaises(ValueError):
            CustomDataset.load('/input')

    @patch('ais_bench.benchmark.datasets.custom.check_meta_json_dict', return_value={'sampling_mode': 'default', 'request_count': 1})
    @patch('ais_bench.benchmark.datasets.custom.check_output_config_from_meta_json', return_value=False)
    @patch('ais_bench.benchmark.datasets.custom.get_sample_data', return_value=[{'question': 'q', 'answer': 'a', 'max_out_len': 3}])
    @patch('ais_bench.benchmark.datasets.custom.get_meta_json', return_value={'sampling_mode': 'default', 'request_count': 1})
    @patch('ais_bench.benchmark.datasets.custom.get_data_path', return_value='/fake/data.jsonl')
    @patch('builtins.open')
    def test_load_with_meta_sampling(self, mock_open_file, mock_get_path, mock_meta, mock_sample, mock_check, mock_check_dict):
        content = '{"question": "q", "answer": "a"}\n{"question": "q2", "answer": "a2"}\n'
        mock_open_file.return_value = mock_open(read_data=content).return_value
        ds = CustomDataset.load('/input')
        self.assertEqual(len(ds), 1)


class TestHelpers(unittest.TestCase):
    def test_stringfy_types(self):
        obj = {'type': DummyCls, 'nested': {'type': DummyCls}}
        out = stringfy_types(obj)
        self.assertEqual(out['type'], f'{DummyCls.__module__}.{DummyCls.__name__}')
        self.assertEqual(out['nested']['type'], f'{DummyCls.__module__}.{DummyCls.__name__}')

    @patch('ais_bench.benchmark.datasets.custom.get_logger')
    def test_make_mcq_gen_config(self, mock_logger):
        mock_logger.return_value = MagicMock()
        meta = {
            'options': ['A', 'B'],
            'output_column': 'answer',
            'input_columns': ['question'],
            'abbr': 'abbr',
            'path': 'path',
            'meta_path': 'meta',
            'test_range': '[:10]',
        }
        dataset = make_mcq_gen_config(meta, mock_logger.return_value)
        self.assertIn('reader_cfg', dataset)
        self.assertIn('test_range', dataset['reader_cfg'])

    @patch('ais_bench.benchmark.datasets.custom.get_logger')
    def test_make_qa_gen_config(self, mock_logger):
        mock_logger.return_value = MagicMock()
        meta = {
            'output_column': 'answer',
            'input_columns': ['question'],
            'abbr': 'abbr',
            'path': 'path',
            'meta_path': 'meta',
        }
        dataset = make_qa_gen_config(meta, mock_logger.return_value)
        self.assertIn('infer_cfg', dataset)

    @patch('builtins.open')
    def test_parse_example_dataset_jsonl(self, mock_open_file):
        mock_open_file.return_value = mock_open(read_data='{"question": "q", "answer": "a"}\n').return_value
        meta = parse_example_dataset({'path': 'sample.jsonl', 'meta_path': None})
        self.assertEqual(meta['data_type'], 'qa')

    @patch('builtins.open')
    def test_parse_example_dataset_csv(self, mock_open_file):
        mock_open_file.return_value = mock_open(read_data='question,answer\nq,a\n').return_value
        meta = parse_example_dataset({'path': 'sample.csv', 'meta_path': None})
        self.assertEqual(meta['data_type'], 'qa')

    def test_parse_example_dataset_invalid(self):
        with self.assertRaises(ValueError):
            parse_example_dataset({'path': 'sample.txt'})

    @patch('ais_bench.benchmark.datasets.custom.parse_example_dataset')
    @patch('ais_bench.benchmark.datasets.custom.get_logger')
    def test_make_custom_dataset_config(self, mock_logger, mock_parse):
        mock_logger.return_value = MagicMock()
        mock_parse.return_value = {
            'data_type': 'mcq',
            'infer_method': 'gen',
            'options': ['A'],
            'output_column': 'answer',
            'input_columns': ['question'],
            'abbr': 'abbr',
            'path': 'path',
            'meta_path': 'meta',
        }
        dataset = make_custom_dataset_config({'path': 'path'})
        self.assertIn('type', dataset)

    @patch('ais_bench.benchmark.datasets.custom.parse_example_dataset', return_value={'data_type': 'qa', 'infer_method': 'unsupported'})
    @patch('ais_bench.benchmark.datasets.custom.get_logger')
    def test_make_custom_dataset_config_invalid(self, mock_logger, mock_parse):
        mock_logger.return_value = MagicMock()
        with self.assertRaises(ValueError):
            make_custom_dataset_config({'path': 'path'})

    @patch('ais_bench.benchmark.datasets.custom.get_logger')
    @patch('numpy.random.randint', return_value=np.array([1, 2, 3]))
    def test_get_max_token_uniform(self, mock_rand, mock_logger):
        mock_logger.return_value = MagicMock()
        cfg = {"method": "uniform", "params": {"min_value": 1, "max_value": 3}}
        result = get_max_token_list_from_meta_json_file(cfg, 3)
        self.assertEqual(result, [1, 2, 3])

    @patch('ais_bench.benchmark.datasets.custom.get_logger')
    def test_get_max_token_percentage(self, mock_logger):
        mock_logger.return_value = MagicMock()
        cfg = {
            "method": "percentage",
            "params": {
                "percentage_distribute": [(1, 0.25), (2, 0.25)]
            }
        }
        result = get_max_token_list_from_meta_json_file(cfg, 3)
        self.assertEqual(result, [2, 2, 2])

    @patch('ais_bench.benchmark.datasets.custom.get_logger')
    def test_get_max_token_invalid(self, mock_logger):
        mock_logger.return_value = MagicMock()
        with self.assertRaises(ValueError):
            get_max_token_list_from_meta_json_file({"method": "other", "params": {}}, 2)


if __name__ == '__main__':
    unittest.main()
