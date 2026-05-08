import ast
import json
import os
import string
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from datasets import Dataset

from ais_bench.benchmark.datasets.mathvision import (
    MathVisionDataset,
    MathVisionEvaluator,
    _build_choices_text,
    _build_mcq_prompt,
    _build_output_image_path,
    _dump_image,
    _infer_local_data_files,
    _load_records,
    _normalize_choice_answer,
    _normalize_options,
    _pick_files,
    _resolve_dataset_path,
    _resolve_existing_image_path,
    _safe_literal,
    _write_image_bytes,
)


class TestResolveDatasetPath(unittest.TestCase):

    def test_empty_path(self):
        path, is_local = _resolve_dataset_path('')
        self.assertEqual(path, '')
        self.assertFalse(is_local)

    def test_none_path(self):
        path, is_local = _resolve_dataset_path(None)
        self.assertIsNone(path)
        self.assertFalse(is_local)

    def test_absolute_path(self):
        path, is_local = _resolve_dataset_path('/tmp/data')
        self.assertEqual(path, '/tmp/data')
        self.assertTrue(is_local)

    @patch('ais_bench.benchmark.datasets.mathvision.get_data_path', return_value='/cache/data')
    def test_relative_path_local_mode(self, mock_get_path):
        path, is_local = _resolve_dataset_path('some/repo')
        self.assertEqual(path, '/cache/data')
        self.assertTrue(is_local)

    @patch('ais_bench.benchmark.datasets.mathvision.get_data_path', side_effect=Exception('not found'))
    def test_relative_path_fallback(self, mock_get_path):
        path, is_local = _resolve_dataset_path('some/repo')
        self.assertEqual(path, 'some/repo')
        self.assertFalse(is_local)


class TestPickFiles(unittest.TestCase):

    def test_non_directory(self):
        result = _pick_files(Path('/nonexistent'), ['*.parquet'])
        self.assertEqual(result, [])

    def test_no_matching_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / 'data.json').write_text('[]')
            result = _pick_files(Path(tmpdir), ['*.parquet'])
            self.assertEqual(result, [])

    def test_matching_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / 'test-0.parquet').write_text('')
            (Path(tmpdir) / 'test-1.parquet').write_text('')
            result = _pick_files(Path(tmpdir), ['test-*.parquet'])
            self.assertEqual(len(result), 2)

    def test_first_pattern_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / 'test.parquet').write_text('')
            (Path(tmpdir) / 'data.json').write_text('[]')
            result = _pick_files(Path(tmpdir), ['*.parquet', '*.json'])
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0].endswith('.parquet'))

    def test_sorted_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / 'b.parquet').write_text('')
            (Path(tmpdir) / 'a.parquet').write_text('')
            result = _pick_files(Path(tmpdir), ['*.parquet'])
            self.assertEqual(result[0], str(Path(tmpdir) / 'a.parquet'))
            self.assertEqual(result[1], str(Path(tmpdir) / 'b.parquet'))


class TestInferLocalDataFiles(unittest.TestCase):

    def test_non_directory(self):
        fmt, files = _infer_local_data_files(Path('/nonexistent'), 'test')
        self.assertIsNone(fmt)
        self.assertIsNone(files)

    def test_parquet_split_sharded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / 'test-0.parquet').write_text('')
            (Path(tmpdir) / 'test-1.parquet').write_text('')
            fmt, files = _infer_local_data_files(Path(tmpdir), 'test')
            self.assertEqual(fmt, 'parquet')
            self.assertIn('test', files)
            self.assertEqual(len(files['test']), 2)

    def test_parquet_single(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / 'test.parquet').write_text('')
            fmt, files = _infer_local_data_files(Path(tmpdir), 'test')
            self.assertEqual(fmt, 'parquet')
            self.assertIn('test', files)
            self.assertIsInstance(files['test'], str)

    def test_jsonl_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / 'test.jsonl').write_text('')
            fmt, files = _infer_local_data_files(Path(tmpdir), 'test')
            self.assertEqual(fmt, 'json')
            self.assertIn('test', files)

    def test_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / 'test.json').write_text('{}')
            fmt, files = _infer_local_data_files(Path(tmpdir), 'test')
            self.assertEqual(fmt, 'json')
            self.assertIn('test', files)

    def test_data_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / 'data'
            data_dir.mkdir()
            (data_dir / 'test.parquet').write_text('')
            fmt, files = _infer_local_data_files(Path(tmpdir), 'test')
            self.assertEqual(fmt, 'parquet')

    def test_parquet_preferred_over_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / 'test.parquet').write_text('')
            (Path(tmpdir) / 'test.json').write_text('{}')
            fmt, files = _infer_local_data_files(Path(tmpdir), 'test')
            self.assertEqual(fmt, 'parquet')


class TestSafeLiteral(unittest.TestCase):

    def test_non_string(self):
        self.assertEqual(_safe_literal(42), 42)
        self.assertEqual(_safe_literal([1, 2]), [1, 2])

    def test_empty_string(self):
        self.assertEqual(_safe_literal(''), [])
        self.assertEqual(_safe_literal('   '), [])

    def test_json_list(self):
        self.assertEqual(_safe_literal('["a", "b"]'), ['a', 'b'])

    def test_json_dict(self):
        result = _safe_literal('{"A": "opt1"}')
        self.assertEqual(result, {'A': 'opt1'})

    def test_literal_eval_fallback(self):
        result = _safe_literal("{'A': 'opt1'}")
        self.assertEqual(result, {'A': 'opt1'})

    def test_invalid_structure(self):
        result = _safe_literal('[invalid')
        self.assertEqual(result, '[invalid')

    def test_plain_string(self):
        self.assertEqual(_safe_literal('hello'), 'hello')


class TestNormalizeOptions(unittest.TestCase):

    def test_none(self):
        self.assertEqual(_normalize_options(None), [])

    def test_list(self):
        self.assertEqual(_normalize_options(['a', 'b']), ['a', 'b'])

    def test_list_with_none_and_empty(self):
        self.assertEqual(_normalize_options(['a', None, '', 'b']), ['a', 'b'])

    def test_dict_ordered_by_key(self):
        result = _normalize_options({'B': 'beta', 'A': 'alpha'})
        self.assertEqual(result, ['alpha', 'beta'])

    def test_dict_numeric_keys(self):
        result = _normalize_options({1: 'one', 2: 'two'})
        self.assertEqual(result, ['one', 'two'])

    def test_string_json_list(self):
        result = _normalize_options('["opt1", "opt2"]')
        self.assertEqual(result, ['opt1', 'opt2'])

    def test_string_json_dict(self):
        result = _normalize_options('{"A": "alpha", "B": "beta"}')
        self.assertEqual(result, ['alpha', 'beta'])

    def test_plain_string(self):
        self.assertEqual(_normalize_options('single option'), ['single option'])

    def test_empty_string(self):
        self.assertEqual(_normalize_options(''), [])

    def test_tuple(self):
        self.assertEqual(_normalize_options(('x', 'y')), ['x', 'y'])


class TestNormalizeChoiceAnswer(unittest.TestCase):

    def test_empty_answer(self):
        self.assertEqual(_normalize_choice_answer('', ['a', 'b']), '')

    def test_uppercase_letter(self):
        self.assertEqual(_normalize_choice_answer('A', ['opt1', 'opt2']), 'A')

    def test_lowercase_letter(self):
        self.assertEqual(_normalize_choice_answer('a', ['opt1', 'opt2']), 'A')

    def test_zero_index(self):
        self.assertEqual(_normalize_choice_answer('0', ['opt1', 'opt2']), 'A')

    def test_two_index(self):
        self.assertEqual(_normalize_choice_answer('2', ['opt1', 'opt2']), 'B')

    def test_letter_with_suffix(self):
        self.assertEqual(_normalize_choice_answer('A)', ['opt1', 'opt2']), 'A')
        self.assertEqual(_normalize_choice_answer('B.', ['opt1', 'opt2']), 'B')

    def test_match_option_text(self):
        self.assertEqual(_normalize_choice_answer('opt1', ['opt1', 'opt2']), 'A')

    def test_no_match(self):
        self.assertEqual(_normalize_choice_answer('X', ['opt1', 'opt2']), 'X')

    def test_out_of_range_index(self):
        self.assertEqual(_normalize_choice_answer('5', ['opt1', 'opt2']), '5')


class TestBuildChoicesText(unittest.TestCase):

    def test_basic(self):
        result = _build_choices_text(['cat', 'dog'])
        self.assertIn('A) cat', result)
        self.assertIn('B) dog', result)

    def test_single(self):
        result = _build_choices_text(['only'])
        self.assertEqual(result, 'A) only')

    def test_many(self):
        options = [f'opt{i}' for i in range(5)]
        result = _build_choices_text(options)
        self.assertIn('A) opt0', result)
        self.assertIn('E) opt4', result)


class TestBuildMcqPrompt(unittest.TestCase):

    def test_without_template(self):
        result = _build_mcq_prompt('What is 2+2?', ['3', '4', '5'])
        self.assertIn('What is 2+2?', result)
        self.assertIn('A) 3', result)
        self.assertIn('B) 4', result)
        self.assertIn('C) 5', result)

    def test_with_template(self):
        template = 'Q: {question}\nChoices:\n{choices}\nAnswer ({letters}):'
        result = _build_mcq_prompt('Q?', ['a', 'b'], template)
        self.assertIn('Q: Q?', result)
        self.assertIn('A) a', result)
        self.assertIn('B) b', result)
        self.assertIn('Answer (A,B):', result)


class TestBuildOutputImagePath(unittest.TestCase):

    def test_with_id(self):
        result = _build_output_image_path({'id': 'test_1'}, 0, '/img_dir')
        self.assertEqual(result, '/img_dir/test_1.png')

    def test_with_index(self):
        result = _build_output_image_path({}, 5, '/img_dir')
        self.assertEqual(result, '/img_dir/5.png')

    def test_special_characters_in_id(self):
        result = _build_output_image_path({'id': 'a/b c:d'}, 0, '/img_dir')
        self.assertIn('_', os.path.basename(result))
        self.assertNotIn('/', os.path.basename(result).replace('.png', ''))

    def test_custom_suffix(self):
        result = _build_output_image_path({'id': 'img'}, 0, '/img_dir', suffix='.jpg')
        self.assertEqual(result, '/img_dir/img.jpg')


class TestResolveExistingImagePath(unittest.TestCase):

    def test_none_path(self):
        self.assertIsNone(_resolve_existing_image_path(None, '/data', '/img'))

    def test_empty_path(self):
        self.assertIsNone(_resolve_existing_image_path('', '/data', '/img'))

    def test_existing_absolute_path(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            try:
                result = _resolve_existing_image_path(f.name, '/data', '/img')
                self.assertEqual(result, os.path.abspath(f.name))
            finally:
                os.unlink(f.name)

    def test_relative_path_with_data_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_file = os.path.join(tmpdir, 'image.png')
            Path(img_file).write_text('')
            result = _resolve_existing_image_path('image.png', tmpdir, '/other')
            self.assertEqual(result, os.path.abspath(img_file))

    def test_relative_path_with_image_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_file = os.path.join(tmpdir, 'image.png')
            Path(img_file).write_text('')
            result = _resolve_existing_image_path('image.png', None, tmpdir)
            self.assertEqual(result, os.path.abspath(img_file))

    def test_list_of_paths(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            try:
                result = _resolve_existing_image_path([None, f.name], '/data', '/img')
                self.assertEqual(result, os.path.abspath(f.name))
            finally:
                os.unlink(f.name)

    def test_list_no_valid_paths(self):
        result = _resolve_existing_image_path(['/nonexistent1.png', '/nonexistent2.png'], '/data', '/img')
        self.assertIsNone(result)

    def test_nonexistent_path(self):
        result = _resolve_existing_image_path('/nonexistent/path.png', '/data', '/img')
        self.assertIsNone(result)


class TestWriteImageBytes(unittest.TestCase):

    def test_write_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'sub', 'img.png')
            data = b'\x89PNG\r\n\x1a\n'
            result = _write_image_bytes(data, output_path)
            self.assertEqual(result, output_path)
            self.assertTrue(os.path.exists(output_path))
            with open(output_path, 'rb') as f:
                self.assertEqual(f.read(), data)

    def test_write_list_of_ints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'img.png')
            data = [0x89, 0x50, 0x4E, 0x47]
            _write_image_bytes(data, output_path)
            with open(output_path, 'rb') as f:
                self.assertEqual(f.read(), bytes(data))


class TestDumpImage(unittest.TestCase):

    def test_existing_image_path(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            try:
                record = {'image_path': f.name, 'id': 'test'}
                result = _dump_image(record, 0, '/img_dir', '/data')
                self.assertEqual(result, os.path.abspath(f.name))
            finally:
                os.unlink(f.name)

    def test_bytes_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {'id': 'test_bytes', 'image': b'\x89PNG\r\n\x1a\n'}
            result = _dump_image(record, 0, tmpdir, None)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))

    def test_dict_image_with_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                'id': 'test_dict',
                'image': {'bytes': b'\x89PNG\r\n\x1a\n', 'path': 'test.png'},
            }
            result = _dump_image(record, 0, tmpdir, None)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))

    def test_dict_image_with_existing_path(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            try:
                record = {
                    'id': 'test_dict_path',
                    'image': {'path': f.name},
                }
                result = _dump_image(record, 0, '/img_dir', '/data')
                self.assertIsNotNone(result)
            finally:
                os.unlink(f.name)

    def test_no_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {'id': 'no_img'}
            result = _dump_image(record, 0, tmpdir, None)
            self.assertIsNone(result)

    def test_pil_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from PIL import Image
            img = Image.new('RGB', (10, 10), color='red')
            record = {'id': 'pil_img', 'image': img}
            result = _dump_image(record, 0, tmpdir, None)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))

    def test_list_image_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {
                'id': 'list_img',
                'image': [b'\x89PNG\r\n\x1a\n'],
            }
            result = _dump_image(record, 0, tmpdir, None)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))


class TestLoadRecords(unittest.TestCase):

    @patch('ais_bench.benchmark.datasets.mathvision.load_dataset')
    @patch('ais_bench.benchmark.datasets.mathvision._resolve_dataset_path')
    def test_remote_dataset(self, mock_resolve, mock_load):
        mock_resolve.return_value = ('evalscope/MathVision', False)
        mock_dataset = [{'question': 'Q?', 'answer': '42', 'image': b'img'}]
        mock_load.return_value = mock_dataset
        records, path, is_local = _load_records('evalscope/MathVision', split='test')
        self.assertEqual(records, mock_dataset)
        self.assertFalse(is_local)

    @patch('ais_bench.benchmark.datasets.mathvision.load_dataset')
    @patch('ais_bench.benchmark.datasets.mathvision._resolve_dataset_path')
    def test_local_json_file(self, mock_resolve, mock_load):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump([{'question': 'Q?', 'answer': '42'}], f)
            f.flush()
            mock_resolve.return_value = (f.name, True)
            mock_load.return_value = DatasetDict_mock({'test': [{'question': 'Q?', 'answer': '42'}]})
            records, path, is_local = _load_records(f.name, split='test')
            self.assertTrue(is_local)
            os.unlink(f.name)

    @patch('ais_bench.benchmark.datasets.mathvision._resolve_dataset_path')
    def test_nonexistent_path(self, mock_resolve):
        mock_resolve.return_value = ('/nonexistent/path.json', True)
        with self.assertRaises(FileNotFoundError):
            _load_records('/nonexistent/path.json')

    @patch('ais_bench.benchmark.datasets.mathvision._resolve_dataset_path')
    def test_unsupported_file_type(self, mock_resolve):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            mock_resolve.return_value = (f.name, True)
            with self.assertRaises(ValueError):
                _load_records(f.name)
            os.unlink(f.name)


class DatasetDict_mock(dict):
    def __getitem__(self, key):
        return self.get(key, [])


class TestMathVisionEvaluatorCleanPrediction(unittest.TestCase):

    def test_clean_special_characters(self):
        result = MathVisionEvaluator._clean_prediction('answer<|im_end|>')
        self.assertEqual(result, 'answer')

    def test_clean_whitespace(self):
        result = MathVisionEvaluator._clean_prediction('  answer  ')
        self.assertEqual(result, 'answer')

    def test_clean_non_string(self):
        result = MathVisionEvaluator._clean_prediction(42)
        self.assertEqual(result, '42')

    def test_clean_no_special_chars(self):
        result = MathVisionEvaluator._clean_prediction('normal answer')
        self.assertEqual(result, 'normal answer')


class TestMathVisionEvaluatorExtractOpenPrediction(unittest.TestCase):

    def test_boxed_answer(self):
        result = MathVisionEvaluator._extract_open_prediction(
            'The answer is \\boxed{42}'
        )
        self.assertEqual(result, '42')

    def test_boxed_answer_double_brace(self):
        result = MathVisionEvaluator._extract_open_prediction(
            'The answer is \\boxed{{42}}'
        )
        self.assertEqual(result, '42')

    def test_extract_answer_pattern(self):
        result = MathVisionEvaluator._extract_open_prediction(
            'ANSWER: 42'
        )
        self.assertEqual(result, '42')

    def test_final_answer_pattern(self):
        result = MathVisionEvaluator._extract_open_prediction(
            'The final answer is 42. Done.'
        )
        self.assertIn('42', result)

    def test_answer_is_pattern(self):
        result = MathVisionEvaluator._extract_open_prediction(
            'The answer is 42. Done.'
        )
        self.assertIn('42', result)

    def test_no_pattern_returns_stripped(self):
        result = MathVisionEvaluator._extract_open_prediction('just a number 42')
        self.assertEqual(result, 'just a number 42')

    def test_regex_boxed_fallback(self):
        result = MathVisionEvaluator._extract_open_prediction(
            'Result: \\boxed{100}'
        )
        self.assertEqual(result, '100')


class TestMathVisionEvaluatorExtractMcqPrediction(unittest.TestCase):

    def test_answer_pattern(self):
        result = MathVisionEvaluator._extract_mcq_prediction(
            'ANSWER: A', {'A': 'opt1', 'B': 'opt2'}
        )
        self.assertEqual(result, 'A')

    def test_answer_pattern_case_insensitive(self):
        result = MathVisionEvaluator._extract_mcq_prediction(
            'answer: B', {'A': 'opt1', 'B': 'opt2'}
        )
        self.assertEqual(result, 'B')

    def test_can_infer_fallback(self):
        result = MathVisionEvaluator._extract_mcq_prediction(
            'I choose opt1', {'A': 'opt1', 'B': 'opt2'}
        )
        self.assertEqual(result, 'A')

    def test_no_match(self):
        result = MathVisionEvaluator._extract_mcq_prediction(
            'I have no idea', {'A': 'opt1', 'B': 'opt2'}
        )
        self.assertFalse(result)


class TestMathVisionEvaluatorScore(unittest.TestCase):

    def test_score_length_mismatch(self):
        evaluator = MathVisionEvaluator()
        result = evaluator.score(['pred1'], ['ref1', 'ref2'])
        self.assertIn('error', result)

    def test_score_mcq_correct(self):
        evaluator = MathVisionEvaluator()
        references = [
            {'type': 'mcq', 'answer': 'A', 'choices': json.dumps({'A': 'opt1', 'B': 'opt2'})},
        ]
        predictions = ['ANSWER: A']
        result = evaluator.score(predictions, references)
        self.assertEqual(result['Accuracy'], 100.0)
        self.assertTrue(result['details'][0]['correct'])

    def test_score_mcq_incorrect(self):
        evaluator = MathVisionEvaluator()
        references = [
            {'type': 'mcq', 'answer': 'B', 'choices': json.dumps({'A': 'opt1', 'B': 'opt2'})},
        ]
        predictions = ['ANSWER: A']
        result = evaluator.score(predictions, references)
        self.assertEqual(result['Accuracy'], 0.0)
        self.assertFalse(result['details'][0]['correct'])

    def test_score_open_correct(self):
        evaluator = MathVisionEvaluator()
        references = [
            {'type': 'open', 'answer': '42'},
        ]
        predictions = ['\\boxed{42}']
        result = evaluator.score(predictions, references)
        self.assertEqual(result['Accuracy'], 100.0)

    def test_score_open_incorrect(self):
        evaluator = MathVisionEvaluator()
        references = [
            {'type': 'open', 'answer': '42'},
        ]
        predictions = ['\\boxed{99}']
        result = evaluator.score(predictions, references)
        self.assertEqual(result['Accuracy'], 0.0)

    def test_score_mixed_types(self):
        evaluator = MathVisionEvaluator()
        references = [
            {'type': 'mcq', 'answer': 'A', 'choices': json.dumps({'A': 'opt1', 'B': 'opt2'})},
            {'type': 'open', 'answer': '42'},
        ]
        predictions = ['ANSWER: A', '\\boxed{42}']
        result = evaluator.score(predictions, references)
        self.assertEqual(result['Accuracy'], 100.0)

    def test_score_with_level_breakdown(self):
        evaluator = MathVisionEvaluator()
        references = [
            {'type': 'mcq', 'answer': 'A', 'choices': json.dumps({'A': 'opt1', 'B': 'opt2'}), 'level': 1},
            {'type': 'mcq', 'answer': 'B', 'choices': json.dumps({'A': 'opt1', 'B': 'opt2'}), 'level': 1},
            {'type': 'mcq', 'answer': 'A', 'choices': json.dumps({'A': 'opt1', 'B': 'opt2'}), 'level': 2},
        ]
        predictions = ['ANSWER: A', 'ANSWER: A', 'ANSWER: B']
        result = evaluator.score(predictions, references)
        self.assertIn('level 1', result)
        self.assertIn('level 2', result)
        self.assertEqual(result['level 1'], 50.0)
        self.assertEqual(result['level 2'], 0.0)

    def test_score_with_subject_breakdown(self):
        evaluator = MathVisionEvaluator()
        references = [
            {'type': 'open', 'answer': '42', 'subject': 'algebra'},
            {'type': 'open', 'answer': '3.14', 'subject': 'geometry'},
        ]
        predictions = ['\\boxed{42}', '\\boxed{3.14}']
        result = evaluator.score(predictions, references)
        self.assertIn('subject: algebra', result)
        self.assertIn('subject: geometry', result)
        self.assertEqual(result['subject: algebra'], 100.0)
        self.assertEqual(result['subject: geometry'], 100.0)

    def test_score_string_reference(self):
        evaluator = MathVisionEvaluator()
        references = ['42']
        predictions = ['\\boxed{42}']
        result = evaluator.score(predictions, references)
        self.assertIn('Accuracy', result)

    def test_score_empty(self):
        evaluator = MathVisionEvaluator()
        result = evaluator.score([], [])
        self.assertEqual(result['Accuracy'], 0.0)

    def test_score_choices_as_dict(self):
        evaluator = MathVisionEvaluator()
        references = [
            {'type': 'mcq', 'answer': 'A', 'choices': {'A': 'opt1', 'B': 'opt2'}},
        ]
        predictions = ['ANSWER: A']
        result = evaluator.score(predictions, references)
        self.assertEqual(result['Accuracy'], 100.0)

    def test_score_detail_parsed_pred_mcq(self):
        evaluator = MathVisionEvaluator()
        references = [
            {'type': 'mcq', 'answer': 'A', 'choices': json.dumps({'A': 'opt1', 'B': 'opt2'})},
        ]
        predictions = ['ANSWER: A']
        result = evaluator.score(predictions, references)
        self.assertEqual(result['details'][0]['parsed_pred'], 'A')

    def test_score_detail_parsed_pred_open(self):
        evaluator = MathVisionEvaluator()
        references = [
            {'type': 'open', 'answer': '42'},
        ]
        predictions = ['\\boxed{42}']
        result = evaluator.score(predictions, references)
        self.assertIn('parsed_pred', result['details'][0])

    def test_score_special_char_cleaned(self):
        evaluator = MathVisionEvaluator()
        references = [
            {'type': 'mcq', 'answer': 'A', 'choices': json.dumps({'A': 'opt1', 'B': 'opt2'})},
        ]
        predictions = ['ANSWER: A<|im_end|>']
        result = evaluator.score(predictions, references)
        self.assertEqual(result['Accuracy'], 100.0)


class TestMathVisionEvaluatorInit(unittest.TestCase):

    def test_init(self):
        evaluator = MathVisionEvaluator()
        self.assertIsNotNone(evaluator.math_evaluator)


if __name__ == '__main__':
    unittest.main()
