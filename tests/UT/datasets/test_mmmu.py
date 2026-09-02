import json
import os
import string
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd

from ais_bench.benchmark.datasets.mmmu import (
    dump_image,
    split_MMMU,
    build_choices,
    _safe_list,
    _answer_character,
    _format_mmmu_choices,
    _format_mmmu_letters,
    _build_mmmu_mcq_prompt,
    _parquet_sort_key,
    _find_mmmu_parquet_files,
    _infer_subject_from_parquet_path,
    _load_mmmu_records,
    _resolve_mmmu_existing_image_path,
    _write_mmmu_image_bytes,
    _build_mmmu_image_path,
    _dump_mmmu_image,
    _collect_mmmu_images,
    _parse_mmmu_text_with_images,
    _parse_mmmu_choice_prediction,
    _extract_mmmu_open_prediction,
    can_infer_option,
    can_infer_text,
    can_infer,
    sort_key,
    MMMUDataset,
    MMMUEvaluator,
    IMAGE_MAP_LEN,
    MMMU_SUBSET_LIST,
    MMMU_MULTI_CHOICE_TYPE,
    MMMU_OPEN_TYPE,
)


class TestConstants(unittest.TestCase):
    def test_image_map_len(self):
        self.assertEqual(IMAGE_MAP_LEN, 64)

    def test_mmmu_subset_list_length(self):
        self.assertEqual(len(MMMU_SUBSET_LIST), 30)

    def test_mmmu_subset_list_contains_known_subjects(self):
        self.assertIn('Math', MMMU_SUBSET_LIST)
        self.assertIn('Physics', MMMU_SUBSET_LIST)
        self.assertIn('Computer_Science', MMMU_SUBSET_LIST)

    def test_mmmu_multi_choice_type(self):
        self.assertEqual(MMMU_MULTI_CHOICE_TYPE, 'multiple-choice')

    def test_mmmu_open_type(self):
        self.assertEqual(MMMU_OPEN_TYPE, 'open')


class TestSafeList(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(_safe_list(None), [])

    def test_nan_returns_empty(self):
        self.assertEqual(_safe_list(float('nan')), [])

    def test_list_passthrough(self):
        self.assertEqual(_safe_list([1, 2, 3]), [1, 2, 3])

    def test_tuple_to_list(self):
        self.assertEqual(_safe_list((1, 2)), [1, 2])

    def test_empty_string_returns_empty(self):
        self.assertEqual(_safe_list(''), [])

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(_safe_list('   '), [])

    def test_json_list_string(self):
        self.assertEqual(_safe_list('["a", "b"]'), ['a', 'b'])

    def test_json_tuple_string(self):
        self.assertEqual(_safe_list('("x", "y")'), ['x', 'y'])

    def test_plain_string_returns_singleton(self):
        self.assertEqual(_safe_list('hello'), ['hello'])

    def test_int_value_wrapped(self):
        self.assertEqual(_safe_list(42), [42])

    def test_bool_value_wrapped(self):
        self.assertEqual(_safe_list(True), [True])

    def test_float_value_wrapped(self):
        self.assertEqual(_safe_list(3.14), [3.14])


class TestAnswerCharacter(unittest.TestCase):
    def test_index_0_is_A(self):
        self.assertEqual(_answer_character(0), 'A')

    def test_index_25_is_Z(self):
        self.assertEqual(_answer_character(25), 'Z')

    def test_index_26_returns_string(self):
        self.assertEqual(_answer_character(26), '1')

    def test_index_27_returns_string(self):
        self.assertEqual(_answer_character(27), '2')

    def test_index_1_is_B(self):
        self.assertEqual(_answer_character(1), 'B')

    def test_index_13_is_N(self):
        self.assertEqual(_answer_character(13), 'N')

class TestBuildMmmuMcqPrompt(unittest.TestCase):
    def test_basic_prompt(self):
        template = 'Question: {question}\n{choices}\nOptions: {letters}'
        result = _build_mmmu_mcq_prompt('What?', ['yes', 'no'], template)
        self.assertIn('What?', result)
        self.assertIn('A) yes', result)
        self.assertIn('B) no', result)
        self.assertIn('A,B', result)

    def test_no_template_raises(self):
        with self.assertRaises(ValueError):
            _build_mmmu_mcq_prompt('Q', ['a'], None)

    def test_empty_template_raises(self):
        with self.assertRaises(ValueError):
            _build_mmmu_mcq_prompt('Q', ['a'], '')


class TestParquetSortKey(unittest.TestCase):
    def test_known_subset_returns_its_index(self):
        path = '/data/Math/validation-001.parquet'
        key = _parquet_sort_key(path)
        expected_index = MMMU_SUBSET_LIST.index('Math')
        self.assertEqual(key[0], expected_index)

    def test_unknown_subset_returns_length(self):
        path = '/data/Unknown/validation-001.parquet'
        key = _parquet_sort_key(path)
        self.assertEqual(key[0], len(MMMU_SUBSET_LIST))

    def test_sort_key_includes_path_str(self):
        path = '/data/Physics/test.parquet'
        key = _parquet_sort_key(path)
        self.assertIsInstance(key[1], str)

    def test_windows_backslash_handled(self):
        path = '\\data\\Math\\validation.parquet'
        key = _parquet_sort_key(path)
        self.assertEqual(key[0], MMMU_SUBSET_LIST.index('Math'))

    def test_subset_in_underscore_pattern(self):
        path = '/data/validation_Math_001.parquet'
        key = _parquet_sort_key(path)
        self.assertEqual(key[0], MMMU_SUBSET_LIST.index('Math'))

    def test_sort_ordering(self):
        k1 = _parquet_sort_key('/data/Accounting/val.parquet')
        k2 = _parquet_sort_key('/data/Math/val.parquet')
        self.assertLess(k1, k2)


class TestInferSubjectFromParquetPath(unittest.TestCase):
    def test_slash_separated(self):
        self.assertEqual(
            _infer_subject_from_parquet_path('/data/Physics/file.parquet'),
            'Physics',
        )

    def test_dash_separated(self):
        self.assertEqual(
            _infer_subject_from_parquet_path('/data/Chemistry-test.parquet'),
            'Chemistry',
        )

    def test_underscore_separated(self):
        self.assertEqual(
            _infer_subject_from_parquet_path('/data/validation_Biology_001.parquet'),
            'Biology',
        )

    def test_unknown_returns_none(self):
        self.assertIsNone(_infer_subject_from_parquet_path('/data/Unknown/file.parquet'))

    def test_pathlib_path(self):
        self.assertEqual(
            _infer_subject_from_parquet_path(Path('/data/Art/file.parquet')),
            'Art',
        )

    def test_first_match_wins(self):
        result = _infer_subject_from_parquet_path('/data/Art/file.parquet')
        self.assertEqual(result, 'Art')


class TestFindMmmuParquetFiles(unittest.TestCase):
    def test_single_file_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / 'test.parquet'
            f.touch()
            result = _find_mmmu_parquet_files(str(f), 'validation')
            self.assertEqual(result, [f])

    def test_non_parquet_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / 'test.csv'
            f.touch()
            with self.assertRaises(ValueError):
                _find_mmmu_parquet_files(str(f), 'validation')

    def test_nonexistent_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            _find_mmmu_parquet_files('/nonexistent/path', 'validation')

    def test_directory_with_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subset_dir = root / 'Math'
            subset_dir.mkdir()
            f = subset_dir / 'validation-001.parquet'
            f.touch()
            result = _find_mmmu_parquet_files(str(root), 'validation')
            self.assertIn(f, result)

    def test_subset_list_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for subj in ['Math', 'Physics']:
                d = root / subj
                d.mkdir()
                (d / 'validation-001.parquet').touch()
            result = _find_mmmu_parquet_files(str(root), 'validation', subset_list=['Math'])
            subjects_found = [_infer_subject_from_parquet_path(f) for f in result]
            self.assertIn('Math', subjects_found)

    def test_data_subdir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subset_dir = root / 'data' / 'Math'
            subset_dir.mkdir(parents=True)
            f = subset_dir / 'validation-001.parquet'
            f.touch()
            result = _find_mmmu_parquet_files(str(root), 'validation', subset_list=['Math'])
            self.assertIn(f, result)

    def test_deduplication(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subset_dir = root / 'Math'
            subset_dir.mkdir()
            f = subset_dir / 'validation-001.parquet'
            f.touch()
            result = _find_mmmu_parquet_files(str(root), 'validation', subset_list=['Math'])
            self.assertEqual(len(result), len(set(result)))


class TestResolveMmmuExistingImagePath(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(_resolve_mmmu_existing_image_path(None, None, None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_resolve_mmmu_existing_image_path('', None, None))

    def test_existing_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, 'img.png')
            open(f, 'w').close()
            result = _resolve_mmmu_existing_image_path(f, None, None)
            self.assertEqual(result, os.path.abspath(f))

    def test_relative_path_in_data_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, 'sub', 'img.png')
            os.makedirs(os.path.dirname(f))
            open(f, 'w').close()
            result = _resolve_mmmu_existing_image_path('sub/img.png', tmpdir, None)
            self.assertEqual(result, os.path.abspath(f))

    def test_relative_path_in_image_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, 'img.png')
            open(f, 'w').close()
            result = _resolve_mmmu_existing_image_path('img.png', None, tmpdir)
            self.assertEqual(result, os.path.abspath(f))

    def test_nonexistent_returns_none(self):
        self.assertIsNone(
            _resolve_mmmu_existing_image_path('missing.png', '/no/root', '/no/root')
        )

    def test_absolute_path_ignores_roots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, 'img.png')
            open(f, 'w').close()
            result = _resolve_mmmu_existing_image_path(f, '/other/root', '/img/root')
            self.assertEqual(result, os.path.abspath(f))


class TestWriteMmmuImageBytes(unittest.TestCase):
    def test_write_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'img.png')
            result = _write_mmmu_image_bytes(b'\x89PNG', out)
            self.assertEqual(result, out)
            with open(out, 'rb') as f:
                self.assertEqual(f.read(), b'\x89PNG')

    def test_write_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'sub', 'img.png')
            result = _write_mmmu_image_bytes([0x89, 0x50], out)
            self.assertEqual(result, out)
            with open(out, 'rb') as f:
                self.assertEqual(f.read(), bytes([0x89, 0x50]))

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'a', 'b', 'img.png')
            _write_mmmu_image_bytes(b'data', out)
            self.assertTrue(os.path.exists(out))

    def test_overwrite_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, 'img.png')
            with open(out, 'wb') as f:
                f.write(b'old')
            _write_mmmu_image_bytes(b'new', out)
            with open(out, 'rb') as f:
                self.assertEqual(f.read(), b'new')


class TestBuildMmmuImagePath(unittest.TestCase):
    def test_with_id(self):
        record = {'id': 'test_001'}
        result = _build_mmmu_image_path(record, 1, '/root')
        self.assertEqual(result, os.path.join('/root', 'test_001_1.png'))

    def test_with_index_fallback(self):
        record = {'index': 42}
        result = _build_mmmu_image_path(record, 2, '/root')
        self.assertEqual(result, os.path.join('/root', '42_2.png'))

    def test_default_stem(self):
        record = {}
        result = _build_mmmu_image_path(record, 1, '/root')
        self.assertEqual(result, os.path.join('/root', 'sample_1.png'))

    def test_custom_suffix(self):
        record = {'id': 'abc'}
        result = _build_mmmu_image_path(record, 1, '/root', suffix='.jpg')
        self.assertEqual(result, os.path.join('/root', 'abc_1.jpg'))

    def test_special_chars_sanitized(self):
        record = {'id': 'test@#$001'}
        result = _build_mmmu_image_path(record, 1, '/root')
        self.assertIn('test_001_1.png', result)

    def test_dots_preserved(self):
        record = {'id': 'img.v2'}
        result = _build_mmmu_image_path(record, 1, '/root')
        self.assertIn('img.v2', result)

    def test_id_takes_precedence_over_index(self):
        record = {'id': 'primary', 'index': 99}
        result = _build_mmmu_image_path(record, 1, '/root')
        self.assertIn('primary', result)
        self.assertNotIn('99', result)


class TestDumpMmmuImage(unittest.TestCase):
    def test_none_candidate(self):
        self.assertIsNone(_dump_mmmu_image(None, {}, 1, '/root', None))

    def test_nan_candidate(self):
        self.assertIsNone(_dump_mmmu_image(float('nan'), {}, 1, '/root', None))

    def test_bytes_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {'id': 'r1'}
            result = _dump_mmmu_image(b'\x89PNG', record, 1, tmpdir, None)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))

    def test_bytearray_candidate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {'id': 'r1b'}
            result = _dump_mmmu_image(bytearray(b'\x89PNG'), record, 1, tmpdir, None)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))

    def test_dict_candidate_with_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, 'existing.png')
            open(f, 'w').close()
            candidate = {'path': f}
            result = _dump_mmmu_image(candidate, {}, 1, tmpdir, tmpdir)
            self.assertEqual(result, os.path.abspath(f))

    def test_dict_candidate_with_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = {'bytes': b'\x89PNG', 'path': 'test.png'}
            record = {'id': 'r2'}
            result = _dump_mmmu_image(candidate, record, 1, tmpdir, None)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))

    def test_dict_candidate_path_not_found_falls_back_to_bytes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = {'bytes': b'\x89PNG', 'path': '/nonexistent/missing.png'}
            record = {'id': 'r2b'}
            result = _dump_mmmu_image(candidate, record, 1, tmpdir, None)
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))

    @patch('ais_bench.benchmark.datasets.mmmu.decode_base64_to_image_file')
    def test_string_candidate_as_existing_path(self, mock_decode):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, 'img.png')
            open(f, 'w').close()
            result = _dump_mmmu_image(f, {}, 1, tmpdir, tmpdir)
            self.assertEqual(result, os.path.abspath(f))
            mock_decode.assert_not_called()

    def test_object_with_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_img = MagicMock()
            record = {'id': 'r3'}
            result = _dump_mmmu_image(mock_img, record, 1, tmpdir, None)
            self.assertIsNotNone(result)
            mock_img.save.assert_called_once()

    @patch('ais_bench.benchmark.datasets.mmmu.decode_base64_to_image_file')
    def test_string_candidate_as_base64(self, mock_decode):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {'id': 'r4'}
            result = _dump_mmmu_image('base64data', record, 1, tmpdir, '/no/path')
            self.assertIsNotNone(result)
            mock_decode.assert_called_once()

    @patch('ais_bench.benchmark.datasets.mmmu.decode_base64_to_image_file', side_effect=Exception('bad'))
    def test_string_candidate_decode_failure(self, mock_decode):
        with tempfile.TemporaryDirectory() as tmpdir:
            record = {'id': 'r5'}
            result = _dump_mmmu_image('not_base64', record, 1, tmpdir, '/no/path')
            self.assertIsNone(result)


class TestCollectMmmuImages(unittest.TestCase):
    @patch('ais_bench.benchmark.datasets.mmmu._dump_mmmu_image')
    def test_image_fields(self, mock_dump):
        mock_dump.return_value = '/tmp/img.png'
        record = {'image_1': 'data1', 'image_2': 'data2'}
        result = _collect_mmmu_images(record, '/tmp', None)
        self.assertIn(1, result)
        self.assertIn(2, result)

    @patch('ais_bench.benchmark.datasets.mmmu._dump_mmmu_image')
    def test_fallback_to_image_field(self, mock_dump):
        mock_dump.side_effect = [None] * 7 + ['/tmp/img.png']
        record = {'image': ['base64data']}
        result = _collect_mmmu_images(record, '/tmp', None)
        self.assertIn(1, result)

    @patch('ais_bench.benchmark.datasets.mmmu._dump_mmmu_image')
    def test_fallback_to_image_path(self, mock_dump):
        mock_dump.side_effect = [None] * 7 + ['/tmp/img.png']
        record = {'image_path': ['path1']}
        result = _collect_mmmu_images(record, '/tmp', None)
        self.assertIn(1, result)

    @patch('ais_bench.benchmark.datasets.mmmu._dump_mmmu_image')
    def test_no_images(self, mock_dump):
        mock_dump.return_value = None
        record = {}
        result = _collect_mmmu_images(record, '/tmp', None)
        self.assertEqual(result, {})

    @patch('ais_bench.benchmark.datasets.mmmu._dump_mmmu_image')
    def test_partial_image_fields(self, mock_dump):
        def side_effect(candidate, rec, idx, irp, dr):
            if idx == 3:
                return '/tmp/img3.png'
            return None
        mock_dump.side_effect = side_effect
        record = {'image_3': 'data3'}
        result = _collect_mmmu_images(record, '/tmp', None)
        self.assertIn(3, result)
        self.assertEqual(len(result), 1)


class TestParseMmmuTextWithImages(unittest.TestCase):
    def test_text_with_placeholder(self):
        image_map = {1: '/path/img1.png'}
        text = 'Look at <image 1> carefully.'
        result = _parse_mmmu_text_with_images(text, image_map)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['type'], 'text')
        self.assertEqual(result[0]['text'], 'Look at ')
        self.assertEqual(result[1]['type'], 'image_url')
        self.assertEqual(result[1]['image_url'], '/path/img1.png')
        self.assertEqual(result[2]['type'], 'text')

    def test_multiple_placeholders(self):
        image_map = {1: '/img1.png', 2: '/img2.png'}
        text = '<image 1> and <image 2>'
        result = _parse_mmmu_text_with_images(text, image_map)
        types = [m['type'] for m in result]
        self.assertEqual(types, ['image_url', 'text', 'image_url'])

    def test_no_placeholder_with_image_map(self):
        image_map = {1: '/img1.png'}
        text = 'No images here.'
        result = _parse_mmmu_text_with_images(text, image_map)
        self.assertEqual(result[0]['type'], 'image_url')
        self.assertEqual(result[0]['image_url'], '/img1.png')

    def test_no_placeholder_no_images(self):
        result = _parse_mmmu_text_with_images('Plain text', {})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], 'Plain text')

    def test_underscore_variant(self):
        image_map = {1: '/img1.png'}
        text = 'See <image_1> here.'
        result = _parse_mmmu_text_with_images(text, image_map)
        self.assertTrue(any(m['type'] == 'image_url' for m in result))

    def test_placeholder_with_space(self):
        image_map = {1: '/img1.png'}
        text = 'See <image 1> here.'
        result = _parse_mmmu_text_with_images(text, image_map)
        self.assertTrue(any(m['type'] == 'image_url' for m in result))

    def test_missing_image_in_map(self):
        image_map = {2: '/img2.png'}
        text = '<image 1> not in map'
        result = _parse_mmmu_text_with_images(text, image_map)
        has_image = any(m['type'] == 'image_url' for m in result)
        self.assertFalse(has_image)

    def test_multiple_images_ordering(self):
        image_map = {1: '/img1.png', 2: '/img2.png', 3: '/img3.png'}
        text = '<image 1> mid <image 3> end'
        result = _parse_mmmu_text_with_images(text, image_map)
        image_urls = [m['image_url'] for m in result if m['type'] == 'image_url']
        self.assertEqual(image_urls, ['/img1.png', '/img3.png'])

    def test_whitespace_only_segments_filtered(self):
        image_map = {1: '/img1.png'}
        text = '   <image 1>   '
        result = _parse_mmmu_text_with_images(text, image_map)
        text_parts = [m for m in result if m['type'] == 'text']
        for part in text_parts:
            self.assertTrue(part['text'].strip())


class TestParseMmmuChoicePrediction(unittest.TestCase):
    def test_answer_prefix(self):
        result = _parse_mmmu_choice_prediction('Answer: B', 4)
        self.assertEqual(result, 'B')

    def test_answer_prefix_lowercase(self):
        result = _parse_mmmu_choice_prediction('answer: C', 4)
        self.assertEqual(result, 'C')

    def test_answer_with_trailing_dot(self):
        result = _parse_mmmu_choice_prediction('ANSWER: A.', 4)
        self.assertEqual(result, 'A')

    def test_invalid_option_returns_empty(self):
        result = _parse_mmmu_choice_prediction('Answer: Z', 3)
        self.assertEqual(result, '')

    def test_no_answer_prefix_fallback(self):
        result = _parse_mmmu_choice_prediction('I think C is correct', 4)
        self.assertEqual(result, 'C')

    def test_no_uppercase_returns_empty(self):
        result = _parse_mmmu_choice_prediction('nothing here', 4)
        self.assertEqual(result, '')

    def test_multiline_with_answer(self):
        result = _parse_mmmu_choice_prediction('reasoning...\nANSWER: A\nmore text', 4)
        self.assertEqual(result, 'A')

    def test_answer_with_extra_text_returns_empty(self):
        result = _parse_mmmu_choice_prediction('ANSWER: AB', 4)
        self.assertEqual(result, '')

    def test_answer_with_spaces_and_commas(self):
        result = _parse_mmmu_choice_prediction('ANSWER: B, ', 4)
        self.assertEqual(result, '')

    def test_returns_last_uppercase_on_no_match(self):
        result = _parse_mmmu_choice_prediction('blah B', 4)
        self.assertEqual(result, 'B')


class TestExtractMmmuOpenPrediction(unittest.TestCase):
    def test_with_answer_prefix(self):
        self.assertEqual(_extract_mmmu_open_prediction('ANSWER: Paris'), 'Paris')

    def test_with_lowercase_prefix(self):
        self.assertEqual(_extract_mmmu_open_prediction('answer: Berlin'), 'answer: Berlin')

    def test_no_prefix(self):
        self.assertEqual(
            _extract_mmmu_open_prediction('The answer is Tokyo'),
            'The answer is Tokyo',
        )

    def test_whitespace_stripped(self):
        self.assertEqual(_extract_mmmu_open_prediction('  ANSWER: Rome  '), 'Rome')

    def test_empty_string(self):
        self.assertEqual(_extract_mmmu_open_prediction(''), '')

    def test_only_prefix(self):
        self.assertEqual(_extract_mmmu_open_prediction('ANSWER:'), '')

    def test_multiline_takes_first_answer_line(self):
        result = _extract_mmmu_open_prediction('thinking...\nANSWER: Tokyo\nmore')
        self.assertEqual(result, 'Tokyo')


class TestBuildChoices(unittest.TestCase):
    def test_with_valid_choices(self):
        item = pd.Series({'A': 'apple', 'B': 'banana', 'C': 'cherry'})
        result = build_choices(item)
        self.assertEqual(result, {'A': 'apple', 'B': 'banana', 'C': 'cherry'})

    def test_with_nan_choices(self):
        item = pd.Series({'A': 'apple', 'B': np.nan})
        result = build_choices(item)
        self.assertEqual(result, {'A': 'apple'})

    def test_no_choices(self):
        item = pd.Series({'question': 'What?'})
        result = build_choices(item)
        self.assertEqual(result, {})

    def test_mixed_valid_and_nan(self):
        item = pd.Series({'A': 'opt1', 'B': np.nan, 'C': 'opt3', 'D': np.nan})
        result = build_choices(item)
        self.assertEqual(result, {'A': 'opt1', 'C': 'opt3'})

    def test_non_letter_keys_ignored(self):
        item = pd.Series({'A': 'yes', 'question': 'What?', '1': 'num'})
        result = build_choices(item)
        self.assertIn('A', result)
        self.assertNotIn('1', result)


class TestSplitMMMU(unittest.TestCase):
    def test_no_image_tags_returns_original(self):
        msgs = [
            {'type': 'text', 'text': 'Hello world'},
            {'type': 'image_url', 'image_url': 'url1'},
        ]
        result = split_MMMU(msgs)
        self.assertEqual(result, msgs)

    def test_single_image_placeholder(self):
        msgs = [
            {'type': 'text', 'text': 'What is <image 1>?'},
            {'type': 'image_url', 'image_url': 'url1'},
        ]
        result = split_MMMU(msgs)
        self.assertEqual(result[0], {'type': 'text', 'text': 'What is '})
        self.assertEqual(result[1], {'type': 'image_url', 'image_url': 'url1'})
        self.assertEqual(result[2], {'type': 'text', 'text': '?'})

    def test_two_image_placeholders(self):
        msgs = [
            {'type': 'text', 'text': '<image 1> vs <image 2>'},
            {'type': 'image_url', 'image_url': 'url1'},
            {'type': 'image_url', 'image_url': 'url2'},
        ]
        result = split_MMMU(msgs)
        types = [m['type'] for m in result]
        self.assertEqual(types, ['text', 'image_url', 'text', 'image_url', 'text'])

    def test_empty_text_parts_filtered(self):
        msgs = [
            {'type': 'text', 'text': '<image 1>'},
            {'type': 'image_url', 'image_url': 'url1'},
        ]
        result = split_MMMU(msgs)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], {'type': 'text', 'text': ''})
        self.assertEqual(result[1], {'type': 'image_url', 'image_url': 'url1'})
        self.assertEqual(result[2], {'type': 'text', 'text': ''})


class TestDumpImage(unittest.TestCase):
    @patch('ais_bench.benchmark.datasets.mmmu.decode_base64_to_image_file')
    @patch('os.path.exists', return_value=False)
    def test_list_images_with_image_path(self, mock_exists, mock_decode):
        line = {
            'image': ['b64_1', 'b64_2'],
            'image_path': ['img1.png', 'img2.png'],
            'index': 'idx1',
        }
        result = dump_image(line, '/root')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], os.path.join('/root', 'img1.png'))
        self.assertEqual(result[1], os.path.join('/root', 'img2.png'))

    @patch('ais_bench.benchmark.datasets.mmmu.decode_base64_to_image_file')
    @patch('os.path.exists', return_value=False)
    def test_list_images_without_image_path(self, mock_exists, mock_decode):
        line = {
            'image': ['b64_1', 'b64_2'],
            'index': 'idx1',
        }
        result = dump_image(line, '/root')
        self.assertEqual(len(result), 2)
        self.assertIn('idx1_0.png', result[0])
        self.assertIn('idx1_1.png', result[1])

    @patch('ais_bench.benchmark.datasets.mmmu.decode_base64_to_image_file')
    @patch('os.path.exists', return_value=False)
    def test_str_image_with_image_path(self, mock_exists, mock_decode):
        line = {
            'image': 'b64data',
            'image_path': 'single.png',
            'index': 'idx2',
        }
        result = dump_image(line, '/root')
        self.assertEqual(len(result), 1)
        self.assertIn('single.png', result[0])

    @patch('ais_bench.benchmark.datasets.mmmu.decode_base64_to_image_file')
    @patch('os.path.exists', return_value=False)
    def test_str_image_without_image_path(self, mock_exists, mock_decode):
        line = {
            'image': 'b64data',
            'index': 'idx3',
        }
        result = dump_image(line, '/root')
        self.assertEqual(len(result), 1)
        self.assertIn('idx3.jpg', result[0])

    @patch('ais_bench.benchmark.datasets.mmmu.toliststr', return_value=['/abs/path.png'])
    @patch('os.path.exists', return_value=True)
    def test_no_image_field_with_absolute_path(self, mock_exists, mock_toliststr):
        line = {'image_path': '/abs/path.png'}
        result = dump_image(line, '/root')
        self.assertEqual(result, ['/abs/path.png'])

    @patch('ais_bench.benchmark.datasets.mmmu.toliststr', return_value=['rel.png'])
    @patch('os.path.exists', return_value=True)
    def test_no_image_field_relative_path(self, mock_exists, mock_toliststr):
        line = {'image_path': 'rel.png'}
        result = dump_image(line, '/root')
        self.assertEqual(len(result), 1)

    @patch('os.path.exists', return_value=True)
    def test_list_images_existing_files(self, mock_exists):
        line = {
            'image': ['b64_1'],
            'image_path': ['img1.png'],
            'index': 'idx',
        }
        result = dump_image(line, '/root')
        self.assertEqual(len(result), 1)

    @patch('ais_bench.benchmark.datasets.mmmu.toliststr', return_value=['a.png', 'b.png'])
    @patch('os.path.exists', return_value=False)
    def test_no_image_field_all_missing(self, mock_exists, mock_toliststr):
        line = {'image_path': 'a.png,b.png'}
        result = dump_image(line, '/root')
        self.assertEqual(len(result), 2)

    @patch('ais_bench.benchmark.datasets.mmmu.toliststr', return_value=['/abs.png'])
    @patch('os.path.exists', return_value=True)
    def test_no_image_field_absolute_exists(self, mock_exists, mock_toliststr):
        line = {'image_path': '/abs.png'}
        result = dump_image(line, '/root')
        self.assertEqual(result[0], '/abs.png')


class TestCanInferOption(unittest.TestCase):
    def test_api_failure(self):
        self.assertFalse(can_infer_option('Failed to obtain answer via API', {'A', 'B'}))

    def test_reject_help_with_images(self):
        self.assertEqual(
            can_infer_option("Sorry, I can't help with images of people yet.", {'A', 'B'}),
            'Z',
        )

    def test_reject_cant_process(self):
        self.assertEqual(
            can_infer_option("I can't process this file.", {'A', 'B'}),
            'Z',
        )

    def test_reject_sorry_no_image(self):
        self.assertEqual(
            can_infer_option("I'm sorry, but without the image provided", {'A', 'B'}),
            'Z',
        )

    def test_reject_cannot_determine(self):
        self.assertEqual(
            can_infer_option('Cannot determine the answer', {'A', 'B'}),
            'Z',
        )

    def test_clear_choice_at_end(self):
        result = can_infer_option('The answer is B', {'A', 'B', 'C'})
        self.assertEqual(result, 'B')

    def test_ambiguous_returns_false(self):
        self.assertFalse(can_infer_option('A and B and C', {'A', 'B', 'C'}))

    def test_no_match_returns_false(self):
        self.assertFalse(can_infer_option('nothing relevant', {'A', 'B', 'C'}))

    def test_single_choice_match(self):
        result = can_infer_option('I believe the answer is A', {'A', 'B', 'C'})
        self.assertEqual(result, 'A')


class TestCanInferText(unittest.TestCase):
    def test_exact_match(self):
        choices = {'A': 'Paris', 'B': 'London', 'C': 'Berlin'}
        self.assertEqual(can_infer_text('The capital is Paris', choices), 'A')

    def test_case_insensitive(self):
        choices = {'A': 'Paris', 'B': 'London'}
        self.assertEqual(can_infer_text('the answer is paris', choices), 'A')

    def test_multiple_matches(self):
        choices = {'A': 'Paris', 'B': 'the Paris'}
        self.assertFalse(can_infer_text('I like Paris and the Paris', choices))

    def test_no_match(self):
        choices = {'A': 'Paris', 'B': 'London'}
        self.assertFalse(can_infer_text('Rome', choices))

    def test_too_long_answer(self):
        choices = {'A': 'x', 'B': 'y'}
        long_answer = 'x' * 100
        self.assertFalse(can_infer_text(long_answer, choices))

    def test_numeric_choices(self):
        choices = {'A': '42', 'B': '3.14'}
        self.assertFalse(can_infer_text('The answer is 42', choices))

    def test_empty_answer(self):
        choices = {'A': 'Paris', 'B': 'London'}
        self.assertFalse(can_infer_text('', choices))


class TestCanInfer(unittest.TestCase):
    def test_option_inferred(self):
        self.assertEqual(can_infer('The answer is C', {'A', 'B', 'C'}), 'C')

    def test_text_fallback(self):
        choices = {'A': 'Paris', 'B': 'London'}
        self.assertEqual(can_infer('Paris', choices), 'A')

    def test_neither_returns_false(self):
        self.assertFalse(can_infer('gibberish', {'A': 'x', 'B': 'y', 'C': 'z'}))

    def test_numeric_answer_cast_to_str(self):
        result = can_infer(123, {'A': 'a', 'B': 'b', 'C': 'c'})
        self.assertFalse(result)


class TestSortKey(unittest.TestCase):
    def test_validation_overall_first(self):
        self.assertEqual(sort_key(('[validation]: Overall', 90))[0], 0)

    def test_dev_overall_second(self):
        self.assertEqual(sort_key(('[dev]: Overall', 85))[0], 1)

    def test_other_third(self):
        self.assertEqual(sort_key(('[test]: Overall', 80))[0], 2)

    def test_overall_before_category(self):
        k1 = sort_key(('[validation]: Overall', 90))
        k2 = sort_key(('[validation]: Math', 90))
        self.assertLess(k1, k2)

    def test_task_string_included(self):
        key = sort_key(('[validation]: Physics', 80))
        self.assertEqual(key[2], 'Physics')

    def test_returns_triple(self):
        key = sort_key(('[dev]: Chemistry', 70))
        self.assertEqual(len(key), 3)


class TestMMMUEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = MMMUEvaluator()

    def test_length_mismatch(self):
        result = self.evaluator.score(['A'], ['ref1', 'ref2'])
        self.assertIn('error', result)
        self.assertIn('different length', result['error'])

    def test_multi_choice_correct(self):
        refer = {
            'type': MMMU_MULTI_CHOICE_TYPE,
            'choices': json.dumps({'A': 'opt1', 'B': 'opt2'}),
            'answer': 'A',
            'split': 'validation',
            'category': 'Math',
        }
        result = self.evaluator.score(['Answer: A'], [refer])
        overall_key = '[validation]: Overall'
        self.assertEqual(result[overall_key], 100.0)
        self.assertTrue(result['details'][0]['correct'])

    def test_multi_choice_incorrect(self):
        refer = {
            'type': MMMU_MULTI_CHOICE_TYPE,
            'choices': json.dumps({'A': 'opt1', 'B': 'opt2'}),
            'answer': 'A',
            'split': 'validation',
            'category': 'Math',
        }
        result = self.evaluator.score(['Answer: B'], [refer])
        self.assertEqual(result['[validation]: Overall'], 0.0)
        self.assertFalse(result['details'][0]['correct'])

    def test_open_type_correct(self):
        refer = {
            'type': MMMU_OPEN_TYPE,
            'answer': 'Paris',
            'split': 'validation',
            'category': 'Geography',
        }
        result = self.evaluator.score(['ANSWER: Paris'], [refer])
        self.assertEqual(result['[validation]: Overall'], 100.0)

    def test_open_type_incorrect(self):
        refer = {
            'type': MMMU_OPEN_TYPE,
            'answer': 'Paris',
            'split': 'validation',
            'category': 'Geography',
        }
        result = self.evaluator.score(['ANSWER: London'], [refer])
        self.assertEqual(result['[validation]: Overall'], 0.0)

    def test_open_type_case_insensitive(self):
        refer = {
            'type': MMMU_OPEN_TYPE,
            'answer': 'Paris',
            'split': 'validation',
            'category': 'Geo',
        }
        result = self.evaluator.score(['ANSWER: paris'], [refer])
        self.assertEqual(result['[validation]: Overall'], 100.0)

    def test_string_reference_treated_as_open(self):
        result = self.evaluator.score(['ANSWER: hello'], ['hello'])
        self.assertEqual(result['[validation]: Overall'], 100.0)

    def test_special_characters_cleaned(self):
        refer = {
            'type': MMMU_OPEN_TYPE,
            'answer': 'test',
            'split': 'validation',
            'category': 'C',
        }
        result = self.evaluator.score(['test<|im_end|>'], [refer])
        self.assertEqual(result['[validation]: Overall'], 100.0)