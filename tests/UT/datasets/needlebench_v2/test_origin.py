import unittest
import sys
import os
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.needlebench_v2.origin import (
        get_random_line_by_language,
        NeedleBenchOriginDataset,
        NeedleBenchOriginEvaluator,
        needlebench_postprocess,
        needlebench_dataset_postprocess
    )
    NEEDLEBENCH_ORIGIN_AVAILABLE = True
except ImportError:
    NEEDLEBENCH_ORIGIN_AVAILABLE = False


class NeedleBenchOriginTestBase(unittest.TestCase):
    """NeedleBenchOrigin测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not NEEDLEBENCH_ORIGIN_AVAILABLE:
            cls.skipTest(cls, "NeedleBenchOrigin modules not available")


class TestGetRandomLineByLanguage(NeedleBenchOriginTestBase):
    """测试get_random_line_by_language函数"""
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"needle": "test1", "retrieval_question": "q1", "arg2": "key1", "language": "English"}\n{"needle": "test2", "retrieval_question": "q2", "arg2": "key2", "language": "Chinese"}\n')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.random')
    def test_get_random_line_by_language(self, mock_random, mock_file):
        """测试获取随机行（指定语言）"""
        mock_random.choice.return_value = {
            'needle': 'test1',
            'retrieval_question': 'q1',
            'arg2': 'key1',
            'language': 'English'
        }
        
        result = get_random_line_by_language(0, '/fake/path', 'English')
        
        self.assertIsNotNone(result)
        self.assertIn('needle', result)
        self.assertIn('retrieval_question', result)
        self.assertIn('keyword', result)
    
    @patch('builtins.open', new_callable=mock_open, read_data='')
    def test_get_random_line_by_language_no_lines(self, mock_file):
        """测试没有匹配语言的行时返回None"""
        result = get_random_line_by_language(0, '/fake/path', 'English')
        self.assertIsNone(result)


class TestNeedleBenchOriginDataset(NeedleBenchOriginTestBase):
    """测试NeedleBenchOriginDataset类"""
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.get_random_line_by_language')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.tiktoken')
    def test_load(self, mock_tiktoken, mock_get_random, mock_open, mock_join, mock_get_path):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        # 模拟文件内容
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "some text content"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        # 模拟needle数据
        mock_get_random.return_value = {
            'needle': 'test needle',
            'retrieval_question': 'test question',
            'keyword': 'test_keyword'
        }
        
        # 模拟tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = 'decoded text'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        result = NeedleBenchOriginDataset.load(
            path='/test/path',
            length=1000,
            depth=50,
            tokenizer_model='gpt-3.5-turbo',
            file_list=['PaulGrahamEssays.jsonl'],
            num_repeats_per_file=1,
            length_buffer=100,
            language='English',
            needle_file_name='needles.jsonl',
            quesiton_position='End'
        )
        
        self.assertIsNotNone(result)
        # Dataset object; check column names present
        self.assertTrue(hasattr(result, 'column_names'))
        self.assertIn('prompt', result.column_names)
        self.assertIn('answer', result.column_names)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.get_random_line_by_language')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.tiktoken')
    def test_load_chinese(self, mock_tiktoken, mock_get_random, mock_open, mock_join, mock_get_path):
        """测试加载中文数据集"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "一些中文内容"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        mock_get_random.return_value = {
            'needle': '测试针',
            'retrieval_question': '测试问题',
            'keyword': '测试关键词'
        }
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = '解码文本'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        result = NeedleBenchOriginDataset.load(
            path='/test/path',
            length=1000,
            depth=50,
            tokenizer_model='gpt-3.5-turbo',
            file_list=['PaulGrahamEssays.jsonl'],
            num_repeats_per_file=1,
            length_buffer=100,
            language='Chinese',
            needle_file_name='needles.jsonl',
            quesiton_position='Start'
        )
        
        self.assertIsNotNone(result)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.get_random_line_by_language')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.tiktoken')
    def test_load_question_position_start_english(self, mock_tiktoken, mock_get_random, mock_open, mock_join, mock_get_path):
        """测试加载数据集，问题在开始位置（英文）"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "some text content"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        mock_get_random.return_value = {
            'needle': 'test needle',
            'retrieval_question': 'test question',
            'keyword': 'test_keyword'
        }
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = 'decoded text'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        result = NeedleBenchOriginDataset.load(
            path='/test/path',
            length=1000,
            depth=50,
            tokenizer_model='gpt-3.5-turbo',
            file_list=['PaulGrahamEssays.jsonl'],
            num_repeats_per_file=1,
            length_buffer=100,
            language='English',
            needle_file_name='needles.jsonl',
            quesiton_position='Start'  # 问题在开始
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'column_names'))
        self.assertIn('prompt', result.column_names)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.get_random_line_by_language')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.tiktoken')
    def test_load_question_position_start_chinese(self, mock_tiktoken, mock_get_random, mock_open, mock_join, mock_get_path):
        """测试加载数据集，问题在开始位置（中文）"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "中文内容"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        mock_get_random.return_value = {
            'needle': '测试针',
            'retrieval_question': '测试问题',
            'keyword': '测试关键词'
        }
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = '解码文本'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        result = NeedleBenchOriginDataset.load(
            path='/test/path',
            length=1000,
            depth=50,
            tokenizer_model='gpt-3.5-turbo',
            file_list=['PaulGrahamEssays.jsonl'],
            num_repeats_per_file=1,
            length_buffer=100,
            language='Chinese',
            needle_file_name='needles.jsonl',
            quesiton_position='Start'  # 问题在开始
        )
        
        self.assertIsNotNone(result)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.get_random_line_by_language')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.tiktoken')
    def test_load_unsupported_question_position(self, mock_tiktoken, mock_get_random, mock_open, mock_join, mock_get_path):
        """测试不支持的问题位置"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "some text"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        mock_get_random.return_value = {
            'needle': 'test',
            'retrieval_question': 'question',
            'keyword': 'keyword'
        }
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = 'decoded'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        with self.assertRaises(ValueError):
            NeedleBenchOriginDataset.load(
                path='/test/path',
                length=1000,
                depth=50,
                tokenizer_model='gpt-3.5-turbo',
                file_list=['PaulGrahamEssays.jsonl'],
                num_repeats_per_file=1,
                length_buffer=100,
                language='English',
                needle_file_name='needles.jsonl',
                quesiton_position='Middle'  # 不支持的位置
            )
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.get_random_line_by_language')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.origin.tiktoken')
    def test_load_unsupported_language(self, mock_tiktoken, mock_get_random, mock_open, mock_join, mock_get_path):
        """测试不支持的语言"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "some text"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        mock_get_random.return_value = {
            'needle': 'test',
            'retrieval_question': 'question',
            'keyword': 'keyword'
        }
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = 'decoded'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        with self.assertRaises(ValueError):
            NeedleBenchOriginDataset.load(
                path='/test/path',
                length=1000,
                depth=50,
                tokenizer_model='gpt-3.5-turbo',
                file_list=['PaulGrahamEssays.jsonl'],
                num_repeats_per_file=1,
                length_buffer=100,
                language='French',
                needle_file_name='needles.jsonl'
            )


class TestNeedleBenchOriginEvaluator(NeedleBenchOriginTestBase):
    """测试NeedleBenchOriginEvaluator类"""
    
    def test_score_with_keyword_match(self):
        """测试评分（关键字匹配）"""
        evaluator = NeedleBenchOriginEvaluator()
        
        predictions = ['some text with test_keyword in it']
        gold = ['test needle*test_keyword']
        
        result = evaluator.score(predictions, gold)
        
        self.assertIn('score', result)
        self.assertEqual(result['score'], 100.0)
        self.assertIn('details', result)
    
    def test_score_without_keyword_match(self):
        """测试评分（无关键字匹配）"""
        evaluator = NeedleBenchOriginEvaluator()
        
        predictions = ['some text without keyword']
        gold = ['test needle*test_keyword']
        
        result = evaluator.score(predictions, gold)
        
        self.assertIn('score', result)
        self.assertEqual(result['score'], 0.0)
    
    def test_score_different_lengths(self):
        """测试不同长度的预测和黄金标准"""
        evaluator = NeedleBenchOriginEvaluator()
        
        predictions = ['pred1']
        gold = ['gold1', 'gold2']
        
        result = evaluator.score(predictions, gold)
        
        self.assertIn('error', result)


class TestPostprocessFunctions(NeedleBenchOriginTestBase):
    """测试后处理函数"""
    
    def test_needlebench_postprocess(self):
        """测试needlebench_postprocess函数"""
        text = 'test text'
        result = needlebench_postprocess(text)
        self.assertEqual(result, text)
    
    def test_needlebench_dataset_postprocess(self):
        """测试needlebench_dataset_postprocess函数"""
        text = 'test dataset text'
        result = needlebench_dataset_postprocess(text)
        self.assertEqual(result, text)


if __name__ == '__main__':
    unittest.main()

