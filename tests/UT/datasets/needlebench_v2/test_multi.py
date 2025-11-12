import unittest
import sys
import os
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.needlebench_v2.multi import (
        get_random_needles,
        NeedleBenchMultiDataset
    )
    NEEDLEBENCH_MULTI_AVAILABLE = True
except ImportError:
    NEEDLEBENCH_MULTI_AVAILABLE = False


class NeedleBenchMultiTestBase(unittest.TestCase):
    """NeedleBenchMulti测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not NEEDLEBENCH_MULTI_AVAILABLE:
            cls.skipTest(cls, "NeedleBenchMulti modules not available")


class TestGetRandomNeedles(NeedleBenchMultiTestBase):
    """测试get_random_needles函数"""
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"English": "Name1,Name2,Name3,Name4,Name5", "Chinese": "名字1,名字2,名字3,名字4,名字5"}')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.random')
    def test_get_random_needles_english(self, mock_random, mock_file):
        """测试获取英文随机needles"""
        mock_random.choice.return_value = 'template'
        mock_random.sample.return_value = ['Name1', 'Name2', 'Name3']
        mock_random.shuffle = lambda x: None
        
        result = get_random_needles(0, '/fake/path', 3, 'English')
        
        self.assertIn('needles', result)
        self.assertIn('answer', result)
        self.assertIn('retrieval_question', result)
        self.assertIn('last_person', result)
        self.assertIsInstance(result['needles'], list)
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"English": "Name1,Name2,Name3", "Chinese": "名字1,名字2,名字3"}')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.random')
    def test_get_random_needles_chinese(self, mock_random, mock_file):
        """测试获取中文随机needles"""
        mock_random.choice.return_value = '模板'
        mock_random.sample.return_value = ['名字1', '名字2', '名字3']
        mock_random.shuffle = lambda x: None
        
        result = get_random_needles(0, '/fake/path', 3, 'Chinese')
        
        self.assertIn('needles', result)
        self.assertIn('answer', result)
        self.assertIn('retrieval_question', result)
        self.assertIn('last_person', result)
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"English": "Name1,Name2", "Chinese": "名字1,名字2", "French": "Jean,Paul"}')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.random')
    def test_get_random_needles_unsupported_language(self, mock_random, mock_file):
        """测试不支持的语言"""
        with self.assertRaises(ValueError):
            get_random_needles(0, '/fake/path', 2, 'French')


class TestNeedleBenchMultiDataset(NeedleBenchMultiTestBase):
    """测试NeedleBenchMultiDataset类"""
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.get_data_path')
    @patch('os.path.join')
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.get_random_needles')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.tiktoken')
    def test_load(self, mock_tiktoken, mock_get_random, mock_open, mock_exists, mock_join, mock_get_path):
        """测试加载多针数据集"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        mock_exists.return_value = True
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "some text"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        mock_get_random.return_value = {
            'needles': ['needle1', 'needle2'],
            'answer': 'answer_name',
            'retrieval_question': 'test question',
            'last_person': 'last_person'
        }
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = 'decoded text'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        result = NeedleBenchMultiDataset.load(
            path='/test/path',
            length=1000,
            depth=50,
            tokenizer_model='gpt-3.5-turbo',
            file_list=['test.jsonl'],
            num_repeats_per_file=1,
            length_buffer=100,
            language='English',
            needle_file_name='names.json',
            num_needles=2,
            diff=10,
            quesiton_position='End'
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'column_names'))
        self.assertIn('prompt', result.column_names)
        self.assertIn('answer', result.column_names)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.tiktoken')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.get_data_path')
    @patch('os.path.join')
    @patch('os.path.exists')
    def test_load_file_not_exists(self, mock_exists, mock_join, mock_get_path, mock_tiktoken):
        """测试文件不存在的情况"""
        # 参数顺序：从最底部的patch到最上部的patch (exists, join, get_path, tiktoken)
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        mock_exists.return_value = False  # 文件不存在
        # tiktoken在函数开始就被调用，需要mock让它成功，这样代码才能继续到文件检查
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = []
        mock_tokenizer.decode.return_value = ''
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        # 使用file_names列表中的真实文件名，这样才会进入检查文件存在的逻辑
        with self.assertRaises(ValueError) as cm:
            NeedleBenchMultiDataset.load(
                path='/test/path',
                length=1000,
                depth=50,
                tokenizer_model='gpt-3.5-turbo',
                file_list=['PaulGrahamEssays.jsonl'],  # 使用file_names中的文件名
                num_repeats_per_file=1,
                length_buffer=100,
                language='English',
                needle_file_name='names.json',
                num_needles=2,
                diff=10
            )
        # 验证错误消息
        self.assertIn('Dataset file does not exist', str(cm.exception))
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.get_data_path')
    @patch('os.path.join')
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.get_random_needles')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.tiktoken')
    def test_load_multiple_needles(self, mock_tiktoken, mock_get_random, mock_open, mock_exists, mock_join, mock_get_path):
        """测试加载多个needles的情况"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        mock_exists.return_value = True
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "some text"}\n',
            '{"text": "more text"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        # 模拟多个needles
        mock_get_random.return_value = {
            'needles': ['needle1', 'needle2', 'needle3'],
            'answer': 'answer_name',
            'retrieval_question': 'test question',
            'last_person': 'last_person'
        }
        
        mock_tokenizer = MagicMock()
        # encode返回列表（模拟多个tokens）
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = 'decoded text'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        result = NeedleBenchMultiDataset.load(
            path='/test/path',
            length=1000,
            depth=50,
            tokenizer_model='gpt-3.5-turbo',
            file_list=['test.jsonl'],
            num_repeats_per_file=1,
            length_buffer=100,
            language='English',
            needle_file_name='names.json',
            num_needles=3,  # 多个needles
            diff=10,
            quesiton_position='End'
        )
        
        self.assertIsNotNone(result)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.get_data_path')
    @patch('os.path.join')
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.get_random_needles')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.tiktoken')
    def test_load_chinese_start(self, mock_tiktoken, mock_get_random, mock_open, mock_exists, mock_join, mock_get_path):
        """测试加载中文数据集，问题在开始位置"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        mock_exists.return_value = True
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "中文文本"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        mock_get_random.return_value = {
            'needles': ['针1', '针2'],
            'answer': '答案',
            'retrieval_question': '问题',
            'last_person': '最后的人'
        }
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = '解码文本'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        result = NeedleBenchMultiDataset.load(
            path='/test/path',
            length=1000,
            depth=50,
            tokenizer_model='gpt-3.5-turbo',
            file_list=['test.jsonl'],
            num_repeats_per_file=1,
            length_buffer=100,
            language='Chinese',
            needle_file_name='names.json',
            num_needles=2,
            diff=10,
            quesiton_position='Start'  # 问题在开始
        )
        
        self.assertIsNotNone(result)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.get_data_path')
    @patch('os.path.join')
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.get_random_needles')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.multi.tiktoken')
    def test_load_unsupported_question_position(self, mock_tiktoken, mock_get_random, mock_open, mock_exists, mock_join, mock_get_path):
        """测试不支持的问题位置"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        mock_exists.return_value = True
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "some text"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        mock_get_random.return_value = {
            'needles': ['needle1'],
            'answer': 'answer',
            'retrieval_question': 'question',
            'last_person': 'person'
        }
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = 'decoded'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        with self.assertRaises(ValueError):
            NeedleBenchMultiDataset.load(
                path='/test/path',
                length=1000,
                depth=50,
                tokenizer_model='gpt-3.5-turbo',
                file_list=['PaulGrahamEssays.jsonl'],
                num_repeats_per_file=1,
                length_buffer=100,
                language='English',
                needle_file_name='names.json',
                num_needles=2,
                diff=10,
                quesiton_position='Middle'  # 不支持的位置
            )


if __name__ == '__main__':
    unittest.main()

