import unittest
import sys
import os
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only import (
        NeedleBenchATCDataset,
        clean_atc_answer,
        needlebench_atc_postprocess_v2,
        NeedleBenchATCEvaluator,
        relationship_templates_zh_CN,
        relationship_templates_en,
        relationship_terms_zh_CN,
        relationship_terms_en
    )
    NEEDLEBENCH_ATC_ELDER_AVAILABLE = True
except ImportError:
    NEEDLEBENCH_ATC_ELDER_AVAILABLE = False


class NeedleBenchATCElderTestBase(unittest.TestCase):
    """NeedleBenchATCElder测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not NEEDLEBENCH_ATC_ELDER_AVAILABLE:
            cls.skipTest(cls, "NeedleBenchATCElder modules not available")


class TestConstants(NeedleBenchATCElderTestBase):
    """测试常量"""
    
    def test_relationship_terms(self):
        """测试关系术语"""
        self.assertIsInstance(relationship_terms_zh_CN, list)
        self.assertIsInstance(relationship_terms_en, list)
        self.assertIn('父亲', relationship_terms_zh_CN)
        self.assertIn('father', relationship_terms_en)
    
    def test_relationship_templates(self):
        """测试关系模板"""
        self.assertIsInstance(relationship_templates_zh_CN, list)
        self.assertIsInstance(relationship_templates_en, list)
        self.assertGreater(len(relationship_templates_zh_CN), 0)
        self.assertGreater(len(relationship_templates_en), 0)


class TestNeedleBenchATCDataset(NeedleBenchATCElderTestBase):
    """测试NeedleBenchATCDataset类（elder_only版本）"""
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only.random')
    def test_load(self, mock_random, mock_open, mock_join, mock_get_path):
        """测试加载数据集"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = '{"English": "Name1,Name2,Name3,Name4", "Chinese": "名字1,名字2,名字3,名字4"}'
        mock_open.return_value = mock_file_handle
        
        mock_random.choice.return_value = 'template'
        mock_random.sample.return_value = ['Name1', 'Name2', 'Name3']
        mock_random.shuffle = lambda x: None
        
        result = NeedleBenchATCDataset.load(
            path='/test/path',
            file_name='names.json',
            num_needles=2,
            language='English',
            repeats=1
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'column_names'))
        self.assertIn('prompt', result.column_names)
        self.assertIn('answer', result.column_names)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only.random')
    def test_load_chinese(self, mock_random, mock_open, mock_join, mock_get_path):
        """测试加载中文数据集"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = '{"English": "Name1", "Chinese": "名字1,名字2,名字3,名字4"}'
        mock_open.return_value = mock_file_handle
        
        mock_random.choice.return_value = '模板'
        mock_random.sample.return_value = ['名字1', '名字2', '名字3']
        mock_random.shuffle = lambda x: None
        
        result = NeedleBenchATCDataset.load(
            path='/test/path',
            file_name='names.json',
            num_needles=2,
            language='Chinese',
            repeats=1
        )
        
        self.assertIsNotNone(result)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    def test_load_unsupported_language(self, mock_open, mock_join, mock_get_path):
        """测试不支持的语言"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = '{"English": "Name1", "Chinese": "名字1"}'
        mock_open.return_value = mock_file_handle
        
        with self.assertRaises(Exception):
            NeedleBenchATCDataset.load(
                path='/test/path',
                file_name='names.json',
                num_needles=1,
                language='French',
                repeats=1
            )


class TestCleanAtcAnswer(NeedleBenchATCElderTestBase):
    """测试clean_atc_answer函数"""
    
    def test_clean_atc_answer_normal(self):
        """测试正常清理答案"""
        text = '\\boxed{John Doe}'
        result = clean_atc_answer(text)
        self.assertIsInstance(result, str)
        self.assertNotIn('\\boxed', result)
    
    def test_clean_atc_answer_none(self):
        """测试None或空字符串"""
        self.assertEqual(clean_atc_answer('None'), 'None')
        self.assertEqual(clean_atc_answer(''), 'None')
        self.assertEqual(clean_atc_answer(None), 'None')
    
    def test_clean_atc_answer_with_latex(self):
        """测试包含LaTeX命令的答案"""
        text = '\\text{John} \\boxed{Doe}'
        result = clean_atc_answer(text)
        self.assertIsInstance(result, str)
    
    def test_clean_atc_answer_with_quotes(self):
        """测试包含引号的答案"""
        text = '"John Doe"'
        result = clean_atc_answer(text)
        self.assertNotIn('"', result)
        self.assertNotIn("'", result)
    
    def test_clean_atc_answer_with_extra_spaces(self):
        """测试多余空格"""
        text = '  John   Doe  '
        result = clean_atc_answer(text)
        self.assertIsInstance(result, str)


class TestNeedlebenchAtcPostprocessV2(NeedleBenchATCElderTestBase):
    """测试needlebench_atc_postprocess_v2函数"""
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only.extract_boxed_answer')
    def test_postprocess_with_boxed_answer(self, mock_extract):
        """测试有boxed答案的后处理"""
        mock_extract.return_value = 'John Doe'
        
        result = needlebench_atc_postprocess_v2('some text with \\boxed{John Doe}')
        
        self.assertIsInstance(result, str)
        mock_extract.assert_called_once()
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only.extract_boxed_answer')
    def test_postprocess_without_boxed_answer(self, mock_extract):
        """测试没有boxed答案的后处理"""
        mock_extract.return_value = None
        
        result = needlebench_atc_postprocess_v2('some text without boxed answer')
        
        self.assertEqual(result, 'None')
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only.extract_boxed_answer')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only.clean_atc_answer')
    def test_postprocess_calls_clean(self, mock_clean, mock_extract):
        """测试后处理调用clean函数"""
        mock_extract.return_value = 'raw answer'
        mock_clean.return_value = 'cleaned answer'
        
        result = needlebench_atc_postprocess_v2('text')
        
        self.assertEqual(result, 'cleaned answer')
        mock_clean.assert_called_once_with('raw answer')


class TestNeedleBenchATCEvaluator(NeedleBenchATCElderTestBase):
    """测试NeedleBenchATCEvaluator类"""
    
    def test_score_correct(self):
        """测试评分（正确答案）"""
        evaluator = NeedleBenchATCEvaluator()
        
        predictions = ['John Doe']
        gold = ['John Doe']
        
        result = evaluator.score(predictions, gold)
        
        self.assertIn('score', result)
        self.assertEqual(result['score'], 100.0)
        self.assertIn('details', result)
    
    def test_score_incorrect(self):
        """测试评分（错误答案）"""
        evaluator = NeedleBenchATCEvaluator()
        
        predictions = ['Jane Doe']
        gold = ['John Doe']
        
        result = evaluator.score(predictions, gold)
        
        self.assertIn('score', result)
        self.assertEqual(result['score'], 0.0)
        self.assertIn('details', result)
    
    def test_score_with_whitespace(self):
        """测试评分（带空格）"""
        evaluator = NeedleBenchATCEvaluator()
        
        predictions = ['  John Doe  ']
        gold = ['John Doe']
        
        result = evaluator.score(predictions, gold)
        
        self.assertIn('score', result)
        self.assertEqual(result['score'], 100.0)
    
    def test_score_different_lengths(self):
        """测试不同长度的预测和黄金标准"""
        evaluator = NeedleBenchATCEvaluator()
        
        predictions = ['pred1']
        gold = ['gold1', 'gold2']
        
        result = evaluator.score(predictions, gold)
        
        self.assertIn('error', result)
    
    def test_score_empty_predictions(self):
        """测试空预测列表"""
        evaluator = NeedleBenchATCEvaluator()
        
        predictions = []
        gold = []
        
        result = evaluator.score(predictions, gold)
        
        self.assertIn('score', result)
        self.assertEqual(result['score'], 0.0)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only.os.environ')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only.random')
    @patch('huggingface_hub.snapshot_download')
    def test_load_hf_dataset_source(self, mock_snapshot, mock_random, mock_open, mock_join, mock_get_path, mock_environ):
        """测试从HuggingFace加载数据集"""
        mock_environ.get.return_value = 'HF'
        mock_snapshot.return_value = '/hf/path'
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.read.return_value = '{"English": "Name1,Name2,Name3,Name4", "Chinese": "名字1,名字2,名字3,名字4"}'
        mock_open.return_value = mock_file_handle
        
        mock_random.choice.return_value = 'father'
        mock_random.sample.return_value = ['Name1', 'Name2', 'Name3']
        mock_random.shuffle = lambda x: None
        
        from ais_bench.benchmark.datasets.needlebench_v2.atc_elder_only import NeedleBenchATCDataset
        result = NeedleBenchATCDataset.load(
            path='/test/path',
            file_name='names.json',
            num_needles=2,
            language='English',
            repeats=1
        )
        
        self.assertIsNotNone(result)
        mock_snapshot.assert_called_once()


if __name__ == '__main__':
    unittest.main()

