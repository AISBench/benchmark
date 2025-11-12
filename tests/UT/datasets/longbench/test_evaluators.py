import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.longbench.evaluators import (
        normalize_answer,
        normalize_zh_answer,
        LongBenchF1Evaluator,
        LongBenchCountEvaluator,
        LongBenchRetrievalEvaluator,
        LongBenchRougeEvaluator,
        LongBenchCodeSimEvaluator,
        LongBenchClassificationEvaluator
    )
    LONGBENCH_EVALUATORS_AVAILABLE = True
except ImportError:
    LONGBENCH_EVALUATORS_AVAILABLE = False


class LongBenchEvaluatorsTestBase(unittest.TestCase):
    """LongBench评估器测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not LONGBENCH_EVALUATORS_AVAILABLE:
            cls.skipTest(cls, "LongBench evaluators modules not available")


class TestNormalizeAnswer(LongBenchEvaluatorsTestBase):
    """测试normalize_answer函数"""
    
    def test_normalize_answer(self):
        """测试normalize_answer函数"""
        text = "The Quick Brown Fox"
        result = normalize_answer(text)
        self.assertEqual(result, "quick brown fox")
    
    def test_normalize_answer_with_punctuation(self):
        """测试normalize_answer函数处理标点符号"""
        text = "Hello, World!"
        result = normalize_answer(text)
        self.assertEqual(result, "hello world")
    
    def test_normalize_answer_with_articles(self):
        """测试normalize_answer函数去除冠词"""
        text = "A quick brown fox"
        result = normalize_answer(text)
        self.assertNotIn("a", result)
    
    def test_normalize_answer_extra_whitespace(self):
        """测试normalize_answer函数处理多余空白"""
        text = "  hello   world  "
        result = normalize_answer(text)
        self.assertEqual(result, "hello world")


class TestNormalizeZhAnswer(LongBenchEvaluatorsTestBase):
    """测试normalize_zh_answer函数"""
    
    def test_normalize_zh_answer(self):
        """测试normalize_zh_answer函数"""
        text = "你好，世界"
        result = normalize_zh_answer(text)
        self.assertIsInstance(result, str)
    
    def test_normalize_zh_answer_with_punctuation(self):
        """测试normalize_zh_answer函数处理中文标点"""
        text = "你好，世界！"
        result = normalize_zh_answer(text)
        # 应该去除标点符号
        self.assertIsInstance(result, str)


class TestLongBenchF1Evaluator(LongBenchEvaluatorsTestBase):
    """测试LongBenchF1Evaluator类"""
    
    def test_init_en(self):
        """测试英文评估器初始化"""
        evaluator = LongBenchF1Evaluator(language='en')
        self.assertEqual(evaluator.language, 'en')
    
    def test_init_zh(self):
        """测试中文评估器初始化"""
        evaluator = LongBenchF1Evaluator(language='zh')
        self.assertEqual(evaluator.language, 'zh')
    
    @patch('ais_bench.benchmark.datasets.longbench.evaluators.jieba.cut')
    def test_score_english(self, mock_jieba):
        """测试英文F1评分"""
        evaluator = LongBenchF1Evaluator(language='en')
        
        predictions = ['The quick brown fox']
        references = [['The quick brown fox jumps']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], float)
        self.assertGreaterEqual(result['score'], 0)
        self.assertLessEqual(result['score'], 100)
    
    @patch('ais_bench.benchmark.datasets.longbench.evaluators.jieba.cut')
    def test_score_chinese(self, mock_jieba):
        """测试中文F1评分"""
        mock_jieba.return_value = ['你好', '世界']
        
        evaluator = LongBenchF1Evaluator(language='zh')
        
        predictions = ['你好世界']
        references = [['你好世界']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], float)
    
    def test_score_multiple_references(self):
        """测试多个参考答案的情况"""
        evaluator = LongBenchF1Evaluator(language='en')
        
        predictions = ['hello world']
        references = [['hello world', 'world hello']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertGreaterEqual(result['score'], 0)


class TestLongBenchCountEvaluator(LongBenchEvaluatorsTestBase):
    """测试LongBenchCountEvaluator类"""
    
    def test_init(self):
        """测试评估器初始化"""
        evaluator = LongBenchCountEvaluator()
        self.assertIsNotNone(evaluator)
    
    def test_score_with_match(self):
        """测试计数评估器匹配的情况"""
        evaluator = LongBenchCountEvaluator()
        
        predictions = ['There are 5 apples']
        references = [['5']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertGreaterEqual(result['score'], 0)
    
    def test_score_without_match(self):
        """测试计数评估器不匹配的情况"""
        evaluator = LongBenchCountEvaluator()
        
        predictions = ['There are 5 apples']
        references = [['10']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], float)
    
    def test_score_no_numbers(self):
        """测试预测中没有数字的情况"""
        evaluator = LongBenchCountEvaluator()
        
        predictions = ['There are no numbers']
        references = [['5']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertEqual(result['score'], 0.0)


class TestLongBenchRetrievalEvaluator(LongBenchEvaluatorsTestBase):
    """测试LongBenchRetrievalEvaluator类"""
    
    def test_init_en(self):
        """测试英文检索评估器初始化"""
        evaluator = LongBenchRetrievalEvaluator(language='en')
        self.assertEqual(evaluator.language, 'en')
    
    def test_init_zh(self):
        """测试中文检索评估器初始化"""
        evaluator = LongBenchRetrievalEvaluator(language='zh')
        self.assertEqual(evaluator.language, 'zh')
    
    def test_score_english(self):
        """测试英文检索评分"""
        evaluator = LongBenchRetrievalEvaluator(language='en')
        
        predictions = ['Paragraph 1 and Paragraph 2']
        references = [['Paragraph 1']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], float)
    
    def test_score_chinese(self):
        """测试中文检索评分"""
        evaluator = LongBenchRetrievalEvaluator(language='zh')
        
        predictions = ['段落1和段落2']
        references = [['段落1']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], float)
    
    def test_score_no_match(self):
        """测试检索评估器无匹配的情况"""
        evaluator = LongBenchRetrievalEvaluator(language='en')
        
        predictions = ['No paragraph number']
        references = [['Paragraph 1']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertEqual(result['score'], 0.0)


class TestLongBenchRougeEvaluator(LongBenchEvaluatorsTestBase):
    """测试LongBenchRougeEvaluator类"""
    
    def test_init_en(self):
        """测试英文Rouge评估器初始化"""
        evaluator = LongBenchRougeEvaluator(language='en')
        self.assertEqual(evaluator.language, 'en')
    
    def test_init_zh(self):
        """测试中文Rouge评估器初始化"""
        evaluator = LongBenchRougeEvaluator(language='zh')
        self.assertEqual(evaluator.language, 'zh')
    
    @patch('ais_bench.benchmark.datasets.longbench.evaluators.Rouge')
    @patch('ais_bench.benchmark.datasets.longbench.evaluators.jieba.cut')
    def test_score_english(self, mock_jieba, mock_rouge_class):
        """测试英文Rouge评分"""
        mock_rouge_instance = MagicMock()
        mock_rouge_instance.get_scores.return_value = {'rouge-l': {'f': 0.8}}
        mock_rouge_class.return_value = mock_rouge_instance
        
        evaluator = LongBenchRougeEvaluator(language='en')
        
        predictions = ['The quick brown fox']
        references = [['The quick brown fox jumps']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], float)
    
    @patch('ais_bench.benchmark.datasets.longbench.evaluators.Rouge')
    @patch('ais_bench.benchmark.datasets.longbench.evaluators.jieba.cut')
    def test_score_chinese(self, mock_jieba, mock_rouge_class):
        """测试中文Rouge评分"""
        mock_jieba.return_value = ['你好', '世界']
        mock_rouge_instance = MagicMock()
        mock_rouge_instance.get_scores.return_value = {'rouge-l': {'f': 0.8}}
        mock_rouge_class.return_value = mock_rouge_instance
        
        evaluator = LongBenchRougeEvaluator(language='zh')
        
        predictions = ['你好世界']
        references = [['你好世界']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], float)
    
    @patch('ais_bench.benchmark.datasets.longbench.evaluators.Rouge')
    def test_score_exception(self, mock_rouge_class):
        """测试Rouge评分异常情况"""
        mock_rouge_instance = MagicMock()
        mock_rouge_instance.get_scores.side_effect = Exception('Rouge error')
        mock_rouge_class.return_value = mock_rouge_instance
        
        evaluator = LongBenchRougeEvaluator(language='en')
        
        predictions = ['test']
        references = [['test']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertEqual(result['score'], 0.0)


class TestLongBenchCodeSimEvaluator(LongBenchEvaluatorsTestBase):
    """测试LongBenchCodeSimEvaluator类"""
    
    def test_init(self):
        """测试评估器初始化"""
        evaluator = LongBenchCodeSimEvaluator()
        self.assertIsNotNone(evaluator)
    
    @patch('ais_bench.benchmark.datasets.longbench.evaluators.fuzz.ratio')
    def test_score(self, mock_fuzz):
        """测试代码相似度评分"""
        mock_fuzz.return_value = 80
        
        evaluator = LongBenchCodeSimEvaluator()
        
        predictions = ['def test(): pass']
        references = [['def test(): pass']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], float)
        mock_fuzz.assert_called()
    
    def test_score_with_comments(self):
        """测试代码相似度评分（包含注释）"""
        evaluator = LongBenchCodeSimEvaluator()
        
        predictions = ['# comment\n`code`\ndef test(): pass']
        references = [['def test(): pass']]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], float)


class TestLongBenchClassificationEvaluator(LongBenchEvaluatorsTestBase):
    """测试LongBenchClassificationEvaluator类"""
    
    def test_init(self):
        """测试评估器初始化"""
        evaluator = LongBenchClassificationEvaluator()
        self.assertIsNotNone(evaluator)
    
    def test_score_exact_match(self):
        """测试分类评估器完全匹配"""
        evaluator = LongBenchClassificationEvaluator()
        
        predictions = ['positive']
        references = [{
            'answers': ['positive'],
            'all_classes': ['positive', 'negative']
        }]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], float)
    
    def test_score_partial_match(self):
        """测试分类评估器部分匹配"""
        evaluator = LongBenchClassificationEvaluator()
        
        predictions = ['positive']
        references = [{
            'answers': ['positive'],
            'all_classes': ['positive', 'negative', 'neutral']
        }]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], float)
    
    def test_score_no_em_match(self):
        """测试分类评估器无精确匹配的情况"""
        evaluator = LongBenchClassificationEvaluator()
        
        # 预测中包含类名但不完全匹配参考答案
        predictions = ['positive']
        references = [{
            'answers': ['negative'],
            'all_classes': ['positive', 'negative']
        }]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], float)
    
    def test_score_no_match(self):
        """测试分类评估器无匹配"""
        evaluator = LongBenchClassificationEvaluator()
        
        predictions = ['unknown']
        references = [{
            'answers': ['positive'],
            'all_classes': ['positive', 'negative']
        }]
        
        result = evaluator.score(predictions, references)
        
        self.assertIn('score', result)
        self.assertIsInstance(result['score'], float)


if __name__ == '__main__':
    unittest.main()

