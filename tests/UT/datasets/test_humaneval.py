"""Unit tests for humaneval.py"""
import json
import unittest
from unittest.mock import patch, mock_open, MagicMock

from datasets import Dataset

from ais_bench.benchmark.datasets.humaneval import (
    HumanevalDataset,
    HumanEvalEvaluator,
    HumanEvalPlusEvaluator,
    humaneval_postprocess_v2,
    humaneval_postprocess_v3,
    humaneval_internal_v2_postprocess,
    humaneval_internal_v1_postprocess,
)


class TestHumanevalDataset(unittest.TestCase):
    """测试 HumanevalDataset"""

    @patch('ais_bench.benchmark.datasets.humaneval.get_data_path', return_value='/fake/humaneval.jsonl')
    @patch('builtins.open', new_callable=mock_open, read_data='{"task_id": "test/0", "prompt": "def hello():"}\n')
    def test_load_basic(self, mock_file, mock_path):
        """测试基本加载功能"""
        ds = HumanevalDataset.load('/input')
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds[0]['task_id'], 'test/0')

    @patch('ais_bench.benchmark.datasets.humaneval.get_data_path', return_value='/fake/humaneval.jsonl')
    @patch('builtins.open', new_callable=mock_open, read_data='{"task_id": "test/0", "prompt": "def hello():"}\n')
    def test_load_with_repeats(self, mock_file, mock_path):
        """测试带重复的加载"""
        ds = HumanevalDataset.load('/input', num_repeats=3)
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0]['task_id'], 'test/0')
        self.assertEqual(ds[1]['task_id'], 'test/0')
        self.assertEqual(ds[2]['task_id'], 'test/0')


class TestHumanEvalEvaluator(unittest.TestCase):
    """测试 HumanEvalEvaluator"""

    def test_init_success(self):
        """测试初始化成功"""
        with patch.dict('sys.modules', {'human_eval': MagicMock()}):
            evaluator = HumanEvalEvaluator(k=[1, 10])
            self.assertEqual(evaluator.k, [1, 10])

    def test_init_import_error(self):
        """测试导入错误"""
        with patch.dict('sys.modules', {'human_eval': None}):
            with self.assertRaises(ImportError) as ctx:
                HumanEvalEvaluator()
            self.assertIn('human_eval', str(ctx.exception))

    def test_score_length_mismatch(self):
        """测试预测和参考长度不匹配"""
        with patch.dict('sys.modules', {'human_eval': MagicMock()}):
            evaluator = HumanEvalEvaluator()
            result = evaluator.score(['pred1'], ['ref1', 'ref2'], [])
            self.assertIn('error', result)
            self.assertIn('different length', result['error'])

    def test_score_basic(self):
        """测试基本评分功能"""
        mock_human_eval = MagicMock()
        mock_human_eval.data.HUMAN_EVAL = 'fake_path'
        mock_human_eval.data.write_jsonl = MagicMock()
        mock_human_eval.evaluation.evaluate_functional_correctness = MagicMock(return_value={1: 0.5, 10: 0.8})
        
        with patch.dict('sys.modules', {'human_eval': mock_human_eval, 'human_eval.data': mock_human_eval.data, 'human_eval.evaluation': mock_human_eval.evaluation}):
            with patch('builtins.open', mock_open(read_data='{"task_id": "test/0", "passed": true, "result": "passed"}\n')):
                evaluator = HumanEvalEvaluator(k=[1, 10])
                test_set = [{'prompt': 'def hello():'}]
                result = evaluator.score(['pred'], ['test/0'], test_set)
                self.assertIn('humaneval_1', result)
                self.assertIn('humaneval_10', result)
                self.assertEqual(result['humaneval_1'], 50.0)
                self.assertEqual(result['humaneval_10'], 80.0)
                self.assertIn('details', result)

    def test_score_multiple_predictions(self):
        """测试多个预测的评分"""
        mock_human_eval = MagicMock()
        mock_human_eval.data.HUMAN_EVAL = 'fake_path'
        mock_write = MagicMock()
        mock_human_eval.data.write_jsonl = mock_write
        mock_human_eval.evaluation.evaluate_functional_correctness = MagicMock(return_value={1: 0.5})
        
        with patch.dict('sys.modules', {'human_eval': mock_human_eval, 'human_eval.data': mock_human_eval.data, 'human_eval.evaluation': mock_human_eval.evaluation}):
            with patch('builtins.open', mock_open(read_data='{"task_id": "test/0", "passed": true, "result": "passed"}\n')):
                evaluator = HumanEvalEvaluator(k=[1])
                test_set = [{'prompt': 'def hello():'}]
                result = evaluator.score([['pred1', 'pred2']], ['test/0'], test_set)
                self.assertIn('humaneval_1', result)
                # 验证 write_jsonl 被调用，且传入了2个预测
                self.assertTrue(mock_write.called)
                call_args = mock_write.call_args[0][1]
                self.assertEqual(len(call_args), 2)


class TestHumanEvalPlusEvaluator(unittest.TestCase):
    """测试 HumanEvalPlusEvaluator"""

    def test_init_success(self):
        """测试初始化成功"""
        with patch.dict('sys.modules', {'evalplus': MagicMock()}):
            evaluator = HumanEvalPlusEvaluator(k=[1, 10])
            self.assertEqual(evaluator.k, [1, 10])

    def test_init_import_error(self):
        """测试导入错误"""
        with patch.dict('sys.modules', {'evalplus': None}):
            with self.assertRaises(ImportError) as ctx:
                HumanEvalPlusEvaluator()
            self.assertIn('evalplus', str(ctx.exception))

    def test_score_length_mismatch(self):
        """测试预测和参考长度不匹配"""
        with patch.dict('sys.modules', {'evalplus': MagicMock()}):
            evaluator = HumanEvalPlusEvaluator()
            result = evaluator.score(['pred1'], ['ref1', 'ref2'], [])
            self.assertIn('error', result)
            self.assertIn('different length', result['error'])

    def test_score_basic(self):
        """测试基本评分功能"""
        mock_evalplus = MagicMock()
        mock_evalplus.data.write_jsonl = MagicMock()
        mock_evalplus.evaluate.evaluate = MagicMock(return_value={1: 0.6, 10: 0.9})
        
        mock_results = {
            'eval': {
                'test/0': {
                    'base': [['success']],
                    'plus': [['success']],
                    'nfiles': 1
                }
            }
        }
        
        with patch.dict('sys.modules', {'evalplus': mock_evalplus, 'evalplus.data': mock_evalplus.data, 'evalplus.evaluate': mock_evalplus.evaluate}):
            with patch('builtins.open', mock_open(read_data=json.dumps(mock_results))):
                evaluator = HumanEvalPlusEvaluator(k=[1, 10])
                test_set = [{'prompt': 'def hello():'}]
                result = evaluator.score(['pred'], ['test/0'], test_set)
                self.assertIn('humaneval_plus_1', result)
                self.assertIn('humaneval_plus_10', result)
                self.assertEqual(result['humaneval_plus_1'], 60.0)
                self.assertEqual(result['humaneval_plus_10'], 90.0)
                self.assertIn('details', result)

    def test_score_with_warning(self):
        """测试带警告的评分（多文件情况）"""
        mock_evalplus = MagicMock()
        mock_evalplus.data.write_jsonl = MagicMock()
        mock_evalplus.evaluate.evaluate = MagicMock(return_value={1: 0.5})
        
        mock_results = {
            'eval': {
                'test/0': {
                    'base': [['success']],
                    'plus': [['success']],
                    'nfiles': 2  # 多个文件
                }
            }
        }
        
        with patch.dict('sys.modules', {'evalplus': mock_evalplus, 'evalplus.data': mock_evalplus.data, 'evalplus.evaluate': mock_evalplus.evaluate}):
            with patch('builtins.open', mock_open(read_data=json.dumps(mock_results))):
                evaluator = HumanEvalPlusEvaluator(k=[1])
                test_set = [{'prompt': 'def hello():'}]
                result = evaluator.score(['pred'], ['test/0'], test_set)
                self.assertIn('details', result)
                self.assertIn('warning', result['details']['0'])


class TestPostprocessFunctions(unittest.TestCase):
    """测试后处理函数"""

    def test_humaneval_postprocess_v2_with_code_block(self):
        """测试 v2 后处理：提取第一个代码块"""
        text = "Some text\n```python\ncode1\n```\nMore text\n```python\ncode2\n```"
        result = humaneval_postprocess_v2(text)
        self.assertEqual(result, 'code1\n')

    def test_humaneval_postprocess_v2_no_code_block(self):
        """测试 v2 后处理：无代码块"""
        text = "No code blocks here"
        result = humaneval_postprocess_v2(text)
        self.assertEqual(result, text)

    def test_humaneval_postprocess_v3_with_code_block(self):
        """测试 v3 后处理：提取最后一个代码块"""
        text = "Some text\n```python\ncode1\n```\nMore text\n```python\ncode2\n```"
        result = humaneval_postprocess_v3(text)
        self.assertEqual(result, 'code2\n')

    def test_humaneval_postprocess_v3_no_code_block(self):
        """测试 v3 后处理：无代码块"""
        text = "No code blocks here"
        result = humaneval_postprocess_v3(text)
        self.assertEqual(result, text)

    def test_humaneval_internal_v2_postprocess_basic(self):
        """测试 internal v2 后处理：基本功能"""
        text = "    line1\n    line2\n\n\nline3"
        result = humaneval_internal_v2_postprocess(text)
        self.assertEqual(result, '    line1\n    line2')

    def test_humaneval_internal_v2_postprocess_with_three_spaces(self):
        """测试 internal v2 后处理：三个空格开头"""
        text = "   line1\n   line2"
        result = humaneval_internal_v2_postprocess(text)
        # 以3个空格开头会在前面加1个空格，然后按行处理
        # 第一行变成 " line1"（加了1个空格），第二行仍是 "   line2"
        self.assertEqual(result, '    line1\n   line2')

    def test_humaneval_internal_v2_postprocess_with_code_fence(self):
        """测试 internal v2 后处理：代码围栏"""
        text = "    line1\n    line2\n```\nline3"
        result = humaneval_internal_v2_postprocess(text)
        self.assertEqual(result, '    line1\n    line2')

    def test_humaneval_internal_v2_postprocess_with_non_space_start(self):
        """测试 internal v2 后处理：非空格开头"""
        text = "    line1\n    line2\nline3\n    line4"
        result = humaneval_internal_v2_postprocess(text)
        self.assertEqual(result, '    line1\n    line2')

    def test_humaneval_internal_v1_postprocess_with_code_block(self):
        """测试 internal v1 后处理：带代码块"""
        text = "```python\ndef hello():\n    return 'world'\n```"
        result = humaneval_internal_v1_postprocess(text)
        self.assertIn('return', result)

    def test_humaneval_internal_v1_postprocess_with_eval_string(self):
        """测试 internal v1 后处理：可 eval 的字符串"""
        text = '"    return True"'
        result = humaneval_internal_v1_postprocess(text)
        self.assertIn('return True', result)

    def test_humaneval_internal_v1_postprocess_with_import(self):
        """测试 internal v1 后处理：带 import 语句"""
        text = "from math import sqrt\nimport os\ndef func():\n    return 1"
        result = humaneval_internal_v1_postprocess(text)
        self.assertNotIn('from math', result)
        self.assertNotIn('import os', result)

    def test_humaneval_internal_v1_postprocess_with_def(self):
        """测试 internal v1 后处理：带 def 语句"""
        text = "def func():\n    return 1\n    return 2"
        result = humaneval_internal_v1_postprocess(text)
        self.assertIn('return 1', result)

    def test_humaneval_internal_v1_postprocess_with_indentation(self):
        """测试 internal v1 后处理：缩进处理"""
        text = "  return 1\n  return 2"
        result = humaneval_internal_v1_postprocess(text)
        self.assertTrue(result.startswith('    '))

    def test_humaneval_internal_v1_postprocess_with_leading_space_reduction(self):
        """测试 internal v1 后处理：前导空格减少"""
        text = "    line1\n    line2\n  line3\n    line4"
        result = humaneval_internal_v1_postprocess(text)
        # 当前导空格减少时，代码块结束
        self.assertIn('line1', result)
        self.assertIn('line2', result)

    def test_humaneval_internal_v1_postprocess_empty_lines(self):
        """测试 internal v1 后处理：空行处理"""
        text = "    line1\n\n    line2\n\n\n    line3"
        result = humaneval_internal_v1_postprocess(text)
        # 空行应该被移除
        self.assertNotIn('\n\n', result)

    def test_humaneval_internal_v1_postprocess_comments_and_strings(self):
        """测试 internal v1 后处理：注释和字符串"""
        text = "    # comment\n    'string'\n    \"another\"\n    code"
        result = humaneval_internal_v1_postprocess(text)
        self.assertIn('code', result)


if __name__ == '__main__':
    unittest.main()

