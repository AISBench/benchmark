"""Unit tests for math.py"""
import importlib
import json
import unittest
from unittest.mock import patch, mock_open, MagicMock

from datasets import Dataset, DatasetDict

# Import the math module explicitly using importlib to avoid conflict with built-in math module
math_module = importlib.import_module('ais_bench.benchmark.datasets.math')
from ais_bench.benchmark.datasets.math import (
    last_boxed_only_string,
    remove_boxed,
    extract_boxed_answer,
    normalize_final_answer,
    extract_answer,
    MATHDataset,
    math_postprocess,
    math_judement_preprocess,
    math_postprocess_v2,
    MATHEvaluator,
    MATHAgentEvaluator,
)


class TestBoxedFunctions(unittest.TestCase):
    """测试 boxed 相关函数"""

    def test_last_boxed_only_string_with_boxed(self):
        """测试提取 boxed"""
        text = "The answer is \\boxed{42}"
        result = last_boxed_only_string(text)
        self.assertEqual(result, '\\boxed{42}')

    def test_last_boxed_only_string_with_fbox(self):
        """测试提取 fbox"""
        text = "The answer is \\fbox{42}"
        result = last_boxed_only_string(text)
        self.assertEqual(result, '\\fbox{42}')

    def test_last_boxed_only_string_no_box(self):
        """测试无 box"""
        text = "No box here"
        result = last_boxed_only_string(text)
        self.assertIsNone(result)

    def test_last_boxed_only_string_unclosed(self):
        """测试未闭合的 box"""
        text = "\\boxed{unclosed"
        result = last_boxed_only_string(text)
        self.assertIsNone(result)

    def test_last_boxed_only_string_nested(self):
        """测试嵌套的 box"""
        text = "\\boxed{a{b}c}"
        result = last_boxed_only_string(text)
        self.assertEqual(result, '\\boxed{a{b}c}')

    def test_remove_boxed_success(self):
        """测试移除 boxed"""
        text = "\\boxed{42}"
        result = remove_boxed(text)
        self.assertEqual(result, '42')

    def test_remove_boxed_failure(self):
        """测试移除 boxed 失败"""
        text = "not boxed"
        result = remove_boxed(text)
        self.assertIsNone(result)

    def test_extract_boxed_answer_basic(self):
        """测试提取 boxed 答案"""
        text = "The answer is \\boxed{42}"
        result = extract_boxed_answer(text)
        self.assertEqual(result, '42')

    def test_extract_boxed_answer_with_double_brace(self):
        """测试提取带双括号的答案"""
        text = "\\boxed{{42}}"
        result = extract_boxed_answer(text, strip_double_curly_brace=True)
        self.assertEqual(result, '42')

    def test_extract_boxed_answer_no_box(self):
        """测试无 box 的情况"""
        text = "No answer"
        result = extract_boxed_answer(text)
        self.assertIsNone(result)


class TestNormalizeFinalAnswer(unittest.TestCase):
    """测试 normalize_final_answer"""

    def test_normalize_basic(self):
        """测试基本规范化"""
        result = normalize_final_answer("42")
        self.assertEqual(result, '42')

    def test_normalize_with_substitutions(self):
        """测试替换"""
        result = normalize_final_answer("an answer")
        self.assertNotIn('an ', result)

    def test_normalize_with_removed_expressions(self):
        """测试移除表达式"""
        result = normalize_final_answer("42 dollars")
        self.assertNotIn('dollars', result)

    def test_normalize_with_text(self):
        """测试 text 标签"""
        result = normalize_final_answer("\\text{answer}")
        self.assertNotIn('\\text', result)

    def test_normalize_with_boxed(self):
        """测试 boxed 标签"""
        result = normalize_final_answer("\\boxed{42}")
        self.assertEqual(result, '42')

    def test_normalize_final_answer_is(self):
        """测试 finalansweris 模式"""
        result = normalize_final_answer("finalansweris42")
        self.assertEqual(result, '42')

    def test_normalize_answer_is(self):
        """测试 answeris 模式"""
        result = normalize_final_answer("answeris:42")
        self.assertEqual(result, '42')

    def test_normalize_with_dollar(self):
        """测试美元符号"""
        result = normalize_final_answer("$42$")
        self.assertEqual(result, '42')

    def test_normalize_with_frac(self):
        """测试 frac"""
        result = normalize_final_answer("rac12")
        self.assertIn('\\frac', result)

    def test_normalize_with_comma_digits(self):
        """测试逗号分隔的数字"""
        result = normalize_final_answer("100,000")
        self.assertEqual(result, '100000')


class TestExtractAnswer(unittest.TestCase):
    """测试 extract_answer"""

    def test_extract_answer_found(self):
        """测试提取答案成功"""
        text = "ANSWER: 42"
        result = extract_answer(text)
        self.assertEqual(result, '42')

    def test_extract_answer_case_insensitive(self):
        """测试大小写不敏感"""
        text = "answer: 42"
        result = extract_answer(text)
        self.assertEqual(result, '42')

    def test_extract_answer_not_found(self):
        """测试未找到答案"""
        text = "No answer here"
        result = extract_answer(text)
        self.assertEqual(result, '')


class TestMATHDataset(unittest.TestCase):
    """测试 MATHDataset"""

    @patch.object(math_module, 'get_data_path', return_value='/fake/path')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_basic(self, mock_file, mock_path):
        """测试基本加载"""
        mock_data = {
            '1': {'problem': 'What is 1+1?', 'solution': '\\boxed{2}'},
            '2': {'problem': 'What is 2+2?', 'solution': '\\boxed{4}'}
        }
        mock_file.return_value.read.return_value = json.dumps(mock_data)
        mock_file.return_value.__enter__.return_value = mock_file.return_value
        
        with patch('json.load', return_value=mock_data):
            ds = MATHDataset.load('/input')
            self.assertIsInstance(ds, DatasetDict)
            self.assertIn('train', ds)
            self.assertIn('test', ds)
            self.assertEqual(len(ds['test']), 2)
            self.assertEqual(ds['test'][0]['problem'], 'What is 1+1?')
            self.assertEqual(ds['test'][0]['solution'], '2')


class TestPostprocessFunctions(unittest.TestCase):
    """测试后处理函数"""

    def test_math_postprocess_with_final_answer(self):
        """测试包含 final answer 的后处理"""
        text = "Let me think. The final answer is 42. Done."
        result = math_postprocess(text)
        self.assertIn('42', result)

    def test_math_postprocess_without_final_answer(self):
        """测试不包含 final answer 的后处理"""
        text = "The result is 42. More text."
        result = math_postprocess(text)
        # 应该返回第一个句子的规范化结果
        self.assertIsNotNone(result)

    def test_math_judement_preprocess(self):
        """测试判断预处理"""
        text = "ANSWER: 42"
        result = math_judement_preprocess(text)
        self.assertEqual(result, '42')

    def test_math_postprocess_v2_with_boxed(self):
        """测试 v2 后处理：带 boxed"""
        text = "The answer is \\boxed{42}"
        result = math_postprocess_v2(text)
        self.assertEqual(result, '42')

    def test_math_postprocess_v2_with_final_answer(self):
        """测试 v2 后处理：带 final answer"""
        text = "The final answer is 42. Done."
        result = math_postprocess_v2(text)
        self.assertIn('42', result)

    def test_math_postprocess_v2_with_answer_is(self):
        """测试 v2 后处理：带 answer is"""
        text = "The answer is 42. Done."
        result = math_postprocess_v2(text)
        self.assertIn('42', result)


class TestMATHEvaluator(unittest.TestCase):
    """测试 MATHEvaluator"""

    def test_init_v1(self):
        """测试 v1 初始化"""
        evaluator = MATHEvaluator(version='v1')
        self.assertEqual(evaluator.version, 'v1')

    def test_init_v2(self):
        """测试 v2 初始化"""
        evaluator = MATHEvaluator(version='v2')
        self.assertEqual(evaluator.version, 'v2')

    def test_init_invalid_version(self):
        """测试无效版本"""
        with self.assertRaises(AssertionError):
            MATHEvaluator(version='v3')

    def test_score_length_mismatch(self):
        """测试长度不匹配"""
        evaluator = MATHEvaluator()
        result = evaluator.score(['pred1'], ['ref1', 'ref2'])
        self.assertIn('error', result)

    def test_score_basic(self):
        """测试基本评分"""
        evaluator = MATHEvaluator()
        result = evaluator.score(['42', '43'], ['42', '44'])
        self.assertIn('accuracy', result)
        self.assertEqual(result['accuracy'], 50.0)
        self.assertIn('details', result)

    def test_is_equiv_both_none(self):
        """测试两个都是 None"""
        evaluator = MATHEvaluator()
        result = evaluator.is_equiv(None, None)
        self.assertTrue(result)

    def test_is_equiv_one_none(self):
        """测试一个是 None"""
        evaluator = MATHEvaluator()
        result = evaluator.is_equiv('42', None)
        self.assertFalse(result)

    def test_is_equiv_same_string(self):
        """测试相同字符串"""
        evaluator = MATHEvaluator()
        result = evaluator.is_equiv('42', '42')
        self.assertTrue(result)

    def test_fix_fracs_basic(self):
        """测试 fix_fracs"""
        evaluator = MATHEvaluator()
        result = evaluator._fix_fracs('\\frac12')
        self.assertIn('{1}{2}', result)

    def test_fix_fracs_already_fixed(self):
        """测试已经修复的 frac"""
        evaluator = MATHEvaluator()
        result = evaluator._fix_fracs('\\frac{1}{2}')
        self.assertEqual(result, '\\frac{1}{2}')

    def test_fix_a_slash_b_success(self):
        """测试 a/b 转换"""
        evaluator = MATHEvaluator()
        result = evaluator._fix_a_slash_b('1/2')
        self.assertEqual(result, '\\frac{1}{2}')

    def test_fix_a_slash_b_failure(self):
        """测试 a/b 转换失败"""
        evaluator = MATHEvaluator()
        result = evaluator._fix_a_slash_b('1/2/3')
        self.assertEqual(result, '1/2/3')

    def test_remove_right_units(self):
        """测试移除右侧单位"""
        evaluator = MATHEvaluator()
        result = evaluator._remove_right_units('42\\text{ meters}')
        self.assertEqual(result, '42')

    def test_remove_right_units_no_units(self):
        """测试无单位"""
        evaluator = MATHEvaluator()
        result = evaluator._remove_right_units('42')
        self.assertEqual(result, '42')

    def test_fix_sqrt_basic(self):
        """测试 fix_sqrt"""
        evaluator = MATHEvaluator()
        result = evaluator._fix_sqrt('\\sqrt2')
        self.assertEqual(result, '\\sqrt{2}')

    def test_fix_sqrt_already_fixed(self):
        """测试已经修复的 sqrt"""
        evaluator = MATHEvaluator()
        result = evaluator._fix_sqrt('\\sqrt{2}')
        self.assertEqual(result, '\\sqrt{2}')

    def test_fix_sqrt_v2(self):
        """测试 fix_sqrt_v2"""
        evaluator = MATHEvaluator()
        result = evaluator._fix_sqrt_v2('\\sqrtabc')
        self.assertIn('{abc}', result)

    def test_strip_string_basic(self):
        """测试 strip_string"""
        evaluator = MATHEvaluator()
        result = evaluator._strip_string('42')
        self.assertEqual(result, '42')

    def test_strip_string_with_newline(self):
        """测试带换行符"""
        evaluator = MATHEvaluator()
        result = evaluator._strip_string('42\n')
        self.assertNotIn('\n', result)

    def test_strip_string_with_dollar(self):
        """测试带美元符号"""
        evaluator = MATHEvaluator()
        result = evaluator._strip_string('\\$42')
        self.assertNotIn('$', result)

    def test_strip_string_with_spaces(self):
        """测试带空格"""
        evaluator = MATHEvaluator()
        result = evaluator._strip_string('4 2')
        self.assertEqual(result, '42')

    def test_strip_string_with_equals(self):
        """测试带等号"""
        evaluator = MATHEvaluator()
        result = evaluator._strip_string('x=42')
        self.assertEqual(result, '42')

    def test_strip_string_point_five(self):
        """测试 0.5"""
        evaluator = MATHEvaluator()
        result = evaluator._strip_string('0.5')
        self.assertEqual(result, '\\frac{1}{2}')

    def test_strip_string_v2_basic(self):
        """测试 strip_string_v2"""
        evaluator = MATHEvaluator(version='v2')
        result = evaluator._strip_string_v2('42')
        self.assertEqual(result, '42')

    def test_strip_string_v2_with_text(self):
        """测试 v2 带 text"""
        evaluator = MATHEvaluator(version='v2')
        result = evaluator._strip_string_v2('42\\text{meters}')
        self.assertEqual(result, '42')

    def test_strip_string_v2_with_infinity(self):
        """测试 v2 带 infinity"""
        evaluator = MATHEvaluator(version='v2')
        result = evaluator._strip_string_v2('infinity')
        self.assertEqual(result, '\\infty')

    def test_strip_string_v2_with_j(self):
        """测试 v2 带 j（虚数单位）"""
        evaluator = MATHEvaluator(version='v2')
        result = evaluator._strip_string_v2('2j')
        self.assertIn('i', result)

    def test_strip_string_v2_with_trailing_zeros(self):
        """测试 v2 带尾随零"""
        evaluator = MATHEvaluator(version='v2')
        result = evaluator._strip_string_v2('42.000')
        self.assertEqual(result, '42')


class TestMATHAgentEvaluator(unittest.TestCase):
    """测试 MATHAgentEvaluator"""

    def test_init(self):
        """测试初始化"""
        evaluator = MATHAgentEvaluator(action='TestAction')
        self.assertEqual(evaluator.action, 'TestAction')

    def test_get_action_found(self):
        """测试获取 action"""
        evaluator = MATHAgentEvaluator()
        steps = [
            {'type': 'Other'},
            {'type': 'PythonInterpreter', 'result': {'text': '42'}}
        ]
        result = evaluator.get_action(steps)
        self.assertEqual(result['type'], 'PythonInterpreter')

    def test_get_action_not_found(self):
        """测试未找到 action"""
        evaluator = MATHAgentEvaluator()
        steps = [{'type': 'Other'}]
        result = evaluator.get_action(steps)
        self.assertIsNone(result)

    def test_soft_equal_success(self):
        """测试 soft_equal 成功"""
        evaluator = MATHAgentEvaluator()
        step = {'result': {'text': '42'}}
        result = evaluator.soft_equal('pred', '42', step)
        self.assertTrue(result)

    def test_soft_equal_failure(self):
        """测试 soft_equal 失败"""
        evaluator = MATHAgentEvaluator()
        step = {'result': {'text': '42'}}
        result = evaluator.soft_equal('pred', '43', step)
        self.assertFalse(result)

    def test_score_length_mismatch(self):
        """测试长度不匹配"""
        evaluator = MATHAgentEvaluator()
        result = evaluator.score(['pred1'], ['ref1', 'ref2'], [])
        self.assertIn('error', result)

    def test_score_basic(self):
        """测试基本评分"""
        evaluator = MATHAgentEvaluator()
        predictions = ['42', '43']
        references = ['42', '44']
        steps = [
            [],
            [{'type': 'PythonInterpreter', 'result': {'text': '43'}, 'errmsg': None}]
        ]
        result = evaluator.score(predictions, references, steps)
        self.assertIn('follow_acc', result)
        self.assertIn('reasoning_acc', result)
        self.assertIn('code_acc', result)
        self.assertIn('action_pct', result)


if __name__ == '__main__':
    unittest.main()

