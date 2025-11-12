"""Unit tests for agieval/post_process.py"""
import unittest

from ais_bench.benchmark.datasets.agieval.post_process import (
    extract_last_line,
    remove_few_shot_prefix,
    try_parse_few_shot_qa_single_answer,
    try_parse_few_shot_pattern,
    parse_few_shot_qa_single_answer,
    find_first_capital_letter,
    extract_answer_in_bracket,
    parse_math_answer,
    parse_qa_multiple_answer,
    post_process,
)


class TestExtractLastLine(unittest.TestCase):
    """测试 extract_last_line"""

    def test_single_line(self):
        """测试单行"""
        result = extract_last_line("This is a line")
        self.assertEqual(result, "This is a line")

    def test_multiple_lines(self):
        """测试多行"""
        text = "Line 1\nLine 2\nLine 3"
        result = extract_last_line(text)
        self.assertEqual(result, "Line 3")

    def test_trailing_empty_lines(self):
        """测试尾随空行"""
        text = "Line 1\nLine 2\n\n\n"
        result = extract_last_line(text)
        self.assertEqual(result, "Line 2")

    def test_all_empty_lines(self):
        """测试全空行"""
        text = "\n\n\n"
        result = extract_last_line(text)
        self.assertEqual(result, "\n\n\n")


class TestRemoveFewShotPrefix(unittest.TestCase):
    """测试 remove_few_shot_prefix"""

    def test_english_prefix_at_start(self):
        """测试英文前缀在开头"""
        text = "The answer is therefore A"
        result = remove_few_shot_prefix(text)
        self.assertEqual(result, "A")

    def test_chinese_prefix_at_start(self):
        """测试中文前缀在开头"""
        text = "答案是 B"
        result = remove_few_shot_prefix(text)
        self.assertEqual(result, "B")

    def test_english_prefix_in_middle(self):
        """测试英文前缀在中间"""
        text = "Some text. The answer is therefore C"
        result = remove_few_shot_prefix(text)
        self.assertEqual(result, "C")

    def test_chinese_prefix_in_middle(self):
        """测试中文前缀在中间"""
        text = "一些文本。答案是 D"
        result = remove_few_shot_prefix(text)
        self.assertEqual(result, "D")

    def test_no_prefix(self):
        """测试无前缀"""
        text = "Just some text"
        result = remove_few_shot_prefix(text)
        self.assertEqual(result, "Just some text")


class TestTryParseFewShotQASingleAnswer(unittest.TestCase):
    """测试 try_parse_few_shot_qa_single_answer"""

    def test_english_answer(self):
        """测试英文答案"""
        text = "The answer is A"
        result = try_parse_few_shot_qa_single_answer(text, 'few-shot', 'en')
        self.assertEqual(result, 'A')

    def test_chinese_answer(self):
        """测试中文答案"""
        text = "答案是 B"
        result = try_parse_few_shot_qa_single_answer(text, 'few-shot', 'zh')
        self.assertEqual(result, 'B')

    def test_few_shot_cot_english(self):
        """测试 few-shot-CoT 英文"""
        text = "Step 1\nStep 2\nThe answer is C"
        result = try_parse_few_shot_qa_single_answer(text, 'few-shot-CoT', 'en')
        self.assertEqual(result, 'C')

    def test_few_shot_cot_chinese(self):
        """测试 few-shot-CoT 中文"""
        text = "步骤1\n步骤2\n答案是 D"
        result = try_parse_few_shot_qa_single_answer(text, 'few-shot-CoT', 'zh')
        self.assertEqual(result, 'D')

    def test_no_match(self):
        """测试无匹配"""
        text = "No answer here"
        result = try_parse_few_shot_qa_single_answer(text, 'few-shot', 'en')
        self.assertIsNone(result)

    def test_unknown_language(self):
        """测试未知语言"""
        with self.assertRaises(ValueError):
            try_parse_few_shot_qa_single_answer("text", 'few-shot', 'fr')


class TestTryParseFewShotPattern(unittest.TestCase):
    """测试 try_parse_few_shot_pattern"""

    def test_chinese_cloze(self):
        """测试中文完形填空"""
        text = "答案是 42"
        result = try_parse_few_shot_pattern(text, 'gaokao-mathcloze', 'few-shot')
        self.assertTrue(result)

    def test_english_cloze(self):
        """测试英文完形填空"""
        text = "The answer is therefore 42"
        result = try_parse_few_shot_pattern(text, 'math', 'few-shot')
        self.assertTrue(result)

    def test_chinese_qa(self):
        """测试中文 QA"""
        text = "答案是 A"
        result = try_parse_few_shot_pattern(text, 'logiqa-zh', 'few-shot')
        self.assertTrue(result)

    def test_english_qa(self):
        """测试英文 QA"""
        text = "The answer is B"
        result = try_parse_few_shot_pattern(text, 'lsat-ar', 'few-shot')
        self.assertTrue(result)

    def test_few_shot_cot(self):
        """测试 few-shot-CoT"""
        text = "Step 1\nStep 2\n答案是 C"
        result = try_parse_few_shot_pattern(text, 'logiqa-zh', 'few-shot-CoT')
        self.assertTrue(result)

    def test_no_match(self):
        """测试无匹配"""
        text = "No pattern here"
        result = try_parse_few_shot_pattern(text, 'lsat-ar', 'few-shot')
        self.assertFalse(result)

    def test_unknown_dataset(self):
        """测试未知数据集"""
        text = "Some text"
        result = try_parse_few_shot_pattern(text, 'unknown-dataset', 'few-shot')
        self.assertFalse(result)


class TestParseFewShotQASingleAnswer(unittest.TestCase):
    """测试 parse_few_shot_qa_single_answer"""

    def test_with_pattern(self):
        """测试有模式"""
        text = "The answer is A"
        result = parse_few_shot_qa_single_answer(text, 'few-shot', 'en')
        self.assertEqual(result, 'A')

    def test_without_pattern(self):
        """测试无模式（回退到查找首字母）"""
        text = "The result is B but not A"
        result = parse_few_shot_qa_single_answer(text, 'few-shot', 'en')
        self.assertEqual(result, 'B')


class TestFindFirstCapitalLetter(unittest.TestCase):
    """测试 find_first_capital_letter"""

    def test_find_a(self):
        """测试查找 A"""
        result = find_first_capital_letter("The answer is A")
        self.assertEqual(result, 'A')

    def test_find_b(self):
        """测试查找 B"""
        result = find_first_capital_letter("Option B is correct")
        self.assertEqual(result, 'B')

    def test_multiple_letters(self):
        """测试多个字母（返回第一个）"""
        result = find_first_capital_letter("A or B or C")
        self.assertEqual(result, 'A')

    def test_no_letter(self):
        """测试无字母"""
        result = find_first_capital_letter("No valid option")
        self.assertEqual(result, '')

    def test_with_g(self):
        """测试不在集合中的字母"""
        result = find_first_capital_letter("G is not valid, A is")
        self.assertEqual(result, 'A')


class TestExtractAnswerInBracket(unittest.TestCase):
    """测试 extract_answer_in_bracket"""

    def test_default_brackets(self):
        """测试默认括号"""
        text = "The answer is 【42】"
        result = extract_answer_in_bracket(text)
        self.assertEqual(result, '42')

    def test_custom_brackets(self):
        """测试自定义括号"""
        text = "The answer is [42]"
        result = extract_answer_in_bracket(text, prefix='[', suffix=']')
        self.assertEqual(result, '42')

    def test_no_prefix(self):
        """测试无前缀"""
        text = "The answer is 42】"
        # 当只有suffix没有prefix时，函数会抛出ValueError
        with self.assertRaises(ValueError):
            extract_answer_in_bracket(text)

    def test_no_suffix(self):
        """测试无后缀"""
        text = "The answer is 【42"
        # 当只有prefix没有suffix时，函数会抛出ValueError
        with self.assertRaises(ValueError):
            extract_answer_in_bracket(text)

    def test_empty_bracket(self):
        """测试空括号"""
        text = "【】"
        result = extract_answer_in_bracket(text)
        self.assertEqual(result, '')


class TestParseMathAnswer(unittest.TestCase):
    """测试 parse_math_answer"""

    def test_few_shot(self):
        """测试 few-shot"""
        text = "The answer is therefore 42"
        result = parse_math_answer('few-shot', text)
        self.assertEqual(result, '42')

    def test_few_shot_cot(self):
        """测试 few-shot-CoT"""
        text = "Step 1\nStep 2\nThe answer is therefore 42"
        result = parse_math_answer('few-shot-CoT', text)
        self.assertEqual(result, '42')

    def test_with_boxed(self):
        """测试带 boxed"""
        text = "The answer is \\boxed{42}"
        result = parse_math_answer('zero-shot', text)
        self.assertEqual(result, '42')

    def test_with_boxed_and_equals(self):
        """测试带 boxed 和等号"""
        text = "\\boxed{x = 42}"
        result = parse_math_answer('zero-shot', text)
        self.assertEqual(result, '42')

    def test_with_fbox(self):
        """测试带 fbox"""
        text = "The answer is \\fbox{42}"
        result = parse_math_answer('zero-shot', text)
        self.assertEqual(result, '42')

    def test_with_dollar_sign(self):
        """测试带美元符号"""
        text = "The answer is $42$"
        result = parse_math_answer('zero-shot', text)
        self.assertEqual(result, '42')

    def test_with_dollar_sign_and_equals(self):
        """测试带美元符号和等号"""
        text = "The result is $x = 42$"
        result = parse_math_answer('zero-shot', text)
        self.assertEqual(result, '42')

    def test_with_equals_no_dollar(self):
        """测试带等号无美元符号"""
        text = "x = 42"
        result = parse_math_answer('zero-shot', text)
        self.assertEqual(result, '42')

    def test_with_equals_and_newline(self):
        """测试带等号和换行"""
        text = "x = 42\\nMore text"
        result = parse_math_answer('zero-shot', text)
        self.assertEqual(result, '42')

    def test_number_pattern(self):
        """测试数字模式"""
        text = "The result is 42.5"
        result = parse_math_answer('zero-shot', text)
        self.assertEqual(result, '42.5')

    def test_invalid_boxed(self):
        """测试无效的 boxed"""
        text = "\\boxed{unclosed"
        result = parse_math_answer('zero-shot', text)
        self.assertIsNone(result)

    def test_chinese_prefix(self):
        """测试中文前缀"""
        text = "答案是 42"
        result = parse_math_answer('few-shot', text)
        self.assertEqual(result, '42')


class TestParseQAMultipleAnswer(unittest.TestCase):
    """测试 parse_qa_multiple_answer"""

    def test_single_answer(self):
        """测试单个答案"""
        text = "answer is (A)"
        result = parse_qa_multiple_answer(text, 'few-shot')
        self.assertEqual(result, ['A'])

    def test_multiple_answers(self):
        """测试多个答案"""
        text = "answers are (A), (B), and (C)"
        result = parse_qa_multiple_answer(text, 'few-shot')
        self.assertEqual(result, ['A', 'B', 'C'])

    def test_with_parentheses(self):
        """测试带括号"""
        text = "选择 (A), (B), (C)"
        result = parse_qa_multiple_answer(text, 'few-shot')
        self.assertEqual(result, ['A', 'B', 'C'])

    def test_few_shot_cot(self):
        """测试 few-shot-CoT"""
        text = "step 1\nstep 2\nanswers are (A) and (B)"
        result = parse_qa_multiple_answer(text, 'few-shot-CoT')
        self.assertEqual(result, ['A', 'B'])

    def test_no_answer(self):
        """测试无答案"""
        text = "no valid answer here"
        result = parse_qa_multiple_answer(text, 'few-shot')
        self.assertEqual(result, [])


class TestPostProcess(unittest.TestCase):
    """测试 post_process"""

    def test_english_cloze(self):
        """测试英文完形填空"""
        result = post_process('math', 'zero-shot', 'The answer is $42$')
        self.assertEqual(result, '42')

    def test_chinese_cloze(self):
        """测试中文完形填空"""
        result = post_process('gaokao-mathcloze', 'few-shot', '答案是 42')
        self.assertEqual(result, '42')

    def test_multiple_answer_dataset(self):
        """测试多答案数据集"""
        result = post_process('jec-qa-kd', 'few-shot', 'answers are (A) and (B)')
        self.assertEqual(result, ['A', 'B'])

    def test_zero_shot_qa(self):
        """测试 zero-shot QA"""
        result = post_process('lsat-ar', 'zero-shot', 'answer is A')
        self.assertEqual(result, 'A')

    def test_zero_shot_cot_qa(self):
        """测试 zero-shot-CoT QA"""
        result = post_process('lsat-ar', 'zero-shot-CoT', 'step by step. answer: B')
        self.assertEqual(result, 'B')

    def test_few_shot_english_qa(self):
        """测试 few-shot 英文 QA"""
        result = post_process('lsat-ar', 'few-shot', 'The answer is C')
        self.assertEqual(result, 'C')

    def test_few_shot_chinese_qa(self):
        """测试 few-shot 中文 QA"""
        result = post_process('logiqa-zh', 'few-shot', '答案是 D')
        self.assertEqual(result, 'D')

    def test_few_shot_cot_english_qa(self):
        """测试 few-shot-CoT 英文 QA"""
        result = post_process('lsat-ar', 'few-shot-CoT', 'Step 1\nStep 2\nThe answer is E')
        self.assertEqual(result, 'E')

    def test_few_shot_cot_chinese_qa(self):
        """测试 few-shot-CoT 中文 QA"""
        result = post_process('logiqa-zh', 'few-shot-CoT', '步骤1\n步骤2\n答案是 F')
        self.assertEqual(result, 'F')

    def test_unsupported_dataset(self):
        """测试不支持的数据集"""
        with self.assertRaises(ValueError):
            post_process('unknown-dataset', 'few-shot', 'Some text')


if __name__ == '__main__':
    unittest.main()

