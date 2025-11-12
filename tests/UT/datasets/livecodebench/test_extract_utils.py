import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.livecodebench.extract_utils import (
        extract_code_generation,
        extract_code_generation_v2,
        extract_code_execution,
        extract_test_output_code
    )
    EXTRACT_UTILS_AVAILABLE = True
except ImportError:
    EXTRACT_UTILS_AVAILABLE = False


class ExtractUtilsTestBase(unittest.TestCase):
    """ExtractUtils测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not EXTRACT_UTILS_AVAILABLE:
            cls.skipTest(cls, "ExtractUtils modules not available")


class TestExtractCodeGeneration(ExtractUtilsTestBase):
    """测试extract_code_generation函数"""
    
    def test_extract_from_chat_model(self):
        """测试从聊天模型输出中提取代码"""
        output = '''Some text before
```python
def test():
    return 1
```
Some text after'''
        result = extract_code_generation(output, model_type='chat')
        self.assertIn('def test()', result)
        self.assertIn('return 1', result)
    
    def test_extract_from_base_model(self):
        """测试从基础模型输出中提取代码"""
        output = 'def test():\n    return 1'
        result = extract_code_generation(output, model_type='base')
        self.assertEqual(result.strip(), output.strip())
    
    def test_extract_no_code_block(self):
        """测试提取无代码块的情况"""
        output = 'Some text without code blocks'
        result = extract_code_generation(output, model_type='chat')
        self.assertEqual(result, '')
    
    def test_extract_invalid_model_type(self):
        """测试无效的模型类型"""
        output = 'test output'
        with self.assertRaises(ValueError):
            extract_code_generation(output, model_type='invalid')


class TestExtractCodeGenerationV2(ExtractUtilsTestBase):
    """测试extract_code_generation_v2函数"""
    
    def test_extract_from_chat_model_single_block(self):
        """测试从聊天模型输出中提取单个代码块"""
        output = '''Some text
```python
def test():
    return 1
```
After'''
        result = extract_code_generation_v2(output, model_type='chat')
        self.assertIn('def test()', result)
    
    def test_extract_from_chat_model_multiple_blocks(self):
        """测试从聊天模型输出中提取多个代码块（取最后一个）"""
        output = '''First block:
```python
def old():
    pass
```
Second block:
```python
def new():
    return 2
```
After'''
        result = extract_code_generation_v2(output, model_type='chat')
        self.assertIn('def new()', result)
        self.assertNotIn('def old()', result)
    
    def test_extract_from_base_model(self):
        """测试从基础模型输出中提取代码"""
        output = 'def test():\n    return 1'
        result = extract_code_generation_v2(output, model_type='base')
        self.assertEqual(result.strip(), output.strip())


class TestExtractCodeExecution(ExtractUtilsTestBase):
    """测试extract_code_execution函数"""
    
    def test_extract_with_cot(self):
        """测试提取带COT的代码执行"""
        # 测试有COT和[ANSWER]标签的情况
        output = '''[ANSWER]
assert test() == 1
[/ANSWER]'''
        result = extract_code_execution(output, cot=True)
        # 如果有[ANSWER]且有COT，会提取[ANSWER]后的内容，然后提取==后的部分，得到"1"
        # 然后如果有[/ANSWER]，会去掉后面的部分
        self.assertIn('1', result.strip())
    
    def test_extract_without_cot(self):
        """测试提取不带COT的代码执行"""
        # 测试没有COT的情况，应该直接提取==后面的部分
        output = '''assert test() == 1'''
        result = extract_code_execution(output, cot=False)
        # 没有[PYTHON]标签匹配，没有[ANSWER]，但有==，会提取==后面的部分，得到"1"
        self.assertIn('1', result.strip())
    
    def test_extract_with_equals(self):
        """测试提取包含等号的输出"""
        output = '''assert test() == 2'''
        result = extract_code_execution(output, cot=False)
        self.assertIn('2', result)
    
    def test_extract_single_line(self):
        """测试提取单行输出"""
        output = 'test output line'
        result = extract_code_execution(output, cot=False)
        self.assertEqual(result.strip(), 'test output line')


class TestExtractTestOutputCode(ExtractUtilsTestBase):
    """测试extract_test_output_code函数"""
    
    def test_extract_from_assert_statement(self):
        """测试从assert语句中提取"""
        output = '''Some text
assert test_func(1) == 2
More text'''
        result = extract_test_output_code(output)
        self.assertEqual(result, 'assert test_func(1) == 2')
    
    def test_extract_from_python_block(self):
        """测试从Python代码块中提取"""
        output = '''Some text
```python
assert test_func(1) == 2
```
After'''
        result = extract_test_output_code(output)
        self.assertIn('assert', result)
    
    def test_extract_from_generic_block(self):
        """测试从通用代码块中提取"""
        output = '''Some text
```
assert test_func(1) == 2
```
After'''
        result = extract_test_output_code(output)
        self.assertIn('assert', result)
    
    def test_extract_no_match(self):
        """测试无匹配的情况"""
        output = 'Just some text without assert or code blocks'
        result = extract_test_output_code(output)
        self.assertEqual(result, '')


if __name__ == '__main__':
    unittest.main()

