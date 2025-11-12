import unittest
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.livecodebench.prompts import (
        CodeGenerationPromptConstants,
        TestOutputPromptConstants,
        SelfRepairPromptConstants,
        make_code_execution_prompt,
        get_generic_question_template_test_completion,
        get_generic_question_template_answer_self_repair
    )
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False


class PromptsTestBase(unittest.TestCase):
    """Prompts测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not PROMPTS_AVAILABLE:
            cls.skipTest(cls, "Prompts modules not available")


class TestCodeGenerationPromptConstants(PromptsTestBase):
    """测试CodeGenerationPromptConstants类"""
    
    def test_constants_exist(self):
        """测试常量存在"""
        self.assertIsNotNone(CodeGenerationPromptConstants.SYSTEM_MESSAGE_GENERIC)
        self.assertIsNotNone(CodeGenerationPromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE)
        self.assertIsNotNone(CodeGenerationPromptConstants.FORMATTING_WITHOUT_STARTER_CODE)
        self.assertIsInstance(CodeGenerationPromptConstants.SYSTEM_MESSAGE_GENERIC, str)


class TestTestOutputPromptConstants(PromptsTestBase):
    """测试TestOutputPromptConstants类"""
    
    def test_constants_exist(self):
        """测试常量存在"""
        self.assertIsNotNone(TestOutputPromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC)
        self.assertIsNotNone(TestOutputPromptConstants.FORMATTING_MESSAGE)
        self.assertIsInstance(TestOutputPromptConstants.SYSTEM_MESSAGE_CHAT_GENERIC, str)


class TestSelfRepairPromptConstants(PromptsTestBase):
    """测试SelfRepairPromptConstants类"""
    
    def test_constants_exist(self):
        """测试常量存在"""
        self.assertIsNotNone(SelfRepairPromptConstants.SYSTEM_MESSAGE_GENERIC)
        self.assertIsNotNone(SelfRepairPromptConstants.FORMATTING_REPEAT)
        self.assertIsInstance(SelfRepairPromptConstants.SYSTEM_MESSAGE_GENERIC, str)


class TestMakeCodeExecutionPrompt(PromptsTestBase):
    """测试make_code_execution_prompt函数"""
    
    def test_make_prompt_with_cot(self):
        """测试生成带COT的提示"""
        code = 'def test(): return 1'
        input_str = 'test() == ??'
        result = make_code_execution_prompt(code, input_str, cot=True)
        
        self.assertIn('[PYTHON]', result)
        self.assertIn('[THOUGHT]', result)
        self.assertIn(code, result)
        self.assertIn(input_str, result)
    
    def test_make_prompt_without_cot(self):
        """测试生成不带COT的提示"""
        code = 'def test(): return 1'
        input_str = 'test() == ??'
        result = make_code_execution_prompt(code, input_str, cot=False)
        
        self.assertIn('[PYTHON]', result)
        self.assertIn('[ANSWER]', result)
        self.assertIn(code, result)
        self.assertIn(input_str, result)


class TestGetGenericQuestionTemplateTestCompletion(PromptsTestBase):
    """测试get_generic_question_template_test_completion函数"""
    
    def test_get_template(self):
        """测试获取测试完成模板"""
        question_content = 'Test question'
        starter_code = 'def test_func(x):\n    return x'
        testcase_input = 'test_func(1)'
        
        result = get_generic_question_template_test_completion(
            question_content, starter_code, testcase_input
        )
        
        self.assertIn('Problem:', result)
        self.assertIn(question_content, result)
        self.assertIn('Function:', result)
        self.assertIn('test_func', result)
        self.assertIn('assert', result)


class TestGetGenericQuestionTemplateAnswerSelfRepair(PromptsTestBase):
    """测试get_generic_question_template_answer_self_repair函数"""
    
    def test_get_template_with_compilation_error(self):
        """测试获取编译错误的自我修复模板"""
        question = 'Test question'
        code = 'def test(): return 1'
        metadata = json.dumps({
            'error_code': -1,
            'error': 'SyntaxError: invalid syntax'
        })
        
        result = get_generic_question_template_answer_self_repair(
            question, code, metadata
        )
        
        self.assertIn('Question:', result)
        self.assertIn('Answer:', result)
        self.assertIn(code, result)
        self.assertIn('compilation error', result)
    
    def test_get_template_with_wrong_answer(self):
        """测试获取错误答案的自我修复模板"""
        question = 'Test question'
        code = 'def test(): return 1'
        metadata = json.dumps({
            'error_code': -2,
            'inputs': 'test_input',
            'output': 'wrong_output',
            'expected': 'correct_output'
        })
        
        result = get_generic_question_template_answer_self_repair(
            question, code, metadata
        )
        
        self.assertIn('wrong answer', result)
        self.assertIn('wrong_output', result)
        self.assertIn('correct_output', result)
    
    def test_get_template_with_timeout(self):
        """测试获取超时的自我修复模板"""
        question = 'Test question'
        code = 'def test(): return 1'
        metadata = json.dumps({
            'error_code': -3,
            'error': 'TimeoutException',
            'inputs': 'test_input',
            'expected': 'correct_output'
        })
        
        result = get_generic_question_template_answer_self_repair(
            question, code, metadata
        )
        
        self.assertIn('time limit exceeded', result)
    
    def test_get_template_with_runtime_error(self):
        """测试获取运行时错误的自我修复模板"""
        question = 'Test question'
        code = 'def test(): return 1'
        metadata = json.dumps({
            'error_code': -4,
            'error': 'RuntimeError: division by zero',
            'inputs': 'test_input',
            'expected': 'correct_output'
        })
        
        result = get_generic_question_template_answer_self_repair(
            question, code, metadata
        )
        
        self.assertIn('runtime error', result)
    
    def test_get_template_no_error(self):
        """测试获取无错误的自我修复模板"""
        question = 'Test question'
        code = 'def test(): return 1'
        metadata = json.dumps({})
        
        result = get_generic_question_template_answer_self_repair(
            question, code, metadata
        )
        
        self.assertIn('Question:', result)
        self.assertIn('Answer:', result)


if __name__ == '__main__':
    unittest.main()

