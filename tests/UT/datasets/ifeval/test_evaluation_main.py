"""Unit tests for ifeval evaluation_main.py to increase coverage to 80%"""
import unittest
import sys
from unittest.mock import patch, MagicMock

# Mock absl.flags 以避免在导入时阻塞
# absl.flags 在模块级别定义了 required=True 的标志，会导致导入时阻塞
# 方法：在导入前 mock DEFINE_string，使其不检查必需参数
_original_argv = sys.argv[:]
sys.argv = ['test']  # 设置一个简单的命令行参数

# 创建一个 mock 标志对象
_mock_flag_obj = MagicMock()
_mock_flag_obj.value = '/tmp/test'

# Mock DEFINE_string 函数，使其返回 mock 对象而不检查必需参数
_original_define_string = None
try:
    import absl.flags as absl_flags_module
    _original_define_string = absl_flags_module.DEFINE_string
    
    def _mock_define_string(name, default=None, help=None, required=False, **kwargs):
        """Mock DEFINE_string，忽略 required 参数"""
        return _mock_flag_obj
    
    absl_flags_module.DEFINE_string = _mock_define_string
except ImportError:
    pass

# Mock nltk.download 以避免在导入时阻塞
# instructions_util.py 在模块级别调用了 nltk.download('punkt_tab')，会导致阻塞
try:
    import nltk
    _original_download = nltk.download
    nltk.download = MagicMock(return_value=True)
except ImportError:
    pass

# 尝试导入评估模块
try:
    from ais_bench.benchmark.datasets.ifeval.evaluation_main import (
        InputExample,
        OutputExample,
    )
    from ais_bench.benchmark.datasets.ifeval.evaluation_main import test_instruction_following_strict as _test_instruction_following_strict_func
    from ais_bench.benchmark.datasets.ifeval.evaluation_main import test_instruction_following_loose as _test_instruction_following_loose_func
    EVALUATION_AVAILABLE = True
except (ImportError, SystemExit, Exception):
    EVALUATION_AVAILABLE = False
finally:
    # 恢复原始函数
    if _original_define_string is not None:
        try:
            import absl.flags as absl_flags_module
            absl_flags_module.DEFINE_string = _original_define_string
        except:
            pass
    sys.argv = _original_argv


class EvaluationTestBase(unittest.TestCase):
    """基础测试类"""
    @classmethod
    def setUpClass(cls):
        if not EVALUATION_AVAILABLE:
            cls.skipTest(cls, "Evaluation modules not available")


class TestInputExample(EvaluationTestBase):
    """测试 InputExample 数据类"""
    
    def test_input_example_creation(self):
        """测试 InputExample 创建"""
        example = InputExample(
            key=1,
            instruction_id_list=['test_instruction'],
            prompt='Test prompt',
            kwargs=[{'param': 'value'}]
        )
        self.assertEqual(example.key, 1)
        self.assertEqual(example.instruction_id_list, ['test_instruction'])
        self.assertEqual(example.prompt, 'Test prompt')
        self.assertEqual(example.kwargs, [{'param': 'value'}])


class TestOutputExample(EvaluationTestBase):
    """测试 OutputExample 数据类"""
    
    def test_output_example_creation(self):
        """测试 OutputExample 创建"""
        example = OutputExample(
            instruction_id_list=['test_instruction'],
            prompt='Test prompt',
            response='Test response',
            follow_all_instructions=True,
            follow_instruction_list=[True]
        )
        self.assertEqual(example.instruction_id_list, ['test_instruction'])
        self.assertEqual(example.prompt, 'Test prompt')
        self.assertEqual(example.response, 'Test response')
        self.assertTrue(example.follow_all_instructions)
        self.assertEqual(example.follow_instruction_list, [True])


class TestInstructionFollowingStrict(EvaluationTestBase):
    """测试 test_instruction_following_strict 函数"""
    
    @patch('ais_bench.benchmark.datasets.ifeval.evaluation_main.instructions_registry.INSTRUCTION_DICT')
    def test_strict_all_instructions_followed(self, mock_instruction_dict):
        """测试所有指令都遵循的情况"""
        # 创建 mock 指令
        mock_instruction = MagicMock()
        mock_instruction.build_description = MagicMock()
        mock_instruction.get_instruction_args = MagicMock(return_value={})
        mock_instruction.check_following = MagicMock(return_value=True)
        
        mock_instruction_cls = MagicMock(return_value=mock_instruction)
        mock_instruction_dict.__getitem__ = MagicMock(return_value=mock_instruction_cls)
        
        # 创建输入
        inp = InputExample(
            key=1,
            instruction_id_list=['test_instruction'],
            prompt='Test prompt',
            kwargs=[{'param': 'value'}]
        )
        response = 'Test response'
        
        # 调用函数
        result = _test_instruction_following_strict_func(inp, response)
        
        # 验证结果
        self.assertIsInstance(result, OutputExample)
        self.assertTrue(result.follow_all_instructions)
        self.assertEqual(result.follow_instruction_list, [True])
        self.assertEqual(result.response, response)
        self.assertEqual(result.prompt, inp.prompt)
    
    @patch('ais_bench.benchmark.datasets.ifeval.evaluation_main.instructions_registry.INSTRUCTION_DICT')
    def test_strict_instruction_not_followed(self, mock_instruction_dict):
        """测试指令未遵循的情况"""
        # 创建 mock 指令
        mock_instruction = MagicMock()
        mock_instruction.build_description = MagicMock()
        mock_instruction.get_instruction_args = MagicMock(return_value={})
        mock_instruction.check_following = MagicMock(return_value=False)
        
        mock_instruction_cls = MagicMock(return_value=mock_instruction)
        mock_instruction_dict.__getitem__ = MagicMock(return_value=mock_instruction_cls)
        
        # 创建输入
        inp = InputExample(
            key=1,
            instruction_id_list=['test_instruction'],
            prompt='Test prompt',
            kwargs=[{'param': 'value'}]
        )
        response = 'Test response'
        
        # 调用函数
        result = _test_instruction_following_strict_func(inp, response)
        
        # 验证结果
        self.assertFalse(result.follow_all_instructions)
        self.assertEqual(result.follow_instruction_list, [False])
    
    @patch('ais_bench.benchmark.datasets.ifeval.evaluation_main.instructions_registry.INSTRUCTION_DICT')
    def test_strict_empty_response(self, mock_instruction_dict):
        """测试空响应的情况"""
        # 创建 mock 指令
        mock_instruction = MagicMock()
        mock_instruction.build_description = MagicMock()
        mock_instruction.get_instruction_args = MagicMock(return_value={})
        mock_instruction.check_following = MagicMock(return_value=True)
        
        mock_instruction_cls = MagicMock(return_value=mock_instruction)
        mock_instruction_dict.__getitem__ = MagicMock(return_value=mock_instruction_cls)
        
        # 创建输入
        inp = InputExample(
            key=1,
            instruction_id_list=['test_instruction'],
            prompt='Test prompt',
            kwargs=[{'param': 'value'}]
        )
        response = '   '  # 空白响应
        
        # 调用函数
        result = _test_instruction_following_strict_func(inp, response)
        
        # 空响应应该被视为未遵循指令
        self.assertFalse(result.follow_all_instructions)
        self.assertEqual(result.follow_instruction_list, [False])
    
    @patch('ais_bench.benchmark.datasets.ifeval.evaluation_main.instructions_registry.INSTRUCTION_DICT')
    def test_strict_with_prompt_in_args(self, mock_instruction_dict):
        """测试参数中包含 prompt 的情况"""
        # 创建 mock 指令
        mock_instruction = MagicMock()
        mock_instruction.build_description = MagicMock()
        mock_instruction.get_instruction_args = MagicMock(return_value={'prompt': 'old_prompt'})
        mock_instruction.check_following = MagicMock(return_value=True)
        
        mock_instruction_cls = MagicMock(return_value=mock_instruction)
        mock_instruction_dict.__getitem__ = MagicMock(return_value=mock_instruction_cls)
        
        # 创建输入
        inp = InputExample(
            key=1,
            instruction_id_list=['test_instruction'],
            prompt='New prompt',
            kwargs=[{'param': 'value'}]
        )
        response = 'Test response'
        
        # 调用函数
        result = _test_instruction_following_strict_func(inp, response)
        
        # 验证 build_description 被调用两次（一次用 kwargs，一次用 prompt）
        self.assertEqual(mock_instruction.build_description.call_count, 2)
        # 第二次调用应该传入 prompt
        mock_instruction.build_description.assert_called_with(prompt=inp.prompt)
        self.assertTrue(result.follow_all_instructions)
    
    @patch('ais_bench.benchmark.datasets.ifeval.evaluation_main.instructions_registry.INSTRUCTION_DICT')
    def test_strict_multiple_instructions(self, mock_instruction_dict):
        """测试多个指令的情况"""
        # 创建 mock 指令
        mock_instruction1 = MagicMock()
        mock_instruction1.build_description = MagicMock()
        mock_instruction1.get_instruction_args = MagicMock(return_value={})
        mock_instruction1.check_following = MagicMock(return_value=True)
        
        mock_instruction2 = MagicMock()
        mock_instruction2.build_description = MagicMock()
        mock_instruction2.get_instruction_args = MagicMock(return_value={})
        mock_instruction2.check_following = MagicMock(return_value=False)
        
        mock_instruction_cls1 = MagicMock(return_value=mock_instruction1)
        mock_instruction_cls2 = MagicMock(return_value=mock_instruction2)
        
        # 根据不同的 instruction_id 返回不同的类
        def get_instruction_cls(instruction_id):
            if instruction_id == 'instruction1':
                return mock_instruction_cls1
            return mock_instruction_cls2
        
        mock_instruction_dict.__getitem__ = MagicMock(side_effect=get_instruction_cls)
        
        # 创建输入
        inp = InputExample(
            key=1,
            instruction_id_list=['instruction1', 'instruction2'],
            prompt='Test prompt',
            kwargs=[{'param1': 'value1'}, {'param2': 'value2'}]
        )
        response = 'Test response'
        
        # 调用函数
        result = _test_instruction_following_strict_func(inp, response)
        
        # 一个指令遵循，一个不遵循
        self.assertFalse(result.follow_all_instructions)
        self.assertEqual(result.follow_instruction_list, [True, False])


class TestInstructionFollowingLoose(EvaluationTestBase):
    """测试 test_instruction_following_loose 函数"""
    
    @patch('ais_bench.benchmark.datasets.ifeval.evaluation_main.instructions_registry.INSTRUCTION_DICT')
    def test_loose_all_instructions_followed(self, mock_instruction_dict):
        """测试宽松模式下所有指令都遵循"""
        # 创建 mock 指令
        mock_instruction = MagicMock()
        mock_instruction.build_description = MagicMock()
        mock_instruction.get_instruction_args = MagicMock(return_value={})
        mock_instruction.check_following = MagicMock(return_value=True)
        
        mock_instruction_cls = MagicMock(return_value=mock_instruction)
        mock_instruction_dict.__getitem__ = MagicMock(return_value=mock_instruction_cls)
        
        # 创建输入
        inp = InputExample(
            key=1,
            instruction_id_list=['test_instruction'],
            prompt='Test prompt',
            kwargs=[{'param': 'value'}]
        )
        response = 'Test response'
        
        # 调用函数
        result = _test_instruction_following_loose_func(inp, response)
        
        # 验证结果
        self.assertTrue(result.follow_all_instructions)
        self.assertEqual(result.follow_instruction_list, [True])
    
    @patch('ais_bench.benchmark.datasets.ifeval.evaluation_main.instructions_registry.INSTRUCTION_DICT')
    def test_loose_with_asterisks(self, mock_instruction_dict):
        """测试宽松模式处理星号"""
        # 创建 mock 指令，只有移除星号后才返回 True
        mock_instruction = MagicMock()
        mock_instruction.build_description = MagicMock()
        mock_instruction.get_instruction_args = MagicMock(return_value={})
        
        def check_following_side_effect(text):
            # 只有不包含星号的文本才返回 True
            return '*' not in text
        
        mock_instruction.check_following = MagicMock(side_effect=check_following_side_effect)
        
        mock_instruction_cls = MagicMock(return_value=mock_instruction)
        mock_instruction_dict.__getitem__ = MagicMock(return_value=mock_instruction_cls)
        
        # 创建输入
        inp = InputExample(
            key=1,
            instruction_id_list=['test_instruction'],
            prompt='Test prompt',
            kwargs=[{'param': 'value'}]
        )
        response = '*Test* response'
        
        # 调用函数
        result = _test_instruction_following_loose_func(inp, response)
        
        # 宽松模式会尝试移除星号的版本，应该能通过
        self.assertTrue(result.follow_all_instructions)
    
    @patch('ais_bench.benchmark.datasets.ifeval.evaluation_main.instructions_registry.INSTRUCTION_DICT')
    def test_loose_remove_first_line(self, mock_instruction_dict):
        """测试宽松模式移除第一行"""
        # 创建 mock 指令，只有移除第一行后才返回 True
        mock_instruction = MagicMock()
        mock_instruction.build_description = MagicMock()
        mock_instruction.get_instruction_args = MagicMock(return_value={})
        
        def check_following_side_effect(text):
            # 只有不以 "First" 开头的文本才返回 True
            return not text.strip().startswith('First')
        
        mock_instruction.check_following = MagicMock(side_effect=check_following_side_effect)
        
        mock_instruction_cls = MagicMock(return_value=mock_instruction)
        mock_instruction_dict.__getitem__ = MagicMock(return_value=mock_instruction_cls)
        
        # 创建输入
        inp = InputExample(
            key=1,
            instruction_id_list=['test_instruction'],
            prompt='Test prompt',
            kwargs=[{'param': 'value'}]
        )
        response = 'First line\nSecond line'
        
        # 调用函数
        result = _test_instruction_following_loose_func(inp, response)
        
        # 宽松模式会尝试移除第一行的版本，应该能通过
        self.assertTrue(result.follow_all_instructions)
    
    @patch('ais_bench.benchmark.datasets.ifeval.evaluation_main.instructions_registry.INSTRUCTION_DICT')
    def test_loose_with_prompt_in_args(self, mock_instruction_dict):
        """测试宽松模式下参数中包含 prompt"""
        # 创建 mock 指令
        mock_instruction = MagicMock()
        mock_instruction.build_description = MagicMock()
        mock_instruction.get_instruction_args = MagicMock(return_value={'prompt': 'old_prompt'})
        mock_instruction.check_following = MagicMock(return_value=True)
        
        mock_instruction_cls = MagicMock(return_value=mock_instruction)
        mock_instruction_dict.__getitem__ = MagicMock(return_value=mock_instruction_cls)
        
        # 创建输入
        inp = InputExample(
            key=1,
            instruction_id_list=['test_instruction'],
            prompt='New prompt',
            kwargs=[{'param': 'value'}]
        )
        response = 'Test response'
        
        # 调用函数
        result = _test_instruction_following_loose_func(inp, response)
        
        # 验证 build_description 被调用两次
        self.assertEqual(mock_instruction.build_description.call_count, 2)
        self.assertTrue(result.follow_all_instructions)
    
    @patch('ais_bench.benchmark.datasets.ifeval.evaluation_main.instructions_registry.INSTRUCTION_DICT')
    def test_loose_all_variations_fail(self, mock_instruction_dict):
        """测试宽松模式下所有变体都失败"""
        # 创建 mock 指令，所有情况都返回 False
        mock_instruction = MagicMock()
        mock_instruction.build_description = MagicMock()
        mock_instruction.get_instruction_args = MagicMock(return_value={})
        mock_instruction.check_following = MagicMock(return_value=False)
        
        mock_instruction_cls = MagicMock(return_value=mock_instruction)
        mock_instruction_dict.__getitem__ = MagicMock(return_value=mock_instruction_cls)
        
        # 创建输入
        inp = InputExample(
            key=1,
            instruction_id_list=['test_instruction'],
            prompt='Test prompt',
            kwargs=[{'param': 'value'}]
        )
        response = 'Test response'
        
        # 调用函数
        result = _test_instruction_following_loose_func(inp, response)
        
        # 所有变体都失败
        self.assertFalse(result.follow_all_instructions)
        self.assertEqual(result.follow_instruction_list, [False])
    
    @patch('ais_bench.benchmark.datasets.ifeval.evaluation_main.instructions_registry.INSTRUCTION_DICT')
    def test_loose_empty_response_in_variations(self, mock_instruction_dict):
        """测试宽松模式下空响应变体"""
        # 创建 mock 指令
        mock_instruction = MagicMock()
        mock_instruction.build_description = MagicMock()
        mock_instruction.get_instruction_args = MagicMock(return_value={})
        
        call_count = [0]
        def check_following_side_effect(text):
            call_count[0] += 1
            # 只有非空文本才返回 True
            return bool(text.strip())
        
        mock_instruction.check_following = MagicMock(side_effect=check_following_side_effect)
        
        mock_instruction_cls = MagicMock(return_value=mock_instruction)
        mock_instruction_dict.__getitem__ = MagicMock(return_value=mock_instruction_cls)
        
        # 创建输入
        inp = InputExample(
            key=1,
            instruction_id_list=['test_instruction'],
            prompt='Test prompt',
            kwargs=[{'param': 'value'}]
        )
        response = 'Test'
        
        # 调用函数
        result = _test_instruction_following_loose_func(inp, response)
        
        # 至少有一个非空变体应该通过
        self.assertTrue(result.follow_all_instructions)


if __name__ == '__main__':
    unittest.main()

