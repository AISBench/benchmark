import unittest
import sys
import os
import json
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.livecodebench.testing_util import (
        CODE_TYPE,
        TimeoutException,
        Capturing,
        truncatefn,
        only_int_check,
        string_int_check,
        combined_int_check,
        custom_compare_,
        stripped_string_compare,
        timeout_handler,
        run_test,
        call_method,
        reliability_guard
    )
    TESTING_UTIL_AVAILABLE = True
except ImportError:
    TESTING_UTIL_AVAILABLE = False


class TestingUtilTestBase(unittest.TestCase):
    """TestingUtil测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not TESTING_UTIL_AVAILABLE:
            cls.skipTest(cls, "TestingUtil modules not available")


class TestCODE_TYPE(TestingUtilTestBase):
    """测试CODE_TYPE枚举"""
    
    def test_code_type_enum(self):
        """测试CODE_TYPE枚举值"""
        self.assertEqual(CODE_TYPE.call_based.value, 0)
        self.assertEqual(CODE_TYPE.standard_input.value, 1)


class TestTimeoutException(TestingUtilTestBase):
    """测试TimeoutException类"""
    
    def test_timeout_exception(self):
        """测试超时异常"""
        exc = TimeoutException()
        self.assertIsInstance(exc, Exception)


class TestTimeoutHandler(TestingUtilTestBase):
    """测试timeout_handler函数"""
    
    def test_timeout_handler(self):
        """测试超时处理器"""
        # timeout_handler会抛出TimeoutException
        with self.assertRaises(TimeoutException):
            timeout_handler(None, None)


class TestCapturing(TestingUtilTestBase):
    """测试Capturing类"""
    
    def test_capturing_context_manager(self):
        """测试Capturing上下文管理器"""
        import sys
        
        with Capturing() as output:
            print('test output')
            print('another line')
        
        self.assertEqual(len(output), 1)
        self.assertIn('test output', output[0])
        self.assertIn('another line', output[0])


class TestTruncatefn(TestingUtilTestBase):
    """测试truncatefn函数"""
    
    def test_truncate_short_string(self):
        """测试截断短字符串"""
        s = 'short'
        result = truncatefn(s, length=300)
        self.assertEqual(result, 'short')
    
    def test_truncate_long_string(self):
        """测试截断长字符串"""
        s = 'a' * 500
        result = truncatefn(s, length=100)
        self.assertLess(len(result), len(s))
        self.assertIn('(truncated)', result)
    
    def test_truncate_exact_length(self):
        """测试截断刚好等于长度的字符串"""
        s = 'a' * 100
        result = truncatefn(s, length=100)
        self.assertEqual(result, s)


class TestIntCheckFunctions(TestingUtilTestBase):
    """测试整数检查函数"""
    
    def test_only_int_check(self):
        """测试only_int_check函数"""
        self.assertTrue(only_int_check(1))
        self.assertTrue(only_int_check(0))
        self.assertFalse(only_int_check('1'))
        self.assertFalse(only_int_check(1.5))
    
    def test_string_int_check(self):
        """测试string_int_check函数"""
        self.assertTrue(string_int_check('1'))
        self.assertTrue(string_int_check('0'))
        self.assertTrue(string_int_check('123'))
        self.assertFalse(string_int_check('abc'))
        self.assertFalse(string_int_check('1.5'))
        self.assertFalse(string_int_check(1))
    
    def test_combined_int_check(self):
        """测试combined_int_check函数"""
        self.assertTrue(combined_int_check(1))
        self.assertTrue(combined_int_check('1'))
        self.assertTrue(combined_int_check(0))
        self.assertTrue(combined_int_check('0'))
        self.assertFalse(combined_int_check('abc'))
        self.assertFalse(combined_int_check(1.5))


class TestCustomCompare(TestingUtilTestBase):
    """测试custom_compare_函数"""
    
    def test_custom_compare_list_match(self):
        """测试自定义比较列表匹配"""
        output = ['line1', 'line2']
        ground_truth = 'line1\nline2'
        result = custom_compare_(output, ground_truth)
        self.assertTrue(result)
    
    def test_custom_compare_list_no_match(self):
        """测试自定义比较列表不匹配"""
        output = ['line1', 'line2']
        ground_truth = 'different'
        result = custom_compare_(output, ground_truth)
        self.assertFalse(result)
    
    def test_custom_compare_empty_list(self):
        """测试自定义比较空列表"""
        # 空列表join后是空字符串，与空字符串去除空白后相等，所以应该返回True
        output = []
        ground_truth = ''
        result = custom_compare_(output, ground_truth)
        self.assertTrue(result)  # 空列表join后等于空字符串


class TestStrippedStringCompare(TestingUtilTestBase):
    """测试stripped_string_compare函数"""
    
    def test_stripped_compare_equal(self):
        """测试去除空白后比较相等的字符串"""
        s1 = '  test  '
        s2 = 'test'
        result = stripped_string_compare(s1, s2)
        self.assertTrue(result)
    
    def test_stripped_compare_different(self):
        """测试去除空白后比较不等的字符串"""
        s1 = '  test1  '
        s2 = 'test2'
        result = stripped_string_compare(s1, s2)
        self.assertFalse(result)
    
    def test_stripped_compare_multiline(self):
        """测试去除空白后比较多行字符串"""
        s1 = '  line1\n  line2  '
        s2 = 'line1\nline2'
        result = stripped_string_compare(s1, s2)
        # 注意：stripped_string_compare只去除首尾空白，不处理内部空白
        self.assertFalse(result)  # 因为内部换行符前后的空白没有被去除


class TestRunTest(TestingUtilTestBase):
    """测试run_test函数"""
    
    def test_run_test_exists(self):
        """测试run_test函数存在"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import testing_util
        
        with patch.object(testing_util, 'reliability_guard'):
            # 只测试函数可以被导入和调用
            self.assertTrue(callable(run_test))


class TestCallMethod(TestingUtilTestBase):
    """测试call_method函数"""
    
    @patch('builtins.open')
    @patch('sys.stdin')
    def test_call_method(self, mock_stdin, mock_open):
        """测试call_method函数"""
        from io import StringIO
        mock_stdin_io = StringIO('test input')
        
        def test_func():
            return "test result"
        
        # call_method会mock stdin等，测试基本调用
        result = call_method(test_func, "test input")
        self.assertEqual(result, "test result")


class TestReliabilityGuardTestingUtil(TestingUtilTestBase):
    """测试reliability_guard函数（testing_util中的）"""
    
    def test_reliability_guard_exists(self):
        """测试reliability_guard函数存在且可调用"""
        # reliability_guard会修改大量全局状态（禁用危险函数），在单元测试中执行可能会影响后续测试
        # 因此只验证函数存在且可调用即可
        self.assertTrue(callable(reliability_guard))
        self.assertIsNotNone(reliability_guard)


if __name__ == '__main__':
    unittest.main()

