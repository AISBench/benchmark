import unittest
from unittest.mock import patch, MagicMock

# 导入被测试的模块
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchBaseException,
    PerfResultCalcException,
    CommandError,
    ConfigError,
    FileMatchError
)
from ais_bench.benchmark.utils.logging.error_codes import BaseErrorCode


class TestAISBenchBaseException(unittest.TestCase):
    """测试AISBenchBaseException类"""

    def setUp(self):
        """设置测试环境"""
        # 模拟有效的BaseErrorCode对象
        self.mock_error_code = MagicMock(spec=BaseErrorCode)
        self.mock_error_code.full_code = "TEST-ERR-001"
        self.mock_error_code.error_code = "TEST-ERR-001"  # 假设exception中使用了error_code属性
        self.mock_error_code.message = "测试错误消息"

        # 模拟格式化后的日志内容
        self.formatted_log_content = "格式化后的日志内容"

    @patch('ais_bench.benchmark.utils.logging.exceptions.error_manager')
    @patch('ais_bench.benchmark.utils.logging.exceptions.get_formatted_log_content')
    def test_init_with_valid_error_code(self, mock_get_formatted, mock_manager):
        """测试使用有效的BaseErrorCode对象初始化异常"""
        # 设置模拟对象的返回值
        mock_manager.get.return_value = self.mock_error_code
        mock_get_formatted.return_value = self.formatted_log_content

        # 创建异常实例
        exception = AISBenchBaseException(self.mock_error_code)

        # 验证调用了error_manager.get和get_formatted_log_content
        mock_manager.get.assert_called_once_with(self.mock_error_code.full_code)
        mock_get_formatted.assert_called_once_with(self.mock_error_code.error_code, None)

        # 验证异常的属性
        self.assertEqual(exception.error_code_str, self.mock_error_code.full_code)
        self.assertEqual(str(exception), self.formatted_log_content)

    @patch('ais_bench.benchmark.utils.logging.exceptions.error_manager')
    @patch('ais_bench.benchmark.utils.logging.exceptions.get_formatted_log_content')
    def test_init_with_message(self, mock_get_formatted, mock_manager):
        """测试提供message参数的情况"""
        # 设置模拟对象的返回值
        mock_manager.get.return_value = self.mock_error_code
        mock_get_formatted.return_value = self.formatted_log_content

        # 创建异常实例并提供message参数
        custom_message = "自定义错误消息"
        exception = AISBenchBaseException(self.mock_error_code, custom_message)

        # 验证调用了error_manager.get和get_formatted_log_content
        mock_manager.get.assert_called_once_with(self.mock_error_code.full_code)
        mock_get_formatted.assert_called_once_with(self.mock_error_code.error_code, custom_message)

        # 验证异常的message属性
        self.assertEqual(str(exception), self.formatted_log_content)

    @patch('ais_bench.benchmark.utils.logging.exceptions.error_manager')
    def test_init_with_invalid_error_code_type(self, mock_manager):
        """测试使用非BaseErrorCode类型的对象初始化异常时抛出ValueError"""
        # 使用字符串作为错误码应该抛出ValueError
        invalid_error_code = "INVALID-ERR-999"
        
        # 验证抛出ValueError
        with self.assertRaises(ValueError) as context:
            AISBenchBaseException(invalid_error_code)
        
        self.assertIn(f"error_code {invalid_error_code} is not instance of BaseErrorCode!", str(context.exception))
        
        # 使用其他类型的对象也应该抛出ValueError
        with self.assertRaises(ValueError):
            AISBenchBaseException(123)
    
    @patch('ais_bench.benchmark.utils.logging.exceptions.error_manager')
    def test_init_with_nonexistent_error_code(self, mock_manager):
        """测试使用不存在的错误码初始化异常时抛出ValueError"""
        # 设置模拟对象返回None，表示错误码不存在
        mock_manager.get.return_value = None

        # 验证抛出ValueError
        with self.assertRaises(ValueError) as context:
            AISBenchBaseException(self.mock_error_code)

        # 验证异常消息
        self.assertIn(f"error_code {self.mock_error_code.full_code} is not exist!", str(context.exception))

        # 验证调用了error_manager.get
        mock_manager.get.assert_called_once_with(self.mock_error_code.full_code)

class TestExceptionSubclasses(unittest.TestCase):
    """测试异常子类"""
    
    def test_subclass_inheritance(self):
        """测试异常子类是否正确继承AISBenchBaseException"""
        self.assertTrue(issubclass(PerfResultCalcException, AISBenchBaseException))
        self.assertTrue(issubclass(CommandError, AISBenchBaseException))
        self.assertTrue(issubclass(ConfigError, AISBenchBaseException))
        self.assertTrue(issubclass(FileMatchError, AISBenchBaseException))
    
    @patch('ais_bench.benchmark.utils.logging.exceptions.error_manager')
    @patch('ais_bench.benchmark.utils.logging.exceptions.get_formatted_log_content')
    def test_subclass_initialization(self, mock_get_formatted, mock_manager):
        """测试异常子类的初始化"""
        mock_error_code = MagicMock(spec=BaseErrorCode)
        mock_error_code.full_code = "TEST-ERR-001"
        mock_error_code.error_code = "TEST-ERR-001"
        mock_manager.get.return_value = mock_error_code
        mock_get_formatted.return_value = "Formatted message"
        
        # 测试每个子类都能正确初始化
        for exception_class in [PerfResultCalcException, CommandError, ConfigError, FileMatchError]:
            exception = exception_class(mock_error_code)
            self.assertIsInstance(exception, AISBenchBaseException)
            self.assertEqual(exception.error_code_str, "TEST-ERR-001")

    @patch('ais_bench.benchmark.utils.logging.exceptions.error_manager')
    @patch('ais_bench.benchmark.utils.logging.exceptions.get_formatted_log_content')
    def test_inheritance(self, mock_get_formatted, mock_manager):
        """测试异常类的继承关系"""
        # 创建模拟的BaseErrorCode对象
        mock_error_code = MagicMock(spec=BaseErrorCode)
        mock_error_code.full_code = "TEST-ERR-001"
        mock_error_code.error_code = "TEST-ERR-001"
        
        # 设置模拟对象的返回值
        mock_manager.get.return_value = mock_error_code
        mock_get_formatted.return_value = "Formatted message"

        # 创建异常实例
        exception = AISBenchBaseException(mock_error_code)

        # 验证继承关系
        self.assertIsInstance(exception, Exception)
        self.assertIsInstance(exception, AISBenchBaseException)


class TestPerfResultCalcException(unittest.TestCase):
    """测试PerfResultCalcException类"""

    @patch('ais_bench.benchmark.utils.logging.exceptions.error_manager')
    @patch('ais_bench.benchmark.utils.logging.exceptions.get_formatted_log_content')
    def test_inheritance(self, mock_get_formatted, mock_manager):
        """测试PerfResultCalcException类的继承关系"""
        # 创建模拟的BaseErrorCode对象
        mock_error_code = MagicMock(spec=BaseErrorCode)
        mock_error_code.full_code = "TEST-ERR-001"
        mock_error_code.error_code = "TEST-ERR-001"
        
        # 设置模拟对象的返回值
        mock_manager.get.return_value = mock_error_code
        mock_get_formatted.return_value = "格式化后的日志内容"

        # 创建异常实例
        exception = PerfResultCalcException(mock_error_code)

        # 验证继承关系
        self.assertIsInstance(exception, Exception)
        self.assertIsInstance(exception, AISBenchBaseException)
        self.assertIsInstance(exception, PerfResultCalcException)


if __name__ == '__main__':
    unittest.main()