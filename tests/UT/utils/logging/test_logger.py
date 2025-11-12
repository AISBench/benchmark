import unittest
import logging
import os
import tempfile
from unittest.mock import patch, MagicMock

from ais_bench.benchmark.utils.logging.logger import (
    Colors,
    ColoredLevelFormatter,
    to_error_code_format,
    to_url_format,
    to_code_msg_format,
    get_formatted_log_content,
    AISLogger
)
from ais_bench.benchmark.utils.logging.error_codes import error_manager, ErrorType
from ais_bench.benchmark.global_consts import LOG_LEVEL


class TestColors(unittest.TestCase):
    def test_colors_exist(self):
        # 测试Colors类中的所有颜色和样式常量是否存在
        self.assertTrue(hasattr(Colors, 'BLACK'))
        self.assertTrue(hasattr(Colors, 'RED'))
        self.assertTrue(hasattr(Colors, 'GREEN'))
        self.assertTrue(hasattr(Colors, 'YELLOW'))
        self.assertTrue(hasattr(Colors, 'BLUE'))
        self.assertTrue(hasattr(Colors, 'MAGENTA'))
        self.assertTrue(hasattr(Colors, 'WHITE'))
        self.assertTrue(hasattr(Colors, 'BG_BLACK'))
        self.assertTrue(hasattr(Colors, 'BG_RED'))
        self.assertTrue(hasattr(Colors, 'BOLD'))
        self.assertTrue(hasattr(Colors, 'UNDERLINE'))
        self.assertTrue(hasattr(Colors, 'RESET'))


class TestFormattingFunctions(unittest.TestCase):
    def test_to_error_code_format(self):
        # 测试错误码格式化函数
        test_str = "TEST-CODE"
        result = to_error_code_format(test_str)
        self.assertIn(Colors.BOLD, result)
        self.assertIn(Colors.BG_RED, result)
        self.assertIn(Colors.YELLOW, result)
        self.assertIn(Colors.RESET, result)
        self.assertIn(test_str, result)

    def test_to_url_format(self):
        # 测试URL格式化函数
        test_str = "http://example.com"
        result = to_url_format(test_str)
        self.assertIn(Colors.UNDERLINE, result)
        self.assertIn(Colors.MAGENTA, result)
        self.assertIn(Colors.RESET, result)
        self.assertIn(test_str, result)

    def test_to_code_msg_format(self):
        # 测试代码消息格式化函数
        test_str = "Error message"
        result = to_code_msg_format(test_str)
        self.assertIn(Colors.BOLD, result)
        self.assertIn(Colors.RED, result)
        self.assertIn(Colors.RESET, result)
        self.assertIn(test_str, result)


class TestColoredLevelFormatter(unittest.TestCase):
    def test_format_debug(self):
        # 测试DEBUG级别日志的格式化
        formatter = ColoredLevelFormatter()
        record = logging.LogRecord(
            name='test_logger',
            level=logging.DEBUG,
            pathname='test.py',
            lineno=10,
            msg='Debug message',
            args=(),
            exc_info=None
        )
        result = formatter.format(record)
        self.assertIn(Colors.BLUE, result)
        self.assertIn('DEBUG', result)
        self.assertIn('test.py', result)
        self.assertIn('10', result)

    def test_format_info(self):
        # 测试INFO级别日志的格式化
        formatter = ColoredLevelFormatter()
        record = logging.LogRecord(
            name='test_logger',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Info message',
            args=(),
            exc_info=None
        )
        result = formatter.format(record)
        self.assertIn(Colors.GREEN, result)
        self.assertIn('INFO', result)
        self.assertNotIn('test.py', result)  # 非DEBUG级别不应该包含路径信息

    def test_format_warning(self):
        # 测试WARNING级别日志的格式化
        formatter = ColoredLevelFormatter()
        record = logging.LogRecord(
            name='test_logger',
            level=logging.WARNING,
            pathname='test.py',
            lineno=10,
            msg='Warning message',
            args=(),
            exc_info=None
        )
        result = formatter.format(record)
        self.assertIn(Colors.YELLOW, result)
        self.assertIn('WARNING', result)

    def test_format_error(self):
        # 测试ERROR级别日志的格式化
        formatter = ColoredLevelFormatter()
        record = logging.LogRecord(
            name='test_logger',
            level=logging.ERROR,
            pathname='test.py',
            lineno=10,
            msg='Error message',
            args=(),
            exc_info=None
        )
        result = formatter.format(record)
        self.assertIn(Colors.BOLD, result)
        self.assertIn(Colors.RED, result)
        self.assertIn('ERROR', result)


class TestGetFormattedLogContent(unittest.TestCase):
    def test_valid_error_code(self):
        # 测试使用有效的错误码
        valid_code = "UTILS-UNK-001"  # 从error_codes.py中存在的错误码
        msg = "Test message"
        result = get_formatted_log_content(valid_code, msg)

        self.assertIn(valid_code, result)
        self.assertIn(msg, result)
        # 由于这是UNKNOWN类型的错误，不应该包含URL
        self.assertNotIn("Visit", result)

    def test_error_code_with_url(self):
        # 测试使用包含URL的错误码（非UNKNOWN类型）
        # 查找一个非UNKNOWN类型的错误码
        non_unknown_code = None
        for code_str, error_code in error_manager.list_all().items():
            if error_code.err_type != ErrorType.UNKNOWN:
                non_unknown_code = code_str
                break

        if non_unknown_code:
            msg = "Test message"
            result = get_formatted_log_content(non_unknown_code, msg)
            self.assertIn(non_unknown_code, result)
            self.assertIn(msg, result)
            self.assertIn("Visit", result)
            self.assertIn("for further help", result)

    def test_invalid_error_code(self):
        # 测试使用无效的错误码
        invalid_code = "INVALID-CODE-999"
        msg = "Test message"
        with self.assertRaises(ValueError):
            get_formatted_log_content(invalid_code, msg)


class TestAISLogger(unittest.TestCase):
    def setUp(self):
        # 创建一个临时文件用于测试日志文件输出
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()

    def tearDown(self):
        # 清理临时文件
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    @patch('logging.getLogger')
    def test_init_main_process_with_log_file(self, mock_getLogger):
        # 测试主进程初始化，带日志文件
        mock_logger = MagicMock()
        mock_getLogger.return_value = mock_logger
        mock_logger.handlers = []
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_logger.root = mock_root

        logger = AISLogger(name="test_logger", level=logging.INFO,
                          is_main_process=True, log_file=self.temp_file.name)

        mock_getLogger.assert_called_with("test_logger")
        self.assertEqual(mock_logger.propagate, False)
        mock_logger.setLevel.assert_called_with(logging.INFO)

    @patch('logging.getLogger')
    def test_init_main_process_without_log_file(self, mock_getLogger):
        # 测试主进程初始化，不带日志文件
        mock_logger = MagicMock()
        mock_getLogger.return_value = mock_logger
        mock_logger.handlers = []
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_logger.root = mock_root

        logger = AISLogger(name="test_logger", level=logging.INFO,
                          is_main_process=True, log_file=None)

        mock_getLogger.assert_called_with("test_logger")
        mock_logger.setLevel.assert_called_with(logging.INFO)

    @patch('logging.getLogger')
    def test_init_subprocess(self, mock_getLogger):
        # 测试子进程初始化
        mock_logger = MagicMock()
        mock_getLogger.return_value = mock_logger
        mock_logger.handlers = []
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_logger.root = mock_root

        logger = AISLogger(name="test_logger", level=logging.INFO,
                          is_main_process=False)

        mock_logger.setLevel.assert_called_with(logging.ERROR)  # SUBPROCESS_LOG_LEVEL

    @patch('logging.getLogger')
    def test_init_with_existing_handlers(self, mock_getLogger):
        # 测试初始化时清理现有处理器
        mock_logger = MagicMock()
        mock_getLogger.return_value = mock_logger
        # 创建一个模拟的handlers列表，它有clear方法
        mock_handlers = MagicMock()
        mock_logger.handlers = mock_handlers
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_logger.root = mock_root

        logger = AISLogger(name="test_logger")

        # 验证handlers被清空
        mock_handlers.clear.assert_called_once()

    @patch('logging.getLogger')
    def test_info_method(self, mock_getLogger):
        # 测试info方法
        mock_logger = MagicMock()
        mock_getLogger.return_value = mock_logger
        mock_logger.handlers = []
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_logger.root = mock_root

        logger = AISLogger()
        logger.info("Test info message")

        mock_logger.info.assert_called_with("Test info message", stacklevel=2)

    @patch('logging.getLogger')
    def test_debug_method(self, mock_getLogger):
        # 测试debug方法
        mock_logger = MagicMock()
        mock_getLogger.return_value = mock_logger
        mock_logger.handlers = []
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_logger.root = mock_root

        logger = AISLogger(level=logging.DEBUG)
        logger.debug("Test debug message")

        mock_logger.debug.assert_called_with("Test debug message", stacklevel=2)

    @patch('logging.getLogger')
    def test_warning_method(self, mock_getLogger):
        # 测试warning方法
        mock_logger = MagicMock()
        mock_getLogger.return_value = mock_logger
        mock_logger.handlers = []
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_logger.root = mock_root

        logger = AISLogger()
        logger.warning("Test warning message")

        mock_logger.warning.assert_called_with("Test warning message", stacklevel=2)

    @patch('logging.getLogger')
    def test_error_method(self, mock_getLogger):
        # 测试error方法
        mock_logger = MagicMock()
        mock_getLogger.return_value = mock_logger
        mock_logger.handlers = []
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_logger.root = mock_root

        # 测试使用BaseErrorCode对象
        from ais_bench.benchmark.utils.logging.error_codes import BaseErrorCode, ErrorModule
        mock_error_code = MagicMock(spec=BaseErrorCode)
        mock_error_code.full_code = "UTILS-UNK-001"
        mock_error_code.err_type = ErrorType.UNKNOWN
        mock_error_code.message = "Unknown error"

        # 模拟get_formatted_log_content函数
        with patch('ais_bench.benchmark.utils.logging.logger.get_formatted_log_content') as mock_formatted_log:
            mock_formatted_log.return_value = "Formatted error message"
            
            logger = AISLogger()
            logger.error(mock_error_code, "Test error message")
            
            # 验证调用了get_formatted_log_content和logger.error
            mock_formatted_log.assert_called_once_with("UTILS-UNK-001", "Test error message")
            mock_logger.error.assert_called_once_with("Formatted error message", stacklevel=2)
    
    @patch('logging.getLogger')
    def test_error_method_with_invalid_error_code(self, mock_getLogger):
        # 测试使用非BaseErrorCode对象时抛出ValueError
        mock_logger = MagicMock()
        mock_getLogger.return_value = mock_logger
        mock_logger.handlers = []
        mock_root = MagicMock()
        mock_root.handlers = []
        mock_logger.root = mock_root
        
        logger = AISLogger()
        
        # 使用字符串作为错误码应该抛出ValueError
        with self.assertRaises(ValueError) as context:
            logger.error("UTILS-UNK-001", "Test error message")
        
        self.assertIn("error_code UTILS-UNK-001 is not instance of BaseErrorCode!", str(context.exception))
        
        # 使用其他类型的对象也应该抛出ValueError
        with self.assertRaises(ValueError):
            logger.error(123, "Test error message")


if __name__ == '__main__':
    unittest.main()