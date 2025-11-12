import unittest
from enum import Enum
from ais_bench.benchmark.utils.logging.error_codes import (
    ErrorModule,
    ErrorType,
    BaseErrorCode,
    ErrorCodeManager,
    error_manager
)


class TestErrorModule(unittest.TestCase):
    def test_error_module_values(self):
        """测试ErrorModule枚举值是否正确"""
        self.assertEqual(ErrorModule.TASK_MANAGER.value, "TMAN")
        self.assertEqual(ErrorModule.PARTITIONER.value, "PARTI")
        self.assertEqual(ErrorModule.SUMMARY.value, "SUMM")
        self.assertEqual(ErrorModule.RUNNER.value, "RUNNER")
        self.assertEqual(ErrorModule.TASK_INFER.value, "TINFER")
        self.assertEqual(ErrorModule.TASK_EVALUATE.value, "TEVAL")
        self.assertEqual(ErrorModule.TASK_MONITOR.value, "TMON")
        self.assertEqual(ErrorModule.TASK_STATUS_MANAGER.value, "TSMAN")
        self.assertEqual(ErrorModule.ICL_INFERENCER.value, "ICLI")
        self.assertEqual(ErrorModule.ICL_EVALUATOR.value, "ICLE")
        self.assertEqual(ErrorModule.ICL_RETRIEVER.value, "ICLR")
        self.assertEqual(ErrorModule.MODEL.value, "MODEL")
        self.assertEqual(ErrorModule.UTILS.value, "UTILS")
        self.assertEqual(ErrorModule.UNKNOWN.value, "UNK")

    def test_error_module_type(self):
        """测试ErrorModule是否为Enum类型"""
        self.assertTrue(issubclass(ErrorModule, Enum))


class TestErrorType(unittest.TestCase):
    def test_error_type_values(self):
        """测试ErrorType枚举值是否正确"""
        self.assertEqual(ErrorType.UNKNOWN.value, "UNK")
        self.assertEqual(ErrorType.COMMAND.value, "CMD")
        self.assertEqual(ErrorType.CONFIG.value, "CFG")
        self.assertEqual(ErrorType.MATCH.value, "MATCH")
        self.assertEqual(ErrorType.FILE.value, "FILE")

    def test_error_type_type(self):
        """测试ErrorType是否为Enum类型"""
        self.assertTrue(issubclass(ErrorType, Enum))


class TestBaseErrorCode(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.error_code = BaseErrorCode(
            code_name="UTILS-CFG-001",
            module=ErrorModule.UTILS,
            err_type=ErrorType.CONFIG,
            code=1,
            message="test error message"
        )

    def test_init(self):
        """测试BaseErrorCode初始化是否正确"""
        self.assertEqual(self.error_code.module, ErrorModule.UTILS)
        self.assertEqual(self.error_code.err_type, ErrorType.CONFIG)
        self.assertEqual(self.error_code.code, 1)
        self.assertEqual(self.error_code.message, "test error message")

    def test_full_code(self):
        """测试full_code属性是否正确生成"""
        self.assertEqual(self.error_code.full_code, "UTILS-CFG-001")

    def test_heading_id(self):
        """测试heading_id属性是否正确生成"""
        self.assertEqual(self.error_code.heading_id, "utils-cfg-001")

    def test_str(self):
        """测试__str__方法是否正确实现"""
        expected_str = "UTILS-CFG-001: test error message"
        self.assertEqual(str(self.error_code), expected_str)

    def test_faq_url(self):
        """测试faq_url是否正确生成"""
        expected_url = "https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/faqs/error_codes.html#utils-cfg-001"
        self.assertEqual(self.error_code.faq_url, expected_url)

    def test_full_code_formatting(self):
        """测试full_code的格式化是否正确，特别是数字补零"""
        # 测试代码小于10的情况
        error_code_single = BaseErrorCode("UTILS-CFG-005", ErrorModule.UTILS, ErrorType.CONFIG, 5, "test")
        self.assertEqual(error_code_single.full_code, "UTILS-CFG-005")

        # 测试代码大于等于10的情况
        error_code_double = BaseErrorCode("UTILS-CFG-012", ErrorModule.UTILS, ErrorType.CONFIG, 12, "test")
        self.assertEqual(error_code_double.full_code, "UTILS-CFG-012")

        # 测试代码大于等于100的情况
        error_code_triple = BaseErrorCode("UTILS-CFG-123", ErrorModule.UTILS, ErrorType.CONFIG, 123, "test")
        self.assertEqual(error_code_triple.full_code, "UTILS-CFG-123")

    def test_invalid_code_name(self):
        """测试code_name与full_code不匹配时抛出ValueError"""
        with self.assertRaises(ValueError) as context:
            BaseErrorCode("INVALID-CODE", ErrorModule.UTILS, ErrorType.CONFIG, 1, "test")

        self.assertIn("code_name INVALID-CODE is not equal to full_code UTILS-CFG-001", str(context.exception))


class TestErrorCodeManager(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.manager = ErrorCodeManager()
        self.error_code1 = BaseErrorCode(
            code_name="UTILS-CFG-001",
            module=ErrorModule.UTILS,
            err_type=ErrorType.CONFIG,
            code=1,
            message="test error message 1"
        )
        self.error_code2 = BaseErrorCode(
            code_name="UTILS-MATCH-001",
            module=ErrorModule.UTILS,
            err_type=ErrorType.MATCH,
            code=1,
            message="test error message 2"
        )

    def test_init(self):
        """测试ErrorCodeManager初始化是否正确"""
        self.assertEqual(len(self.manager._error_codes), 0)

    def test_register(self):
        """测试register方法是否正确注册错误码"""
        self.manager.register(self.error_code1)
        self.assertIn(self.error_code1.full_code, self.manager._error_codes)
        self.assertEqual(self.manager._error_codes[self.error_code1.full_code], self.error_code1)

    def test_register_duplicate(self):
        """测试register方法对重复错误码的处理"""
        self.manager.register(self.error_code1)
        with self.assertRaises(ValueError) as context:
            self.manager.register(self.error_code1)
        self.assertIn("error code UTILS-CFG-001 is exist!", str(context.exception))

    def test_get(self):
        """测试get方法是否正确获取错误码"""
        self.manager.register(self.error_code1)
        self.manager.register(self.error_code2)

        # 测试获取存在的错误码
        retrieved_error = self.manager.get("UTILS-CFG-001")
        self.assertEqual(retrieved_error, self.error_code1)

        # 测试获取另一个存在的错误码
        retrieved_error = self.manager.get("UTILS-MATCH-001")
        self.assertEqual(retrieved_error, self.error_code2)

        # 测试获取不存在的错误码
        non_existent_error = self.manager.get("NONEXISTENT-CODE-001")
        self.assertIsNone(non_existent_error)

    def test_list_all(self):
        """测试list_all方法是否正确返回所有错误码的副本"""
        self.manager.register(self.error_code1)
        self.manager.register(self.error_code2)

        all_errors = self.manager.list_all()
        self.assertEqual(len(all_errors), 2)
        self.assertIn("UTILS-CFG-001", all_errors)
        self.assertIn("UTILS-MATCH-001", all_errors)

        # 测试返回的是副本
        all_errors["NEW-CODE"] = "test"
        self.assertNotIn("NEW-CODE", self.manager._error_codes)


class TestErrorManagerInstance(unittest.TestCase):
    def test_error_manager_instance(self):
        """测试error_manager实例是否正确初始化"""
        self.assertIsInstance(error_manager, ErrorCodeManager)

    def test_error_manager_has_errors(self):
        """测试error_manager是否包含所有预定义的错误码"""
        all_errors = error_manager.list_all()
        # 验证至少包含一些预定义的错误码
        self.assertGreater(len(all_errors), 0)

        # 验证一些关键错误码是否存在
        self.assertIn("TMAN-UNK-001", all_errors)  # TaskManager未知错误
        self.assertIn("UTILS-UNK-001", all_errors)  # Utils未知错误
        self.assertIn("UNK-UNK-001", all_errors)   # 未知模块未知错误


class TestIntegration(unittest.TestCase):
    def test_full_error_workflow(self):
        """测试完整的错误码工作流程"""
        # 创建错误码
        error_code = BaseErrorCode(
            code_name="MODEL-CMD-005",
            module=ErrorModule.MODEL,
            err_type=ErrorType.COMMAND,
            code=5,
            message="test integration error"
        )

        # 验证属性
        self.assertEqual(error_code.full_code, "MODEL-CMD-005")
        self.assertEqual(error_code.heading_id, "model-cmd-005")
        self.assertEqual(str(error_code), "MODEL-CMD-005: test integration error")

        # 创建新的管理器并注册
        manager = ErrorCodeManager()
        manager.register(error_code)

        # 验证注册和获取
        retrieved = manager.get("MODEL-CMD-005")
        self.assertEqual(retrieved, error_code)

        # 验证列表
        all_errors = manager.list_all()
        self.assertEqual(len(all_errors), 1)
        self.assertIn("MODEL-CMD-005", all_errors)


if __name__ == '__main__':
    unittest.main()