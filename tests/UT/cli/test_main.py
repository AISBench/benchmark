import unittest
from unittest.mock import patch, MagicMock


class TestMain(unittest.TestCase):

    @patch('ais_bench.benchmark.cli.main.TaskManager')
    def test_main_function_calls_task_manager_run(self, mock_task_manager_class):
        """测试main函数调用了TaskManager的run方法"""
        # 模拟TaskManager实例
        mock_task_manager = MagicMock()
        mock_task_manager_class.return_value = mock_task_manager

        # 导入并执行main函数
        from ais_bench.benchmark.cli.main import main
        main()

        # 验证TaskManager被实例化
        mock_task_manager_class.assert_called_once()
        # 验证run方法被调用
        mock_task_manager.run.assert_called_once()

    @patch('ais_bench.benchmark.cli.main.TaskManager')
    def test_main_with_command_line_execution(self, mock_task_manager_class):
        """测试作为主程序执行时的情况"""
        # 模拟TaskManager实例
        mock_task_manager = MagicMock()
        mock_task_manager_class.return_value = mock_task_manager
        
        # 直接模拟主程序执行逻辑
        from ais_bench.benchmark.cli import main
        
        # 手动调用main函数来模拟__name__ == '__main__'的情况
        main.main()
        
        # 验证TaskManager被实例化
        mock_task_manager_class.assert_called_once()
        # 验证run方法被调用
        mock_task_manager.run.assert_called_once()

    @patch('ais_bench.benchmark.cli.main.TaskManager')
    def test_main_with_exception_handling(self, mock_task_manager_class):
        """测试main函数中TaskManager实例化或run方法抛出异常的情况"""
        # 模拟TaskManager实例化时抛出异常
        mock_task_manager_class.side_effect = Exception("TaskManager initialization failed")

        # 验证异常会被传播
        with self.assertRaises(Exception) as context:
            from ais_bench.benchmark.cli.main import main
            main()

        # 验证异常信息
        self.assertIn("TaskManager initialization failed", str(context.exception))

    @patch('ais_bench.benchmark.cli.main.TaskManager')
    def test_main_with_run_exception(self, mock_task_manager_class):
        """测试TaskManager.run方法抛出异常的情况"""
        # 模拟TaskManager实例，其run方法会抛出异常
        mock_task_manager = MagicMock()
        mock_task_manager.run.side_effect = Exception("Run method failed")
        mock_task_manager_class.return_value = mock_task_manager

        # 验证异常会被传播
        with self.assertRaises(Exception) as context:
            from ais_bench.benchmark.cli.main import main
            main()

        # 验证异常信息
        self.assertIn("Run method failed", str(context.exception))

    @patch('ais_bench.benchmark.cli.main.TaskManager')
    def test_main_multiple_calls(self, mock_task_manager_class):
        """测试多次调用main函数时TaskManager的行为"""
        # 模拟TaskManager实例
        mock_task_manager = MagicMock()
        mock_task_manager_class.return_value = mock_task_manager

        # 多次调用main函数
        from ais_bench.benchmark.cli.main import main
        main()
        main()
        main()

        # 验证TaskManager被实例化了3次
        self.assertEqual(mock_task_manager_class.call_count, 3)
        # 验证run方法被调用了3次
        self.assertEqual(mock_task_manager.run.call_count, 3)

    @patch('ais_bench.benchmark.cli.main.TaskManager')
    def test_main_with_mocked_task_manager_attributes(self, mock_task_manager_class):
        """测试main函数与TaskManager属性的交互"""
        # 模拟TaskManager实例，设置一些属性
        mock_task_manager = MagicMock()
        mock_task_manager.some_attribute = "test_value"
        mock_task_manager_class.return_value = mock_task_manager

        # 执行main函数
        from ais_bench.benchmark.cli.main import main
        main()

        # 验证TaskManager被正确使用
        mock_task_manager_class.assert_called_once()
        mock_task_manager.run.assert_called_once()


if __name__ == '__main__':
    unittest.main()