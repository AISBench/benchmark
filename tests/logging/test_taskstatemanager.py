import os
import sys
import json
import unittest
import tempfile
from unittest import mock
import time

# 导入被测试的类
from ais_bench.benchmark.tasks.base import TaskStateManager
from ais_bench.benchmark.utils.file import write_status


class TestTaskStateManager(unittest.TestCase):
    def setUp(self):
        # 创建临时目录作为测试环境
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = self.tmp_dir.name
        self.task_name = "test_task"
        self.is_debug = False
        self.refresh_interval = 0.1

        # 保存原始的os.getpid函数
        self.original_getpid = os.getpid
        # 保存原始的write_status函数
        self.original_write_status = write_status
        # 保存原始的time.time和time.sleep函数
        self.original_time = time.time
        self.original_sleep = time.sleep

        # 模拟进程ID
        self.mock_pid = 12345
        os.getpid = mock.MagicMock(return_value=self.mock_pid)

        # 模拟时间戳
        self.mock_timestamp = 1620000000.0
        time.time = mock.MagicMock(return_value=self.mock_timestamp)

        # 模拟time.sleep
        time.sleep = mock.MagicMock()

        # 模拟write_status函数 - 修复关键位置：同时替换两个地方的引用
        self.mock_write_status = mock.MagicMock(return_value=True)
        # 替换utils.file模块中的write_status
        sys.modules['ais_bench.benchmark.utils.file'].write_status = self.mock_write_status
        # 替换base模块中已经导入的write_status引用
        sys.modules['ais_bench.benchmark.tasks.base'].write_status = self.mock_write_status

    def tearDown(self):
        # 恢复原始函数
        os.getpid = self.original_getpid
        time.time = self.original_time
        time.sleep = self.original_sleep
        # 恢复两个地方的write_status引用
        sys.modules['ais_bench.benchmark.utils.file'].write_status = self.original_write_status
        sys.modules['ais_bench.benchmark.tasks.base'].write_status = self.original_write_status

        # 清理临时目录
        self.tmp_dir.cleanup()

    def test_init(self):
        # 测试初始化功能
        task_manager = TaskStateManager(
            self.tmp_path,
            self.task_name,
            self.is_debug,
            self.refresh_interval
        )

        # 验证临时文件路径是否正确
        expected_tmp_file = os.path.join(self.tmp_path, f"tmp_{self.task_name}.json")
        self.assertEqual(task_manager.tmp_file, expected_tmp_file)

        # 验证临时文件是否被创建且内容为空列表
        self.assertTrue(os.path.exists(expected_tmp_file))
        with open(expected_tmp_file, 'r') as f:
            content = json.load(f)
            self.assertEqual(content, [])

        # 验证任务状态初始化是否正确
        self.assertEqual(task_manager.task_state["task_name"], self.task_name)
        self.assertEqual(task_manager.task_state["process_id"], self.mock_pid)
        self.assertEqual(task_manager.is_debug, self.is_debug)
        self.assertEqual(task_manager.refresh_interval, self.refresh_interval)

    def test_init_with_existing_file(self):
        # 测试初始化时存在已有文件的情况
        existing_file = os.path.join(self.tmp_path, f"tmp_{self.task_name}.json")

        # 创建一个已存在的文件并写入内容
        with open(existing_file, 'w') as f:
            json.dump([{"test": "data"}], f)

        # 初始化TaskStateManager
        task_manager = TaskStateManager(
            self.tmp_path,
            self.task_name,
            self.is_debug
        )

        # 验证已有文件是否被清空
        with open(existing_file, 'r') as f:
            content = json.load(f)
            self.assertEqual(content, [])

    def test_init_with_special_characters_in_task_name(self):
        # 测试任务名包含特殊字符的情况
        special_task_name = "task/with/slashes"
        task_manager = TaskStateManager(
            self.tmp_path,
            special_task_name,
            self.is_debug
        )

        # 验证斜杠是否被替换为下划线
        expected_tmp_file = os.path.join(self.tmp_path, "tmp_task_with_slashes.json")
        self.assertEqual(task_manager.tmp_file, expected_tmp_file)
        self.assertTrue(os.path.exists(expected_tmp_file))

    @mock.patch('builtins.print')
    def test_launch_debug_mode(self, mock_print):
        # 测试在调试模式下启动
        task_manager = TaskStateManager(
            self.tmp_path,
            self.task_name,
            True  # is_debug=True
        )

        # 模拟_display_task_state方法
        task_manager._display_task_state = mock.MagicMock()

        # 调用launch方法
        task_manager.launch()

        # 验证start_time是否被设置
        self.assertEqual(task_manager.task_state["start_time"], self.mock_timestamp)

        # 验证print和_display_task_state被调用
        mock_print.assert_called_once_with("debug mode, print progress directly")
        task_manager._display_task_state.assert_called_once()

        # 验证在调试模式下不会调用_post_task_state
        with mock.patch.object(task_manager, '_post_task_state') as mock_post:
            task_manager.launch()
            mock_post.assert_not_called()

    def test_launch_non_debug_mode(self):
        # 测试在非调试模式下启动
        task_manager = TaskStateManager(
            self.tmp_path,
            self.task_name,
            False  # is_debug=False
        )

        # 模拟_post_task_state方法以避免无限循环
        task_manager._post_task_state = mock.MagicMock()

        # 调用launch方法
        task_manager.launch()

        # 验证start_time是否被设置
        self.assertEqual(task_manager.task_state["start_time"], self.mock_timestamp)

        # 验证在非调试模式下会调用_post_task_state
        task_manager._post_task_state.assert_called_once()

    def test_update_task_state(self):
        # 测试更新任务状态
        task_manager = TaskStateManager(
            self.tmp_path,
            self.task_name,
            self.is_debug
        )

        # 准备更新的状态
        update_state = {
            "status": "running",
            "progress": 50,
            "some_key": "some_value"
        }

        # 更新任务状态
        task_manager.update_task_state(update_state)

        # 验证状态是否被正确更新
        self.assertEqual(task_manager.task_state["status"], "running")
        self.assertEqual(task_manager.task_state["progress"], 50)
        self.assertEqual(task_manager.task_state["some_key"], "some_value")
        # 验证原有状态是否保留
        self.assertEqual(task_manager.task_state["task_name"], self.task_name)
        self.assertEqual(task_manager.task_state["process_id"], self.mock_pid)

    def test_post_task_state_finish_status(self):
        # 测试状态为finish时退出循环
        task_manager = TaskStateManager(
            self.tmp_path,
            self.task_name,
            self.is_debug,
            self.refresh_interval
        )

        # 设置状态为finish
        task_manager.task_state["status"] = "finish"

        # 调用_post_task_state
        task_manager._post_task_state()

        # 验证write_status被调用
        self.mock_write_status.assert_called_once_with(task_manager.tmp_file, task_manager.task_state)
        # 验证time.sleep没有被调用（因为状态是finish，直接退出循环）
        time.sleep.assert_not_called()

    def test_post_task_state_error_status(self):
        # 测试状态为error时退出循环
        task_manager = TaskStateManager(
            self.tmp_path,
            self.task_name,
            self.is_debug,
            self.refresh_interval
        )

        # 设置状态为error
        task_manager.task_state["status"] = "error"

        # 调用_post_task_state
        task_manager._post_task_state()

        # 验证write_status被调用
        self.mock_write_status.assert_called_once_with(task_manager.tmp_file, task_manager.task_state)
        # 验证time.sleep没有被调用（因为状态是error，直接退出循环）
        time.sleep.assert_not_called()

    def test_post_task_state_write_failure(self):
        # 测试write_status失败的情况
        task_manager = TaskStateManager(
            self.tmp_path,
            self.task_name,
            self.is_debug,
            self.refresh_interval
        )

        # 设置write_status返回False表示写入失败
        self.mock_write_status.return_value = False

        # 设置状态为finish以退出循环
        task_manager.task_state["status"] = "finish"

        # 调用_post_task_state
        task_manager._post_task_state()

        # 验证write_status被调用
        self.mock_write_status.assert_called_once_with(task_manager.tmp_file, task_manager.task_state)

    def test_display_task_state(self):
        # 测试_display_task_state方法（空实现）
        task_manager = TaskStateManager(
            self.tmp_path,
            self.task_name,
            self.is_debug
        )

        # 调用方法，确保不会抛出异常
        try:
            task_manager._display_task_state()
            # 如果执行到这里，说明方法调用成功
            success = True
        except Exception:
            success = False

        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()