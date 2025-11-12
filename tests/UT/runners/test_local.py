import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

from mmengine.config import ConfigDict

from ais_bench.benchmark.runners.local import LocalRunner, get_command_template


class TestGetCommandTemplate(unittest.TestCase):
    """测试get_command_template函数"""

    def test_get_command_template_cuda(self):
        """测试CUDA设备模板"""
        gpu_ids = [0, 1, 2]
        
        with patch('ais_bench.benchmark.runners.local.is_npu_available', return_value=False):
            with patch('ais_bench.benchmark.runners.local.sys.platform', 'linux'):
                result = get_command_template(gpu_ids)
                self.assertIn("CUDA_VISIBLE_DEVICES=0,1,2", result)
                self.assertIn("{task_cmd}", result)

    def test_get_command_template_npu(self):
        """测试NPU设备模板"""
        gpu_ids = [0, 1]
        
        with patch('ais_bench.benchmark.runners.local.is_npu_available', return_value=True):
            result = get_command_template(gpu_ids)
            self.assertIn("ASCEND_RT_VISIBLE_DEVICES=0,1", result)
            self.assertIn("{task_cmd}", result)

    def test_get_command_template_windows(self):
        """测试Windows平台模板"""
        gpu_ids = [0, 1]
        
        with patch('ais_bench.benchmark.runners.local.is_npu_available', return_value=False):
            with patch('ais_bench.benchmark.runners.local.sys.platform', 'win32'):
                result = get_command_template(gpu_ids)
                self.assertIn("set CUDA_VISIBLE_DEVICES=0,1", result)
                self.assertIn("&", result)
                self.assertIn("{task_cmd}", result)

    def test_get_command_template_empty(self):
        """测试空GPU列表"""
        gpu_ids = []
        
        with patch('ais_bench.benchmark.runners.local.is_npu_available', return_value=False):
            with patch('ais_bench.benchmark.runners.local.sys.platform', 'linux'):
                result = get_command_template(gpu_ids)
                self.assertIn("CUDA_VISIBLE_DEVICES=", result)
                self.assertIn("{task_cmd}", result)


class TestLocalRunner(unittest.TestCase):
    """测试LocalRunner类"""

    def setUp(self):
        """设置测试环境"""
        self.task_cfg = ConfigDict({
            "type": "OpenICLInferTask"
        })
        self.debug = False
        self.max_num_workers = 4
        self.max_workers_per_gpu = 1

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_init(self, mock_logger_class):
        """测试LocalRunner初始化"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=self.debug,
            max_num_workers=self.max_num_workers,
            max_workers_per_gpu=self.max_workers_per_gpu
        )
        
        self.assertEqual(runner.debug, self.debug)
        self.assertEqual(runner.max_num_workers, self.max_num_workers)
        self.assertEqual(runner.max_workers_per_gpu, self.max_workers_per_gpu)
        self.assertFalse(runner.keep_tmp_file)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_init_with_keep_tmp_file(self, mock_logger_class):
        """测试使用keep_tmp_file参数初始化"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=self.debug,
            keep_tmp_file=True
        )
        
        self.assertTrue(runner.keep_tmp_file)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_init_with_unknown_kwargs(self, mock_logger_class):
        """测试使用未知参数时记录警告"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=self.debug,
            unknown_param="test"
        )
        
        # 验证记录了警告
        mock_logger.warning.assert_called()
        call_args = mock_logger.warning.call_args[0][0]
        self.assertIn("unknown_param", call_args)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    @patch('ais_bench.benchmark.runners.local.LocalRunner._run_debug')
    def test_launch_debug_mode(self, mock_run_debug, mock_logger_class):
        """测试debug模式下的launch方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=True
        )
        
        tasks = [{
            "work_dir": "/tmp/test",
            "cli_args": {"run_in_background": False}
        }]
        
        mock_run_debug.return_value = [("task1", 0)]
        
        with patch('ais_bench.benchmark.runners.local.multiprocessing.Process') as mock_process:
            mock_process_instance = MagicMock()
            mock_process.return_value = mock_process_instance
            
            with patch('ais_bench.benchmark.runners.local.task_abbr_from_cfg', return_value="task1"):
                with patch('ais_bench.benchmark.runners.local.is_npu_available', return_value=False):
                    with patch('ais_bench.benchmark.runners.local.torch.cuda.device_count', return_value=2):
                        status = runner.launch(tasks)
            
            mock_run_debug.assert_called_once()
            self.assertEqual(len(status), 1)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    @patch('ais_bench.benchmark.runners.local.LocalRunner._run_normal')
    def test_launch_normal_mode(self, mock_run_normal, mock_logger_class):
        """测试正常模式下的launch方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=False
        )
        
        tasks = [{
            "work_dir": "/tmp/test",
            "cli_args": {"run_in_background": False}
        }]
        
        mock_run_normal.return_value = [("task1", 0)]
        
        with patch('ais_bench.benchmark.runners.local.multiprocessing.Process') as mock_process:
            mock_process_instance = MagicMock()
            mock_process.return_value = mock_process_instance
            
            with patch('ais_bench.benchmark.runners.local.task_abbr_from_cfg', return_value="task1"):
                with patch('ais_bench.benchmark.runners.local.is_npu_available', return_value=False):
                    with patch('ais_bench.benchmark.runners.local.torch.cuda.device_count', return_value=2):
                        status = runner.launch(tasks)
            
            mock_run_normal.assert_called_once()
            self.assertEqual(len(status), 1)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    @patch('ais_bench.benchmark.runners.local.os.environ')
    def test_launch_with_visible_devices(self, mock_environ, mock_logger_class):
        """测试使用CUDA_VISIBLE_DEVICES环境变量"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        mock_environ.__contains__ = lambda self, key: key == "CUDA_VISIBLE_DEVICES"
        mock_environ.get = lambda key, default: "0,1" if key == "CUDA_VISIBLE_DEVICES" else default
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=False
        )
        
        tasks = [{
            "work_dir": "/tmp/test",
            "cli_args": {"run_in_background": False}
        }]
        
        with patch('ais_bench.benchmark.runners.local.LocalRunner._run_normal') as mock_run_normal:
            mock_run_normal.return_value = [("task1", 0)]
            
            with patch('ais_bench.benchmark.runners.local.multiprocessing.Process') as mock_process:
                mock_process_instance = MagicMock()
                mock_process.return_value = mock_process_instance
                
                with patch('ais_bench.benchmark.runners.local.task_abbr_from_cfg', return_value="task1"):
                    with patch('ais_bench.benchmark.runners.local.is_npu_available', return_value=False):
                        with patch('ais_bench.benchmark.runners.local.torch.cuda.device_count', return_value=2):
                            status = runner.launch(tasks)
                            
                            # 验证从环境变量中读取了GPU IDs
                            mock_logger.debug.assert_called()
                            call_args = str(mock_logger.debug.call_args_list)
                            self.assertIn("Available devices", call_args)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    @patch('ais_bench.benchmark.runners.local.TASKS')
    def test_run_debug(self, mock_tasks, mock_logger_class):
        """测试_run_debug方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=True
        )
        
        tasks = [{
            "work_dir": "/tmp/test",
            "cli_args": {}
        }]
        
        # 创建模拟的task对象
        mock_task = MagicMock()
        mock_task.name = "test_task"
        mock_task.num_gpus = 0
        mock_task.get_command.return_value = "python test.py"
        mock_task.cfg = MagicMock()
        mock_task.cfg.dump = MagicMock()
        
        mock_tasks.build.return_value = mock_task
        
        all_gpu_ids = [0, 1]
        mock_monitor_p = MagicMock()
        
        with patch('ais_bench.benchmark.runners.local.subprocess.Popen') as mock_popen:
            mock_proc = MagicMock()
            mock_proc.wait.return_value = None
            mock_popen.return_value = mock_proc
            
            with patch('ais_bench.benchmark.runners.local.os.remove') as mock_remove:
                with patch('ais_bench.benchmark.runners.local.mmengine.mkdir_or_exist'):
                    with patch('uuid.uuid4', return_value=MagicMock(hex="test")):
                        with patch('builtins.open', create=True):
                            status = runner._run_debug(tasks, all_gpu_ids, mock_monitor_p)
                            
                            self.assertEqual(len(status), 1)
                            self.assertEqual(status[0][0], "test_task")
                            self.assertEqual(status[0][1], 0)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_launch_method(self, mock_logger_class):
        """测试_launch方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=False
        )
        
        # 创建模拟的task对象
        mock_task = MagicMock()
        mock_task.name = "test_task"
        mock_task.get_command.return_value = "python test.py"
        mock_task.get_log_path.return_value = "/tmp/test.out"
        mock_task.cfg = MagicMock()
        mock_task.cfg.dump = MagicMock()
        
        gpu_ids = [0]
        
        with patch('ais_bench.benchmark.runners.local.subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            
            with patch('ais_bench.benchmark.runners.local.os.remove') as mock_remove:
                with patch('ais_bench.benchmark.runners.local.mmengine.mkdir_or_exist'):
                    with patch('uuid.uuid4', return_value=MagicMock(hex="test")):
                        with patch('builtins.open', create=True):
                            task_name, exit_code = runner._launch(mock_task, gpu_ids, 0)
                            
                            self.assertEqual(task_name, "test_task")
                            self.assertEqual(exit_code, 0)
                            mock_run.assert_called_once()

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_launch_method_failure(self, mock_logger_class):
        """测试_launch方法处理任务失败"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=False
        )
        
        # 创建模拟的task对象
        mock_task = MagicMock()
        mock_task.name = "test_task"
        mock_task.get_command.return_value = "python test.py"
        mock_task.get_log_path.return_value = "/tmp/test.out"
        mock_task.cfg = MagicMock()
        mock_task.cfg.dump = MagicMock()
        
        gpu_ids = [0]
        
        with patch('ais_bench.benchmark.runners.local.subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_run.return_value = mock_result
            
            with patch('ais_bench.benchmark.runners.local.os.remove') as mock_remove:
                with patch('ais_bench.benchmark.runners.local.mmengine.mkdir_or_exist'):
                    with patch('uuid.uuid4', return_value=MagicMock(hex="test")):
                        with patch('builtins.open', create=True):
                            with patch('ais_bench.benchmark.runners.local.RUNNER_CODES') as mock_codes:
                                mock_error_code = MagicMock()
                                mock_codes.TASK_FAILED = mock_error_code
                                
                                task_name, exit_code = runner._launch(mock_task, gpu_ids, 0)
                                
                                self.assertEqual(task_name, "test_task")
                                self.assertEqual(exit_code, 1)
                                # 验证记录了错误日志
                                mock_logger.error.assert_called_once()

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    @patch('ais_bench.benchmark.runners.local.TASKS')
    def test_run_normal(self, mock_tasks, mock_logger_class):
        """测试_run_normal方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=False,
            max_num_workers=2
        )
        
        tasks = [{
            "work_dir": "/tmp/test",
            "cli_args": {}
        }]
        
        # 创建模拟的task对象
        mock_task = MagicMock()
        mock_task.name = "test_task"
        mock_task.num_gpus = 0
        mock_task.get_command.return_value = "python test.py"
        mock_task.get_log_path.return_value = "/tmp/test.out"
        mock_task.cfg = MagicMock()
        mock_task.cfg.dump = MagicMock()
        
        mock_tasks.build.return_value = mock_task
        
        all_gpu_ids = []
        mock_monitor_p = MagicMock()
        
        with patch('ais_bench.benchmark.runners.local.subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            
            with patch('ais_bench.benchmark.runners.local.os.remove') as mock_remove:
                with patch('ais_bench.benchmark.runners.local.mmengine.mkdir_or_exist'):
                    with patch('uuid.uuid4', return_value=MagicMock(hex="test")):
                        with patch('builtins.open', create=True):
                            with patch('ais_bench.benchmark.runners.local.ThreadPoolExecutor') as mock_executor:
                                mock_executor_instance = MagicMock()
                                mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
                                mock_executor_instance.__exit__ = MagicMock(return_value=False)
                                mock_executor_instance.map = MagicMock(return_value=[("test_task", 0)])
                                mock_executor.return_value = mock_executor_instance
                                
                                status = runner._run_normal(tasks, all_gpu_ids, mock_monitor_p)
                                
                                self.assertEqual(len(status), 1)
                                self.assertEqual(status[0][0], "test_task")
                                self.assertEqual(status[0][1], 0)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    @patch('ais_bench.benchmark.runners.local.TASKS')
    def test_run_debug_with_keyboard_interrupt(self, mock_tasks, mock_logger_class):
        """测试_run_debug方法处理KeyboardInterrupt"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=True
        )
        
        tasks = [{
            "work_dir": "/tmp/test",
            "cli_args": {}
        }]
        
        mock_task = MagicMock()
        mock_task.name = "test_task"
        mock_task.num_gpus = 0
        mock_task.get_command.return_value = "python test.py"
        mock_task.cfg = MagicMock()
        mock_task.cfg.dump = MagicMock()
        
        mock_tasks.build.return_value = mock_task
        
        all_gpu_ids = [0]
        mock_monitor_p = MagicMock()
        mock_monitor_p.join = MagicMock()  # 确保join方法不会抛出异常
        
        with patch('ais_bench.benchmark.runners.local.subprocess.Popen') as mock_popen:
            mock_proc = MagicMock()
            # 第一次调用wait()抛出KeyboardInterrupt，第二次调用返回None（正常完成）
            mock_proc.wait.side_effect = [KeyboardInterrupt(), None]
            mock_popen.return_value = mock_proc
            
            with patch('ais_bench.benchmark.runners.local.os.remove') as mock_remove:
                with patch('ais_bench.benchmark.runners.local.mmengine.mkdir_or_exist'):
                    with patch('uuid.uuid4', return_value=MagicMock(hex="test")):
                        with patch('builtins.open', create=True):
                            # 确保KeyboardInterrupt被捕获，不会传播到测试框架
                            status = runner._run_debug(tasks, all_gpu_ids, mock_monitor_p)
                            
                            # 验证记录了警告
                            mock_logger.warning.assert_called()
                            call_args = mock_logger.warning.call_args[0][0]
                            self.assertIn("interrupted", call_args)
                            # 验证join被调用
                            mock_monitor_p.join.assert_called_once()
                            # 验证wait被调用了两次（第一次抛出异常，第二次确保完成）
                            self.assertEqual(mock_proc.wait.call_count, 2)
                            # 验证状态正常返回
                            self.assertEqual(len(status), 1)
                            self.assertEqual(status[0][0], "test_task")

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_launch_with_keep_tmp_file(self, mock_logger_class):
        """测试keep_tmp_file为True时不删除临时文件"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=False,
            keep_tmp_file=True
        )
        
        mock_task = MagicMock()
        mock_task.name = "test_task"
        mock_task.get_command.return_value = "python test.py"
        mock_task.get_log_path.return_value = "/tmp/test.out"
        mock_task.cfg = MagicMock()
        mock_task.cfg.dump = MagicMock()
        
        gpu_ids = [0]
        
        with patch('ais_bench.benchmark.runners.local.subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            
            with patch('ais_bench.benchmark.runners.local.os.remove') as mock_remove:
                with patch('ais_bench.benchmark.runners.local.mmengine.mkdir_or_exist'):
                    with patch('uuid.uuid4', return_value=MagicMock(hex="test")):
                        with patch('builtins.open', create=True):
                            task_name, exit_code = runner._launch(mock_task, gpu_ids, 0)
                            
                            # 验证没有删除临时文件
                            mock_remove.assert_not_called()

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_launch_with_npu(self, mock_logger_class):
        """测试NPU设备"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=False
        )
        
        tasks = [{
            "work_dir": "/tmp/test",
            "cli_args": {"run_in_background": False}
        }]
        
        with patch('ais_bench.benchmark.runners.local.LocalRunner._run_normal') as mock_run_normal:
            mock_run_normal.return_value = [("task1", 0)]
            
            with patch('ais_bench.benchmark.runners.local.multiprocessing.Process') as mock_process:
                mock_process_instance = MagicMock()
                mock_process.return_value = mock_process_instance
                
                with patch('ais_bench.benchmark.runners.local.task_abbr_from_cfg', return_value="task1"):
                    with patch('ais_bench.benchmark.runners.local.is_npu_available', return_value=True):
                        mock_npu = MagicMock()
                        mock_npu.device_count.return_value = 2
                        with patch('ais_bench.benchmark.runners.local.torch.npu', mock_npu, create=True):
                            status = runner.launch(tasks)
                            
                            # 验证使用了NPU设备
                            mock_logger.debug.assert_called()
                            call_args = str(mock_logger.debug.call_args_list)
                            self.assertIn("Available devices", call_args)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    @patch('ais_bench.benchmark.runners.local.TASKS')
    def test_run_debug_with_keep_tmp_file(self, mock_tasks, mock_logger_class):
        """测试_run_debug中keep_tmp_file=True时不删除临时文件"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=True,
            keep_tmp_file=True
        )
        
        tasks = [{
            "work_dir": "/tmp/test",
            "cli_args": {}
        }]
        
        mock_task = MagicMock()
        mock_task.name = "test_task"
        mock_task.num_gpus = 0
        mock_task.get_command.return_value = "python test.py"
        mock_task.cfg = MagicMock()
        mock_task.cfg.dump = MagicMock()
        
        mock_tasks.build.return_value = mock_task
        
        all_gpu_ids = [0]
        mock_monitor_p = MagicMock()
        
        with patch('ais_bench.benchmark.runners.local.subprocess.Popen') as mock_popen:
            mock_proc = MagicMock()
            mock_proc.wait.return_value = None
            mock_popen.return_value = mock_proc
            
            with patch('ais_bench.benchmark.runners.local.os.remove') as mock_remove:
                with patch('ais_bench.benchmark.runners.local.mmengine.mkdir_or_exist'):
                    with patch('uuid.uuid4', return_value=MagicMock(hex="test")):
                        with patch('builtins.open', create=True):
                            status = runner._run_debug(tasks, all_gpu_ids, mock_monitor_p)
                            
                            # 验证没有删除临时文件
                            mock_remove.assert_not_called()
                            self.assertEqual(len(status), 1)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    @patch('ais_bench.benchmark.runners.local.TASKS')
    def test_run_debug_with_gpu_warning(self, mock_tasks, mock_logger_class):
        """测试_run_debug中GPU数量警告"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=True
        )
        
        tasks = [{
            "work_dir": "/tmp/test",
            "cli_args": {}
        }]
        
        mock_task = MagicMock()
        mock_task.name = "test_task"
        mock_task.num_gpus = 1  # 只需要1个GPU
        mock_task.get_command.return_value = "python test.py"
        mock_task.cfg = MagicMock()
        mock_task.cfg.dump = MagicMock()
        
        mock_tasks.build.return_value = mock_task
        
        all_gpu_ids = [0, 1, 2]  # 有3个GPU可用，但只需要1个
        mock_monitor_p = MagicMock()
        
        with patch('ais_bench.benchmark.runners.local.subprocess.Popen') as mock_popen:
            mock_proc = MagicMock()
            mock_proc.wait.return_value = None
            mock_popen.return_value = mock_proc
            
            with patch('ais_bench.benchmark.runners.local.os.remove') as mock_remove:
                with patch('ais_bench.benchmark.runners.local.mmengine.mkdir_or_exist'):
                    with patch('uuid.uuid4', return_value=MagicMock(hex="test")):
                        with patch('builtins.open', create=True):
                            status = runner._run_debug(tasks, all_gpu_ids, mock_monitor_p)
                            
                            # 验证警告被调用
                            mock_logger.warning.assert_called()
                            call_args = str(mock_logger.warning.call_args[0][0])
                            self.assertIn("Only use", call_args)
                            self.assertIn("GPUs", call_args)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    @patch('ais_bench.benchmark.runners.local.TASKS')
    def test_run_normal_with_gpu_pool(self, mock_tasks, mock_logger_class):
        """测试_run_normal中GPU资源池初始化"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=False,
            max_num_workers=2,
            max_workers_per_gpu=1
        )
        
        tasks = [{
            "work_dir": "/tmp/test",
            "cli_args": {}
        }]
        
        mock_task = MagicMock()
        mock_task.name = "test_task"
        mock_task.num_gpus = 0  # CPU任务，不需要GPU
        mock_task.get_command.return_value = "python test.py"
        mock_task.get_log_path.return_value = "/tmp/test.out"
        mock_task.cfg = MagicMock()
        mock_task.cfg.dump = MagicMock()
        
        mock_tasks.build.return_value = mock_task
        
        all_gpu_ids = [0, 1]  # 有GPU可用
        mock_monitor_p = MagicMock()
        
        with patch('ais_bench.benchmark.runners.local.subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            
            with patch('ais_bench.benchmark.runners.local.os.remove') as mock_remove:
                with patch('ais_bench.benchmark.runners.local.mmengine.mkdir_or_exist'):
                    with patch('uuid.uuid4', return_value=MagicMock(hex="test")):
                        with patch('builtins.open', create=True):
                            with patch('ais_bench.benchmark.runners.local.ThreadPoolExecutor') as mock_executor:
                                mock_executor_instance = MagicMock()
                                mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
                                mock_executor_instance.__exit__ = MagicMock(return_value=False)
                                mock_executor_instance.map = MagicMock(return_value=[("test_task", 0)])
                                mock_executor.return_value = mock_executor_instance
                                
                                status = runner._run_normal(tasks, all_gpu_ids, mock_monitor_p)
                                
                                # 验证GPU资源池初始化日志被调用
                                mock_logger.debug.assert_any_call(
                                    f"GPU resource pool initialized: {len(all_gpu_ids)} GPUs with {runner.max_workers_per_gpu} workers per GPU"
                                )
                                self.assertEqual(len(status), 1)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    @patch('ais_bench.benchmark.runners.local.TASKS')
    def test_run_normal_keyboard_interrupt(self, mock_tasks, mock_logger_class):
        """测试_run_normal中KeyboardInterrupt处理"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=False,
            max_num_workers=2
        )
        
        tasks = [{
            "work_dir": "/tmp/test",
            "cli_args": {}
        }]
        
        all_gpu_ids = []
        mock_monitor_p = MagicMock()
        mock_monitor_p.join = MagicMock()
        
        with patch('ais_bench.benchmark.runners.local.ThreadPoolExecutor') as mock_executor:
            mock_executor_instance = MagicMock()
            mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
            mock_executor_instance.__exit__ = MagicMock(return_value=False)
            # 模拟KeyboardInterrupt
            mock_executor_instance.map = MagicMock(side_effect=KeyboardInterrupt())
            mock_executor.return_value = mock_executor_instance
            
            status = runner._run_normal(tasks, all_gpu_ids, mock_monitor_p)
            
            # 验证join被调用
            mock_monitor_p.join.assert_called_once()
            # 验证警告被调用
            mock_logger.warning.assert_called_with("Main process interrupted by user! Waiting for running tasks to complete...")
            # 验证返回空状态列表
            self.assertEqual(status, [])

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    @patch('ais_bench.benchmark.runners.local.TASKS')
    def test_run_normal_with_submit_function(self, mock_tasks, mock_logger_class):
        """测试_run_normal中的submit函数逻辑"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=False,
            max_num_workers=1,
            max_workers_per_gpu=1
        )
        
        tasks = [{
            "work_dir": "/tmp/test",
            "cli_args": {}
        }]
        
        mock_task = MagicMock()
        mock_task.name = "test_task"
        mock_task.num_gpus = 1  # 需要1个GPU
        mock_task.get_command.return_value = "python test.py"
        mock_task.get_log_path.return_value = "/tmp/test.out"
        mock_task.cfg = MagicMock()
        mock_task.cfg.dump = MagicMock()
        
        mock_tasks.build.return_value = mock_task
        
        all_gpu_ids = [0]  # 有1个GPU可用
        mock_monitor_p = MagicMock()
        
        with patch('ais_bench.benchmark.runners.local.subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result
            
            with patch('ais_bench.benchmark.runners.local.os.remove') as mock_remove:
                with patch('ais_bench.benchmark.runners.local.mmengine.mkdir_or_exist'):
                    with patch('uuid.uuid4', return_value=MagicMock(hex="test")):
                        with patch('builtins.open', create=True):
                            with patch('ais_bench.benchmark.runners.local.time.sleep') as mock_sleep:
                                status = runner._run_normal(tasks, all_gpu_ids, mock_monitor_p)
                                
                                # 验证submit函数被调用（通过executor.map）
                                self.assertEqual(len(status), 1)
                                self.assertEqual(status[0][0], "test_task")
                                # 验证GPU资源被分配和释放
                                mock_run.assert_called_once()

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_launch_with_monitor_process(self, mock_logger_class):
        """测试launch方法中的monitor_process函数"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        runner = LocalRunner(
            task=self.task_cfg,
            debug=False
        )
        
        tasks = [{
            "work_dir": "/tmp/test",
            "cli_args": {"run_in_background": True}
        }]
        
        with patch('ais_bench.benchmark.runners.local.multiprocessing.Process') as mock_process:
            mock_process_instance = MagicMock()
            mock_process_instance.pid = 12345
            mock_process.return_value = mock_process_instance
            
            with patch('ais_bench.benchmark.runners.local.task_abbr_from_cfg', return_value="task1"):
                with patch('ais_bench.benchmark.runners.local.LocalRunner._run_normal') as mock_run_normal:
                    mock_run_normal.return_value = [("task1", 0)]
                    
                    with patch('ais_bench.benchmark.runners.local.is_npu_available', return_value=False):
                        with patch('ais_bench.benchmark.runners.local.torch.cuda.device_count', return_value=2):
                            with patch('ais_bench.benchmark.runners.local.TasksMonitor') as mock_monitor:
                                # Mock monitor_process函数中的TasksMonitor
                                mock_monitor_instance = MagicMock()
                                mock_monitor.return_value = mock_monitor_instance
                                
                                status = runner.launch(tasks)
                                
                                # 验证process被创建和启动
                                mock_process.assert_called_once()
                                mock_process_instance.start.assert_called_once()
                                mock_process_instance.join.assert_called_once()
                                self.assertEqual(len(status), 1)


if __name__ == '__main__':
    unittest.main()

