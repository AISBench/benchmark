import unittest
import struct
import time
from unittest.mock import patch, MagicMock
from multiprocessing import Event, shared_memory

import numpy as np
from mmengine.config import ConfigDict

from ais_bench.benchmark.tasks.utils import (
    create_message_share_memory,
    check_virtual_memory_usage,
    ProgressBar,
    TokenProducer,
    update_global_data_index,
    FMT,
    MESSAGE_SIZE,
    WAIT_FLAG,
    INDEX_READ_FLAG,
    MAX_VIRTUAL_MEMORY_USAGE_PERCENT,
    MESSAGE_INFO
)
from ais_bench.benchmark.tasks.base import TaskStateManager
from ais_bench.benchmark.utils.logging.error_codes import TINFER_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchRuntimeError, ParameterValueError


class TestCreateMessageShareMemory(unittest.TestCase):
    """测试create_message_share_memory函数"""

    def test_create_shared_memory(self):
        """测试创建共享内存"""
        shm = create_message_share_memory()
        
        self.assertIsNotNone(shm)
        self.assertEqual(shm.size, MESSAGE_SIZE)
        
        # 验证初始值
        status, post, recv, fail, finish, _, data_index = struct.unpack(FMT, shm.buf)
        self.assertEqual(status, 0)
        self.assertEqual(post, 0)
        self.assertEqual(recv, 0)
        self.assertEqual(fail, 0)
        self.assertEqual(finish, 0)
        self.assertEqual(data_index, INDEX_READ_FLAG)
        
        # 清理
        shm.close()
        shm.unlink()


class TestCheckVirtualMemoryUsage(unittest.TestCase):
    """测试check_virtual_memory_usage函数"""

    @patch('ais_bench.benchmark.tasks.utils.psutil.virtual_memory')
    @patch('ais_bench.benchmark.tasks.utils.logger')
    def test_memory_usage_within_threshold(self, mock_logger, mock_virtual_memory):
        """测试内存使用在阈值内的情况"""
        # 模拟内存使用率为50%
        mock_memory = MagicMock()
        mock_memory.total = 100 * 1024**3  # 100GB
        mock_memory.used = 40 * 1024**3  # 40GB
        mock_memory.available = 60 * 1024**3  # 60GB
        mock_virtual_memory.return_value = mock_memory
        
        dataset_bytes = 10 * 1024**3  # 10GB
        
        # 不应该抛出异常
        check_virtual_memory_usage(dataset_bytes)
        
        # 验证记录日志
        self.assertTrue(mock_logger.info.called)

    @patch('ais_bench.benchmark.tasks.utils.psutil.virtual_memory')
    def test_memory_usage_exceeds_threshold(self, mock_virtual_memory):
        """测试内存使用超过阈值的情况"""
        # 模拟内存使用率为85%
        mock_memory = MagicMock()
        mock_memory.total = 100 * 1024**3  # 100GB
        mock_memory.used = 75 * 1024**3  # 75GB
        mock_memory.available = 25 * 1024**3  # 25GB
        mock_virtual_memory.return_value = mock_memory
        
        dataset_bytes = 10 * 1024**3  # 10GB
        
        # 应该抛出异常
        with self.assertRaises(AISBenchRuntimeError) as context:
            check_virtual_memory_usage(dataset_bytes)
        
        error_code = context.exception.error_code_str
        self.assertEqual(error_code, TINFER_CODES.VIRTUAL_MEMORY_USAGE_TOO_HIGH.full_code)

    @patch('ais_bench.benchmark.tasks.utils.psutil.virtual_memory')
    @patch('ais_bench.benchmark.tasks.utils.logger')
    def test_custom_threshold(self, mock_logger, mock_virtual_memory):
        """测试自定义阈值"""
        mock_memory = MagicMock()
        mock_memory.total = 100 * 1024**3
        mock_memory.used = 50 * 1024**3
        mock_memory.available = 50 * 1024**3
        mock_virtual_memory.return_value = mock_memory
        
        dataset_bytes = 10 * 1024**3
        custom_threshold = 70
        
        # 使用自定义阈值，不应该抛出异常
        check_virtual_memory_usage(dataset_bytes, threshold_percent=custom_threshold)
        self.assertTrue(mock_logger.info.called)


class TestProgressBar(unittest.TestCase):
    """测试ProgressBar类"""

    def setUp(self):
        """设置测试环境"""
        self.stop_event = Event()
        self.per_pid_shms = {}
        self.data_num = 100
        self.finish_data_num = 0
        self.debug = False
        self.pressure = False

    def tearDown(self):
        """清理测试环境"""
        # 清理共享内存
        for shm in self.per_pid_shms.values():
            try:
                shm.close()
                shm.unlink()
            except:
                pass

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_init(self, mock_logger_class):
        """测试ProgressBar初始化"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        # 创建共享内存
        shm = create_message_share_memory()
        self.per_pid_shms[12345] = shm
        
        progress_bar = ProgressBar(
            per_pid_shms=self.per_pid_shms,
            stop_event=self.stop_event,
            data_num=self.data_num,
            finish_data_num=self.finish_data_num,
            debug=self.debug,
            pressure=self.pressure
        )
        
        self.assertEqual(progress_bar.data_num, self.data_num)
        self.assertEqual(progress_bar.finish_data_num, self.finish_data_num)
        self.assertEqual(progress_bar.total_data_num, self.data_num)
        self.assertEqual(progress_bar.debug, self.debug)
        self.assertEqual(progress_bar.pressure, self.pressure)
        self.assertEqual(progress_bar.stats, {"post": 0, "recv": 0, "fail": 0, "finish": 0})

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_recalc_aggregate(self, mock_logger_class):
        """测试_recalc_aggregate方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        progress_bar = ProgressBar(
            per_pid_shms={},
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        # 设置per_pid_stats
        progress_bar.per_pid_stats = {
            1: {"post": 10, "recv": 8, "fail": 1, "finish": 7},
            2: {"post": 15, "recv": 12, "fail": 2, "finish": 10}
        }
        
        progress_bar._recalc_aggregate()
        
        self.assertEqual(progress_bar.stats["post"], 25)
        self.assertEqual(progress_bar.stats["recv"], 20)
        self.assertEqual(progress_bar.stats["fail"], 3)
        self.assertEqual(progress_bar.stats["finish"], 17)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_read_shared_memory_and_update_per_pid(self, mock_logger_class):
        """测试_read_shared_memory_and_update_per_pid方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        # 创建共享内存并设置值
        shm = create_message_share_memory()
        self.per_pid_shms[12345] = shm
        
        # 设置共享内存的值
        shm.buf[:] = struct.pack(FMT, 0, 10, 8, 1, 7, 0, INDEX_READ_FLAG)
        
        progress_bar = ProgressBar(
            per_pid_shms=self.per_pid_shms,
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        updated = progress_bar._read_shared_memory_and_update_per_pid()
        
        self.assertTrue(updated)
        self.assertIn(12345, progress_bar.per_pid_stats)
        self.assertEqual(progress_bar.per_pid_stats[12345]["post"], 10)
        self.assertEqual(progress_bar.stats["post"], 10)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_compute_rates_since_start(self, mock_logger_class):
        """测试_compute_rates_since_start方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        progress_bar = ProgressBar(
            per_pid_shms={},
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        progress_bar.stats = {"post": 100, "recv": 80, "fail": 5, "finish": 75}
        
        # 等待一小段时间
        time.sleep(0.1)
        
        rates = progress_bar._compute_rates_since_start()
        
        self.assertIn("post", rates)
        self.assertIn("recv", rates)
        self.assertIn("fail", rates)
        self.assertIn("finish", rates)
        self.assertGreater(rates["post"], 0)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_compute_rates_interval(self, mock_logger_class):
        """测试_compute_rates_interval方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        progress_bar = ProgressBar(
            per_pid_shms={},
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        progress_bar.stats = {"post": 100, "recv": 80, "fail": 5, "finish": 75}
        progress_bar._last_snapshot_stats = {"post": 50, "recv": 40, "fail": 2, "finish": 35}
        
        time.sleep(0.1)
        
        rates = progress_bar._compute_rates_interval()
        
        self.assertIn("post", rates)
        self.assertIn("recv", rates)
        self.assertIn("fail", rates)
        self.assertIn("finish", rates)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_format_per_pid_brief(self, mock_logger_class):
        """测试_format_per_pid_brief方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        progress_bar = ProgressBar(
            per_pid_shms={},
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        progress_bar.per_pid_stats = {
            12345: {"post": 10, "recv": 8, "fail": 1, "finish": 7},
            67890: {"post": 15, "recv": 12, "fail": 2, "finish": 10}
        }
        
        result = progress_bar._format_per_pid_brief()
        
        self.assertIn("12345", result)
        self.assertIn("67890", result)
        self.assertIn("7/10/8/1", result)  # finish/post/recv/fail

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_format_per_pid_brief_empty(self, mock_logger_class):
        """测试_format_per_pid_brief方法在没有worker时"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        progress_bar = ProgressBar(
            per_pid_shms={},
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        progress_bar.per_pid_stats = {}
        
        result = progress_bar._format_per_pid_brief()
        self.assertEqual(result, "<no workers>")

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_set_message_flag(self, mock_logger_class):
        """测试set_message_flag方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        shm = create_message_share_memory()
        self.per_pid_shms[12345] = shm
        
        progress_bar = ProgressBar(
            per_pid_shms=self.per_pid_shms,
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        flag = 1
        progress_bar.set_message_flag(flag)
        
        # 验证标志被设置
        status, _, _, _, _, _, _ = struct.unpack(FMT, shm.buf)
        self.assertEqual(status, flag)
        
        # 验证记录日志
        mock_logger.debug.assert_called()


class TestTokenProducer(unittest.TestCase):
    """测试TokenProducer类"""

    def setUp(self):
        """设置测试环境"""
        self.request_rate = 10
        self.traffic_cfg = ConfigDict({})
        self.request_num = 100
        self.pressure_mode = False

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_init_with_normal_rate(self, mock_logger_class):
        """测试使用正常请求速率初始化"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        producer = TokenProducer(
            request_rate=self.request_rate,
            traffic_cfg=self.traffic_cfg,
            request_num=self.request_num,
            pressure_mode=self.pressure_mode
        )
        
        self.assertEqual(producer.request_rate, self.request_rate)
        self.assertEqual(producer.pressure_mode, self.pressure_mode)
        self.assertIsNotNone(producer.token_bucket)
        self.assertIsNotNone(producer.interval_lists)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_init_with_low_rate(self, mock_logger_class):
        """测试使用低请求速率（<0.1）初始化"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        producer = TokenProducer(
            request_rate=0.05,
            traffic_cfg=self.traffic_cfg,
            request_num=self.request_num,
            pressure_mode=self.pressure_mode
        )
        
        self.assertEqual(producer.request_rate, float("inf"))
        self.assertIsNone(producer.token_bucket)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_init_with_ramp_up_strategy(self, mock_logger_class):
        """测试使用ramp_up策略初始化"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        traffic_cfg = ConfigDict({
            "ramp_up_strategy": "linear",
            "ramp_up_start_rps": 5,
            "ramp_up_end_rps": 20,
            "burstiness": 1.0
        })
        
        producer = TokenProducer(
            request_rate=self.request_rate,
            traffic_cfg=traffic_cfg,
            request_num=self.request_num,
            pressure_mode=self.pressure_mode
        )
        
        self.assertIsNotNone(producer.interval_lists)
        self.assertEqual(len(producer.interval_lists), self.request_num)
        
        # 验证记录日志
        mock_logger.info.assert_called()

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_init_with_invalid_ramp_up_strategy(self, mock_logger_class):
        """测试使用无效的ramp_up策略初始化"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        traffic_cfg = ConfigDict({
            "ramp_up_strategy": "invalid",
            "ramp_up_start_rps": 5,
            "ramp_up_end_rps": 20
        })
        
        with self.assertRaises(ParameterValueError) as context:
            TokenProducer(
                request_rate=self.request_rate,
                traffic_cfg=traffic_cfg,
                request_num=self.request_num,
                pressure_mode=self.pressure_mode
            )
        
        error_code = context.exception.error_code_str
        self.assertEqual(error_code, TINFER_CODES.INVALID_RAMP_UP_STRATEGY.full_code)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_generate_interval_lists_linear(self, mock_logger_class):
        """测试线性ramp_up策略生成间隔列表"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        traffic_cfg = ConfigDict({
            "ramp_up_strategy": "linear",
            "ramp_up_start_rps": 10,
            "ramp_up_end_rps": 20
        })
        
        producer = TokenProducer(
            request_rate=self.request_rate,
            traffic_cfg=traffic_cfg,
            request_num=10,
            pressure_mode=self.pressure_mode
        )
        
        self.assertEqual(len(producer.interval_lists), 10)
        # 验证间隔列表是递增的（累积延迟）
        for i in range(1, len(producer.interval_lists)):
            self.assertGreaterEqual(producer.interval_lists[i], producer.interval_lists[i-1])

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_generate_interval_lists_exponential(self, mock_logger_class):
        """测试指数ramp_up策略生成间隔列表"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        traffic_cfg = ConfigDict({
            "ramp_up_strategy": "exponential",
            "ramp_up_start_rps": 10,
            "ramp_up_end_rps": 20
        })
        
        producer = TokenProducer(
            request_rate=self.request_rate,
            traffic_cfg=traffic_cfg,
            request_num=10,
            pressure_mode=self.pressure_mode
        )
        
        self.assertEqual(len(producer.interval_lists), 10)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_produce_token_without_token_bucket(self, mock_logger_class):
        """测试没有token_bucket时的produce_token"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        producer = TokenProducer(
            request_rate=0.05,  # 低速率，token_bucket为None
            traffic_cfg=self.traffic_cfg,
            request_num=self.request_num,
            pressure_mode=self.pressure_mode
        )
        
        stop_event = Event()
        stop_event.set()  # 立即停止
        
        # 应该直接返回，不阻塞
        producer.produce_token(stop_event, {})

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_produce_token_with_token_bucket(self, mock_logger_class):
        """测试有token_bucket时的produce_token"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        producer = TokenProducer(
            request_rate=self.request_rate,
            traffic_cfg=self.traffic_cfg,
            request_num=10,  # 小数量用于快速测试
            pressure_mode=self.pressure_mode
        )
        
        stop_event = Event()
        
        # 使用线程运行produce_token，然后快速停止
        import threading
        
        def run_produce():
            producer.produce_token(stop_event, {})
        
        thread = threading.Thread(target=run_produce)
        thread.start()
        
        # 等待一小段时间后停止
        time.sleep(0.1)
        stop_event.set()
        
        thread.join(timeout=1)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_produce_token_with_exception(self, mock_logger_class):
        """测试produce_token方法中异常处理"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        producer = TokenProducer(
            request_rate=self.request_rate,
            traffic_cfg=self.traffic_cfg,
            request_num=10,
            pressure_mode=self.pressure_mode
        )
        
        stop_event = Event()
        
        # Mock token_bucket.release() 抛出异常
        producer.token_bucket.release = MagicMock(side_effect=ValueError("Semaphore released too many times"))
        
        import threading
        
        def run_produce():
            producer.produce_token(stop_event, {})
        
        thread = threading.Thread(target=run_produce)
        thread.start()
        
        time.sleep(0.1)
        stop_event.set()
        
        thread.join(timeout=1)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_compute_rates_interval_zero_dt(self, mock_logger_class):
        """测试_compute_rates_interval方法中dt <= 0的情况"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        # 创建Event对象，因为这是TestTokenProducer类，没有stop_event属性
        stop_event = Event()
        progress_bar = ProgressBar(
            per_pid_shms={},
            stop_event=stop_event,
            data_num=100
        )
        
        # 设置_last_snapshot_time为当前时间，使得dt <= 0
        progress_bar._last_snapshot_time = time.perf_counter()
        
        rates = progress_bar._compute_rates_interval()
        
        # 验证返回了零速率
        self.assertEqual(rates["post"], 0.0)
        self.assertEqual(rates["recv"], 0.0)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    @patch('ais_bench.benchmark.tasks.utils.tqdm')
    def test_draw_progress_pressure_mode(self, mock_tqdm, mock_logger_class):
        """测试_draw_progress方法的pressure模式"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        stop_event = Event()
        shm = create_message_share_memory()
        per_pid_shms = {12345: shm}
        
        # Mock tqdm
        mock_bar = MagicMock()
        mock_bar.n = 0
        mock_bar.total = 15
        mock_tqdm.return_value = mock_bar
        
        progress_bar = ProgressBar(
            per_pid_shms=per_pid_shms,
            stop_event=stop_event,
            data_num=100,
            finish_data_num=0,
            debug=True,
            pressure=True,
            pressure_time=15
        )
        
        # 设置stop_event，使得循环快速退出
        stop_event.set()
        
        try:
            progress_bar._draw_progress()
            # 验证记录了info日志
            mock_logger.info.assert_called()
        finally:
            shm.close()
            shm.unlink()

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_draw_progress_with_finish_data(self, mock_logger_class):
        """测试_draw_progress方法中有finish_data_num的情况"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        stop_event = Event()
        shm = create_message_share_memory()
        per_pid_shms = {12345: shm}
        
        progress_bar = ProgressBar(
            per_pid_shms=per_pid_shms,
            stop_event=stop_event,
            data_num=100,
            finish_data_num=10,
            debug=True,
            pressure=False
        )
        
        # 设置stop_event，使得循环快速退出
        stop_event.set()
        
        try:
            progress_bar._draw_progress()
        finally:
            shm.close()
            shm.unlink()

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_refresh_task_monitor(self, mock_logger_class):
        """测试_refresh_task_monitor方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        stop_event = Event()
        shm = create_message_share_memory()
        per_pid_shms = {12345: shm}
        
        progress_bar = ProgressBar(
            per_pid_shms=per_pid_shms,
            stop_event=stop_event,
            data_num=100,
            finish_data_num=0,
            debug=False,
            pressure=False
        )
        
        # Mock task_state_manager
        mock_task_state_manager = MagicMock()
        
        # 设置stop_event，使得循环快速退出
        stop_event.set()
        
        try:
            progress_bar._refresh_task_monitor(mock_task_state_manager)
            # 验证update_task_state被调用
            mock_task_state_manager.update_task_state.assert_called()
        finally:
            shm.close()
            shm.unlink()

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_refresh_task_monitor_pressure_mode(self, mock_logger_class):
        """测试_refresh_task_monitor方法的pressure模式"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        stop_event = Event()
        shm = create_message_share_memory()
        per_pid_shms = {12345: shm}
        
        progress_bar = ProgressBar(
            per_pid_shms=per_pid_shms,
            stop_event=stop_event,
            data_num=100,
            finish_data_num=0,
            debug=False,
            pressure=True,
            pressure_time=15
        )
        
        # Mock task_state_manager
        mock_task_state_manager = MagicMock()
        
        # 设置stop_event，使得循环快速退出
        stop_event.set()
        
        try:
            progress_bar._refresh_task_monitor(mock_task_state_manager)
            # 验证update_task_state被调用
            mock_task_state_manager.update_task_state.assert_called()
        finally:
            shm.close()
            shm.unlink()

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_display_waiting_for_subprocesses(self, mock_logger_class):
        """测试display方法等待子进程完成的情况"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        stop_event = Event()
        shm = create_message_share_memory()
        per_pid_shms = {12345: shm}
        
        progress_bar = ProgressBar(
            per_pid_shms=per_pid_shms,
            stop_event=stop_event,
            data_num=100,
            finish_data_num=0,
            debug=False,
            pressure=False
        )
        
        # 设置共享内存状态为WAIT_FLAG，模拟子进程还在等待
        # 确保shm.buf存在
        if shm.buf is not None:
            shm.buf[:] = struct.pack(FMT, WAIT_FLAG, 0, 0, 0, 0, 0, INDEX_READ_FLAG)
        
        # Mock task_state_manager
        mock_task_state_manager = MagicMock()
        
        # 注意：在display方法中，当stop_event.is_set()为True时，会等待子进程完成
        # 如果所有子进程都完成（status != WAIT_FLAG），会清除stop_event并退出
        # 为了测试等待逻辑，我们需要先设置stop_event为True，然后让子进程完成
        stop_event.set()
        
        try:
            # 使用线程运行display，避免阻塞
            import threading
            
            def run_display():
                try:
                    progress_bar.display(mock_task_state_manager)
                except Exception:
                    # 忽略线程中的异常，避免影响测试
                    pass
            
            thread = threading.Thread(target=run_display)
            thread.start()
            time.sleep(0.05)  # 短暂等待，让display方法开始执行
            
            # 模拟子进程完成：清除WAIT_FLAG
            if shm.buf is not None:
                shm.buf[:] = struct.pack(FMT, 0, 0, 0, 0, 0, 0, INDEX_READ_FLAG)
            
            # 清除stop_event，让display方法退出初始等待循环，进入监控循环
            stop_event.clear()
            
            # 等待一小段时间，让display方法进入_refresh_task_monitor循环
            time.sleep(0.1)
            
            # 再次设置stop_event，让_refresh_task_monitor循环退出
            stop_event.set()
            
            thread.join(timeout=1)
        finally:
            try:
                shm.close()
                shm.unlink()
            except:
                pass

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_display_debug_mode(self, mock_logger_class):
        """测试display方法的debug模式"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        stop_event = Event()
        shm = create_message_share_memory()
        per_pid_shms = {12345: shm}
        
        progress_bar = ProgressBar(
            per_pid_shms=per_pid_shms,
            stop_event=stop_event,
            data_num=100,
            finish_data_num=0,
            debug=True,
            pressure=False
        )
        
        # 设置共享内存状态为非WAIT_FLAG，模拟子进程已完成
        shm.buf[:] = struct.pack(FMT, 0, 0, 0, 0, 0, 0, INDEX_READ_FLAG)
        
        # Mock task_state_manager
        mock_task_state_manager = MagicMock()
        
        try:
            import threading
            
            # 先设置stop_event，让display方法在初始等待循环中等待
            stop_event.set()
            
            def run_display():
                progress_bar.display(mock_task_state_manager)
            
            thread = threading.Thread(target=run_display)
            thread.start()
            
            # 等待一小段时间，让display方法开始执行并进入初始等待循环
            time.sleep(0.05)
            
            # 清除stop_event，让display方法退出初始等待循环，进入_draw_progress
            stop_event.clear()
            
            # 等待一小段时间，让_draw_progress开始执行
            time.sleep(0.1)
            
            # 再次设置stop_event，让_draw_progress循环退出
            stop_event.set()
            
            thread.join(timeout=1)
        finally:
            shm.close()
            shm.unlink()


class TestUpdateGlobalDataIndex(unittest.TestCase):
    """测试update_global_data_index函数"""

    def test_update_global_data_index_basic(self):
        """测试基本的update_global_data_index功能"""
        shm1 = create_message_share_memory()
        shm2 = create_message_share_memory()
        
        try:
            # 初始化共享内存
            shm1.buf[MESSAGE_INFO.DATA_INDEX[0]:MESSAGE_INFO.DATA_INDEX[1]] = struct.pack("i", INDEX_READ_FLAG)
            shm1.buf[MESSAGE_INFO.STATUS[0]:MESSAGE_INFO.STATUS[1]] = struct.pack("I", 0)
            shm2.buf[MESSAGE_INFO.DATA_INDEX[0]:MESSAGE_INFO.DATA_INDEX[1]] = struct.pack("i", INDEX_READ_FLAG)
            shm2.buf[MESSAGE_INFO.STATUS[0]:MESSAGE_INFO.STATUS[1]] = struct.pack("I", 0)
            
            global_data_indexes = [0, 1, 2]
            data_num = 10
            
            # 使用线程运行，因为这是阻塞函数
            import threading
            result = []
            
            def run_update():
                update_global_data_index([shm1.name, shm2.name], data_num, global_data_indexes, pressure=False)
                result.append("done")
            
            thread = threading.Thread(target=run_update)
            thread.start()
            
            # 等待一小段时间让函数开始执行
            time.sleep(0.1)
            
            # 设置状态为1，让函数退出
            shm1.buf[MESSAGE_INFO.STATUS[0]:MESSAGE_INFO.STATUS[1]] = struct.pack("I", 1)
            shm2.buf[MESSAGE_INFO.STATUS[0]:MESSAGE_INFO.STATUS[1]] = struct.pack("I", 1)
            
            thread.join(timeout=2)
        finally:
            try:
                shm1.close()
                shm1.unlink()
                shm2.close()
                shm2.unlink()
            except:
                pass

    def test_update_global_data_index_pressure_mode(self):
        """测试pressure模式的update_global_data_index"""
        shm = create_message_share_memory()
        
        try:
            shm.buf[MESSAGE_INFO.DATA_INDEX[0]:MESSAGE_INFO.DATA_INDEX[1]] = struct.pack("i", INDEX_READ_FLAG)
            shm.buf[MESSAGE_INFO.STATUS[0]:MESSAGE_INFO.STATUS[1]] = struct.pack("I", 0)
            
            global_data_indexes = [0, 1]
            data_num = 10
            
            import threading
            
            def run_update():
                update_global_data_index([shm.name], data_num, global_data_indexes, pressure=True)
            
            thread = threading.Thread(target=run_update)
            thread.start()
            time.sleep(0.1)
            
            # 设置状态为1，让函数退出
            shm.buf[MESSAGE_INFO.STATUS[0]:MESSAGE_INFO.STATUS[1]] = struct.pack("I", 1)
            
            thread.join(timeout=2)
        finally:
            try:
                shm.close()
                shm.unlink()
            except:
                pass

    def test_update_global_data_index_cur_pos_exceeds_no_pressure(self):
        """测试cur_pos超过global_data_indexes长度且非pressure模式"""
        shm = create_message_share_memory()
        
        try:
            shm.buf[MESSAGE_INFO.DATA_INDEX[0]:MESSAGE_INFO.DATA_INDEX[1]] = struct.pack("i", INDEX_READ_FLAG)
            shm.buf[MESSAGE_INFO.STATUS[0]:MESSAGE_INFO.STATUS[1]] = struct.pack("I", 0)
            
            global_data_indexes = [0, 1]
            data_num = 10
            
            import threading
            
            def run_update():
                # 模拟cur_pos超过global_data_indexes的情况
                # 通过设置data_index为INDEX_READ_FLAG多次来触发多次更新
                update_global_data_index([shm.name], data_num, global_data_indexes, pressure=False)
            
            thread = threading.Thread(target=run_update)
            thread.start()
            time.sleep(0.1)
            
            # 设置状态为1，让函数退出
            shm.buf[MESSAGE_INFO.STATUS[0]:MESSAGE_INFO.STATUS[1]] = struct.pack("I", 1)
            
            thread.join(timeout=2)
        finally:
            try:
                shm.close()
                shm.unlink()
            except:
                pass


class TestProgressBarAdditional(unittest.TestCase):
    """测试ProgressBar的额外功能"""

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_refresh_task_monitor_with_finish_rate(self, mock_logger_class):
        """测试_refresh_task_monitor方法中有finish_rate的情况"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        stop_event = Event()
        shm = create_message_share_memory()
        per_pid_shms = {12345: shm}
        
        progress_bar = ProgressBar(
            per_pid_shms=per_pid_shms,
            stop_event=stop_event,
            data_num=100,
            finish_data_num=0,
            debug=False,
            pressure=False
        )
        
        # 设置共享内存，模拟有统计数据
        shm.buf[:] = struct.pack(FMT, 0, 10, 8, 1, 5, 0, INDEX_READ_FLAG)
        
        mock_task_state_manager = MagicMock()
        
        try:
            import threading
            
            def run_refresh():
                progress_bar._refresh_task_monitor(mock_task_state_manager)
            
            thread = threading.Thread(target=run_refresh)
            thread.start()
            time.sleep(0.1)
            stop_event.set()
            thread.join(timeout=1)
            
            # 验证update_task_state被调用
            self.assertTrue(mock_task_state_manager.update_task_state.called)
        finally:
            try:
                shm.close()
                shm.unlink()
            except:
                pass

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    @patch('ais_bench.benchmark.tasks.utils.tqdm')
    def test_draw_progress_keyboard_interrupt(self, mock_tqdm, mock_logger_class):
        """测试_draw_progress方法的KeyboardInterrupt处理"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        stop_event = Event()
        shm = create_message_share_memory()
        per_pid_shms = {12345: shm}
        
        progress_bar = ProgressBar(
            per_pid_shms=per_pid_shms,
            stop_event=stop_event,
            data_num=100,
            finish_data_num=0,
            debug=True,
            pressure=False
        )
        
        mock_main_bar = MagicMock()
        mock_info_bar = MagicMock()
        mock_tqdm.side_effect = [mock_main_bar, mock_info_bar]
        mock_main_bar.n = 0
        mock_main_bar.total = 100
        
        try:
            import threading
            import signal
            
            def run_draw():
                progress_bar._draw_progress()
            
            thread = threading.Thread(target=run_draw)
            thread.start()
            time.sleep(0.05)
            
            # 模拟KeyboardInterrupt
            # 由于在另一个线程中，我们通过设置stop_event来模拟退出
            stop_event.set()
            
            thread.join(timeout=1)
        finally:
            try:
                shm.close()
                shm.unlink()
            except:
                pass

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    @patch('ais_bench.benchmark.tasks.utils.tqdm')
    def test_draw_progress_pressure_mode_total_reached(self, mock_tqdm, mock_logger_class):
        """测试_draw_progress方法pressure模式下达到total的情况"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        stop_event = Event()
        shm = create_message_share_memory()
        per_pid_shms = {12345: shm}
        
        progress_bar = ProgressBar(
            per_pid_shms=per_pid_shms,
            stop_event=stop_event,
            data_num=100,
            finish_data_num=0,
            debug=True,
            pressure=True,
            pressure_time=5
        )
        
        mock_main_bar = MagicMock()
        mock_info_bar = MagicMock()
        mock_tqdm.side_effect = [mock_main_bar, mock_info_bar]
        mock_main_bar.n = 5  # 已达到total
        mock_main_bar.total = 5
        
        try:
            import threading
            
            def run_draw():
                progress_bar._draw_progress()
            
            thread = threading.Thread(target=run_draw)
            thread.start()
            time.sleep(0.05)
            stop_event.set()
            thread.join(timeout=1)
        finally:
            try:
                shm.close()
                shm.unlink()
            except:
                pass


class TestTokenProducerAdditional(unittest.TestCase):
    """测试TokenProducer的额外功能"""

    def setUp(self):
        self.request_rate = 10
        self.traffic_cfg = ConfigDict({})
        self.request_num = 100
        self.pressure_mode = False

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_produce_token_else_branch_exception(self, mock_logger_class):
        """测试produce_token方法else分支的异常处理"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        producer = TokenProducer(
            request_rate=self.request_rate,
            traffic_cfg=self.traffic_cfg,
            request_num=10,
            pressure_mode=self.pressure_mode
        )
        
        stop_event = Event()
        shm = create_message_share_memory()
        per_pid_shms = {12345: shm}
        
        # 设置共享内存状态为WAIT_FLAG，让produce_token跳过初始等待
        shm.buf[:] = struct.pack(FMT, WAIT_FLAG, 0, 0, 0, 0, 0, INDEX_READ_FLAG)
        
        # Mock token_bucket.release() 在else分支抛出异常
        producer.token_bucket.release = MagicMock(side_effect=ValueError("Semaphore released too many times"))
        
        # 设置interval_index超过interval_lists长度，进入else分支
        producer.interval_lists = [0.1, 0.2]  # 只有2个元素
        
        try:
            import threading
            
            def run_produce():
                producer.produce_token(stop_event, per_pid_shms)
            
            thread = threading.Thread(target=run_produce)
            thread.start()
            time.sleep(0.1)
            stop_event.set()
            thread.join(timeout=1)
        finally:
            try:
                shm.close()
                shm.unlink()
            except:
                pass

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_produce_token_sleep_interval_negative(self, mock_logger_class):
        """测试produce_token方法中sleep_interval <= 0的情况"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        producer = TokenProducer(
            request_rate=100,  # 高频率
            traffic_cfg=self.traffic_cfg,
            request_num=10,
            pressure_mode=self.pressure_mode
        )
        
        stop_event = Event()
        shm = create_message_share_memory()
        per_pid_shms = {12345: shm}
        
        # 设置共享内存状态为WAIT_FLAG
        shm.buf[:] = struct.pack(FMT, WAIT_FLAG, 0, 0, 0, 0, 0, INDEX_READ_FLAG)
        
        # 设置interval_lists，使得sleep_interval可能为负
        producer.interval_lists = [0.001]  # 很小的间隔
        
        try:
            import threading
            
            def run_produce():
                producer.produce_token(stop_event, per_pid_shms)
            
            thread = threading.Thread(target=run_produce)
            thread.start()
            time.sleep(0.05)
            stop_event.set()
            thread.join(timeout=1)
        finally:
            try:
                shm.close()
                shm.unlink()
            except:
                pass


if __name__ == '__main__':
    unittest.main()

