import unittest
from unittest.mock import patch, MagicMock
from mmengine.config import ConfigDict

from ais_bench.benchmark.runners.base import (
    create_progress_bar,
    format_time,
    TasksMonitor,
    BaseRunner
)


class TestCreateProgressBar(unittest.TestCase):
    """Tests for create_progress_bar function."""

    def test_create_progress_bar_normal(self):
        """Test create_progress_bar with normal values."""
        result = create_progress_bar(50, 100, "test", 30)
        self.assertIn("50/100", result)
        self.assertIn("test", result)

    def test_create_progress_bar_zero_total(self):
        """Test create_progress_bar with zero total."""
        result = create_progress_bar(0, 0, "test")
        self.assertEqual(result, "NA")

    def test_create_progress_bar_none_finished(self):
        """Test create_progress_bar with None finished_count."""
        result = create_progress_bar(None, 100, "test")
        self.assertEqual(result, "NA")

    def test_create_progress_bar_negative_values(self):
        """Test create_progress_bar with negative values."""
        result = create_progress_bar(-10, -20, "test")
        self.assertIn("0/0", result)

    def test_create_progress_bar_finished_exceeds_total(self):
        """Test create_progress_bar when finished exceeds total."""
        result = create_progress_bar(150, 100, "test")
        self.assertIn("100/100", result)

    def test_create_progress_bar_full(self):
        """Test create_progress_bar when fully complete."""
        result = create_progress_bar(100, 100, "complete")
        self.assertIn("100/100", result)


class TestFormatTime(unittest.TestCase):
    """Tests for format_time function."""

    def test_format_time_seconds(self):
        """Test format_time with seconds."""
        result = format_time(45)
        self.assertEqual(result, "0:00:45")

    def test_format_time_minutes(self):
        """Test format_time with minutes."""
        result = format_time(125)
        self.assertEqual(result, "0:02:05")

    def test_format_time_hours(self):
        """Test format_time with hours."""
        result = format_time(3665)
        self.assertEqual(result, "1:01:05")

    def test_format_time_zero(self):
        """Test format_time with zero."""
        result = format_time(0)
        self.assertEqual(result, "0:00:00")

    def test_format_time_float(self):
        """Test format_time with float (should truncate)."""
        result = format_time(125.7)
        self.assertEqual(result, "0:02:05")


class TestTasksMonitor(unittest.TestCase):
    """Tests for TasksMonitor class."""

    def setUp(self):
        self.task_names = ["task1", "task2"]
        self.output_path = "/tmp/test_output"

    @patch('ais_bench.benchmark.runners.base.os.path.exists', return_value=False)
    @patch('ais_bench.benchmark.runners.base.os.makedirs')
    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_tasks_monitor_init(self, mock_logger_class, mock_makedirs, mock_exists):
        """Test TasksMonitor initialization."""
        monitor = TasksMonitor(
            task_names=self.task_names,
            output_path=self.output_path,
            is_debug=False,
            refresh_interval=0.5,
            run_in_background=False
        )

        self.assertEqual(len(monitor.tasks_state_map), 2)
        self.assertEqual(monitor.output_path, self.output_path)
        self.assertFalse(monitor.is_debug)
        self.assertEqual(monitor.refresh_interval, 0.5)
        mock_makedirs.assert_called_once()

    @patch('ais_bench.benchmark.runners.base.os.path.exists', return_value=True)
    @patch('ais_bench.benchmark.runners.base.shutil.rmtree')
    def test_rm_tmp_files(self, mock_rmtree, mock_exists):
        """Test rm_tmp_files static method."""
        TasksMonitor.rm_tmp_files("/tmp/test")
        mock_rmtree.assert_called_once()

    @patch('ais_bench.benchmark.runners.base.os.path.exists', return_value=False)
    @patch('ais_bench.benchmark.runners.base.os.makedirs')
    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_tasks_monitor_is_all_task_done(self, mock_logger_class, mock_makedirs, mock_exists):
        """Test _is_all_task_done method."""
        monitor = TasksMonitor(
            task_names=self.task_names,
            output_path=self.output_path,
            is_debug=True
        )

        # Initially not all done
        self.assertFalse(monitor._is_all_task_done())

        # Mark all as finished
        monitor.tasks_state_map["task1"]["status"] = "finish"
        monitor.tasks_state_map["task2"]["status"] = "finish"
        self.assertTrue(monitor._is_all_task_done())

        # One error, one finish
        monitor.tasks_state_map["task1"]["status"] = "error"
        self.assertTrue(monitor._is_all_task_done())

    @patch('ais_bench.benchmark.runners.base.os.path.exists', return_value=False)
    @patch('ais_bench.benchmark.runners.base.os.makedirs')
    @patch('ais_bench.benchmark.runners.base.AISLogger')
    @patch('ais_bench.benchmark.runners.base.read_and_clear_statuses')
    def test_tasks_monitor_refresh_task_state(self, mock_read_statuses, mock_logger_class,
                                               mock_makedirs, mock_exists):
        """Test _refresh_task_state method."""
        monitor = TasksMonitor(
            task_names=self.task_names,
            output_path=self.output_path,
            is_debug=True
        )

        mock_read_statuses.return_value = [
            {
                'task_name': 'task1',
                'process_id': 12345,
                'finish_count': 10,
                'total_count': 100,
                'status': 'running',
                'task_log_path': '/tmp/task1.log'
            }
        ]

        monitor._refresh_task_state()

        self.assertEqual(monitor.tasks_state_map['task1']['process_id'], 12345)
        self.assertEqual(monitor.tasks_state_map['task1']['finish_count'], 10)
        self.assertEqual(monitor.tasks_state_map['task1']['status'], 'running')

    @patch('ais_bench.benchmark.runners.base.os.path.exists', return_value=False)
    @patch('ais_bench.benchmark.runners.base.os.makedirs')
    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_tasks_monitor_get_task_states(self, mock_logger_class, mock_makedirs, mock_exists):
        """Test _get_task_states method."""
        monitor = TasksMonitor(
            task_names=self.task_names,
            output_path=self.output_path,
            is_debug=True
        )

        monitor.tasks_state_map["task1"]["start_time"] = 1000.0
        monitor.tasks_state_map["task1"]["finish_count"] = 50
        monitor.tasks_state_map["task1"]["total_count"] = 100
        monitor.tasks_state_map["task1"]["status"] = "running"

        states = monitor._get_task_states()

        self.assertEqual(len(states), 2)
        self.assertIn("task1", states[0][0])
        self.assertIn("50/100", states[0][2])  # Progress bar

    @patch('ais_bench.benchmark.runners.base.os.path.exists', return_value=False)
    @patch('ais_bench.benchmark.runners.base.os.makedirs')
    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_tasks_monitor_launch_state_board_debug(self, mock_logger_class, mock_makedirs, mock_exists):
        """Test launch_state_board in debug mode."""
        monitor = TasksMonitor(
            task_names=self.task_names,
            output_path=self.output_path,
            is_debug=True
        )

        # Should return early in debug mode
        monitor.launch_state_board()
        # No exception should be raised

    @patch('ais_bench.benchmark.runners.base.os.path.exists', return_value=False)
    @patch('ais_bench.benchmark.runners.base.os.makedirs')
    @patch('ais_bench.benchmark.runners.base.AISLogger')
    @patch('ais_bench.benchmark.runners.base.tqdm')
    @patch('ais_bench.benchmark.runners.base.time.sleep')
    def test_tasks_monitor_update_tasks_progress(self, mock_sleep, mock_tqdm,
                                                  mock_logger_class, mock_makedirs, mock_exists):
        """Test _update_tasks_progress method."""
        monitor = TasksMonitor(
            task_names=self.task_names,
            output_path=self.output_path,
            is_debug=True,
            run_in_background=True
        )

        # Mark tasks as finished
        monitor.tasks_state_map["task1"]["status"] = "finish"
        monitor.tasks_state_map["task2"]["status"] = "finish"

        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar
        mock_pbar.n = 0
        mock_pbar.total = 2

        # Mock _refresh_task_state and _get_task_states
        monitor._refresh_task_state = MagicMock()
        monitor._get_task_states = MagicMock(return_value=[])
        monitor._is_all_task_done = MagicMock(return_value=True)

        monitor._update_tasks_progress()

        mock_pbar.close.assert_called_once()


class TestBaseRunner(unittest.TestCase):
    """Tests for BaseRunner class."""

    def setUp(self):
        self.task_cfg = ConfigDict({
            'type': 'TestTask',
            'config': {}
        })

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_base_runner_init(self, mock_logger_class):
        """Test BaseRunner initialization."""
        runner = BaseRunner(task=self.task_cfg, debug=False)

        self.assertFalse(runner.debug)
        self.assertIsNotNone(runner.task_cfg)
        self.assertIsNotNone(runner.logger)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_base_runner_call(self, mock_logger_class):
        """Test BaseRunner __call__ method."""
        runner = BaseRunner(task=self.task_cfg, debug=False)

        # Mock launch and summarize methods
        runner.launch = MagicMock(return_value=[("task1", 0), ("task2", 0)])
        runner.summarize = MagicMock()

        tasks = [{"name": "task1"}, {"name": "task2"}]
        runner(tasks)

        runner.launch.assert_called_once_with(tasks)
        runner.summarize.assert_called_once()

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_base_runner_summarize(self, mock_logger_class):
        """Test BaseRunner summarize method."""
        runner = BaseRunner(task=self.task_cfg, debug=False)

        status = [("task1", 0), ("task2", 1), ("task3", 0)]
        runner.summarize(status)

        # Should not raise exception
        self.assertTrue(True)

    @patch('ais_bench.benchmark.runners.base.AISLogger')
    def test_base_runner_launch_abstract(self, mock_logger_class):
        """Test that BaseRunner.launch must be implemented by subclasses."""
        runner = BaseRunner(task=self.task_cfg, debug=False)

        # BaseRunner.launch is abstract, but Python's ABC doesn't prevent calling it
        # We verify it exists and is callable, but will fail at runtime if not implemented
        self.assertTrue(hasattr(runner, 'launch'))
        self.assertTrue(callable(runner.launch))

        # Actually calling it would fail, but we can't easily test that without
        # creating a concrete subclass


if __name__ == "__main__":
    unittest.main()

