import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from ais_bench.benchmark.runners.harbor_runner import HarborRunner
from ais_bench.benchmark.utils.config import ConfigDict


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class TestHarborRunnerTaskJobDir(unittest.TestCase):
    def test_task_job_dir(self):
        runner = HarborRunner(task={"type": "X"})
        task = {
            "work_dir": "/wd",
            "models": [{"abbr": "m"}],
            "datasets": [[{"abbr": "d"}]],
        }
        self.assertEqual(
            runner._task_job_dir(task),
            os.path.join("/wd", "results", "m", "d", "details"),
        )


class TestHarborRunnerPurge(unittest.TestCase):
    def setUp(self):
        self.runner = HarborRunner(task={"type": "X"})
        self.runner.logger = mock.MagicMock()
        self.temp_dir = tempfile.mkdtemp()
        self.job_dir = Path(self.temp_dir) / "details"
        self.job_dir.mkdir(parents=True)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_no_result_json_noop(self):
        self.runner._purge_exception_cases(str(self.job_dir))

    def test_no_job_dir_noop(self):
        self.runner._purge_exception_cases("")

    def test_removes_exception_cases_from_stats(self):
        _write_json(
            self.job_dir / "result.json",
            {
                "stats": {
                    "evals": {
                        "e": {"exception_stats": {"Timeout": ["caseA", "caseB"]}}
                    }
                }
            },
        )
        a = self.job_dir / "caseA"
        a.mkdir()
        b = self.job_dir / "caseB"
        b.mkdir()
        ok = self.job_dir / "caseOK"
        ok.mkdir()
        self.runner._purge_exception_cases(str(self.job_dir))
        self.assertFalse(a.exists())
        self.assertFalse(b.exists())
        self.assertTrue(ok.exists())

    def test_removes_exception_cases_from_trial_results(self):
        _write_json(
            self.job_dir / "result.json",
            {
                "trial_results": [
                    {"trial_name": "t1", "exception_info": {"type": "E"}},
                    {"trial_name": "t2", "exception_info": {"type": "E"}},
                    {"trial_name": "t3", "exception_info": None},
                ]
            },
        )
        for name in ("t1", "t2", "t3"):
            (self.job_dir / name).mkdir()
        self.runner._purge_exception_cases(str(self.job_dir))
        self.assertFalse((self.job_dir / "t1").exists())
        self.assertFalse((self.job_dir / "t2").exists())
        self.assertTrue((self.job_dir / "t3").exists())

    def test_invalid_json_noop(self):
        (self.job_dir / "result.json").write_text("{broken", encoding="utf-8")
        self.runner._purge_exception_cases(str(self.job_dir))


class TestHarborRunnerLaunch(unittest.TestCase):
    def setUp(self):
        self.runner = HarborRunner(
            task={"type": "X"}, purge_exception_cases=False
        )
        self.runner.logger = mock.MagicMock()
        self.tasks_single = [self._make_task()]
        self.tasks_multi = [self._make_task(), self._make_task()]

    def _make_task(self):
        return {
            "work_dir": "/wd",
            "models": [{"abbr": "m"}],
            "datasets": [[{"abbr": "d"}]],
        }

    @mock.patch.object(HarborRunner, "_launch_inline")
    def test_single_task_inline(self, mock_inline):
        mock_inline.return_value = ("t", 0)
        result = self.runner.launch(self.tasks_single)
        mock_inline.assert_called_once_with(self.tasks_single[0])
        self.assertEqual(result, [("t", 0)])

    @mock.patch.object(HarborRunner, "_launch_multi")
    def test_multi_task_subprocess(self, mock_multi):
        mock_multi.return_value = [("t", 0)]
        result = self.runner.launch(self.tasks_multi)
        mock_multi.assert_called_once_with(self.tasks_multi)
        self.assertEqual(result, [("t", 0)])

    @mock.patch.object(HarborRunner, "_purge_exception_cases")
    @mock.patch.object(HarborRunner, "_launch_inline")
    def test_purge_runs_before_dispatch(self, mock_inline, mock_purge):
        mock_inline.return_value = ("t", 0)
        runner = HarborRunner(task={"type": "X"}, purge_exception_cases=True)
        runner.logger = mock.MagicMock()
        runner.launch(self.tasks_single)
        self.assertEqual(mock_purge.call_count, 1)

    @mock.patch.object(HarborRunner, "_launch_inline")
    def test_no_purge_when_disabled(self, mock_inline):
        mock_inline.return_value = ("t", 0)
        self.runner.launch(self.tasks_single)


class TestHarborRunnerWaitForCleanup(unittest.TestCase):
    def setUp(self):
        self.runner = HarborRunner(task={"type": "X"})
        self.runner.logger = mock.MagicMock()

    def test_returns_when_no_popen(self):
        self.runner._wait_for_cleanup(timeout=0.01)

    @mock.patch("time.sleep")
    def test_terminates_on_timeout(self, mock_sleep):
        class FakePopen:
            def poll(self):
                return None

        self.runner._active_popens = [FakePopen()]
        with mock.patch.object(
            HarborRunner, "_terminate_popens"
        ) as mock_terminate:
            # force timeout path: elapsed >= timeout on first pass
            self.runner._wait_for_cleanup(timeout=-1)
            mock_terminate.assert_called_once()

    @mock.patch("time.sleep")
    def test_terminate_popens(self, mock_sleep):
        class Stuck:
            def __init__(self):
                self.terminated = False
                self.killed = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.killed = True

        p = Stuck()
        # fast-forward time so the 10s grace wait ends immediately
        def _fast_time():
            _fast_time.now += 100
            return _fast_time.now

        _fast_time.now = 0.0
        with mock.patch("time.time", side_effect=_fast_time):
            self.runner._terminate_popens([p])
        self.assertTrue(p.terminated)
        self.assertTrue(p.killed)


class TestHarborRunnerPurgeMore(unittest.TestCase):
    def setUp(self):
        self.runner = HarborRunner(task={"type": "X"})
        self.runner.logger = mock.MagicMock()
        self.temp_dir = tempfile.mkdtemp()
        self.job_dir = Path(self.temp_dir) / "details"
        self.job_dir.mkdir(parents=True)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_data_not_dict(self):
        _write_json(self.job_dir / "result.json", [1, 2])
        self.runner._purge_exception_cases(str(self.job_dir))

    def test_filters_non_string_names(self):
        _write_json(
            self.job_dir / "result.json",
            {
                "stats": {
                    "evals": {
                        "e": {"exception_stats": {"Timeout": ["a", "b", 123]}}
                    }
                },
                "trial_results": [
                    {"task_name": "x", "exception_info": {"type": "E"}},
                    {"trial_name": None, "exception_info": {"type": "E"}},
                    {"exception_info": None},
                    "notadict",
                ],
            },
        )
        for name in ("a", "b", "x"):
            (self.job_dir / name).mkdir()
        self.runner._purge_exception_cases(str(self.job_dir))
        self.assertFalse((self.job_dir / "a").exists())
        self.assertFalse((self.job_dir / "b").exists())
        self.assertFalse((self.job_dir / "x").exists())

    def test_no_names_noop(self):
        _write_json(
            self.job_dir / "result.json",
            {"stats": {"evals": {}}, "trial_results": [{"exception_info": None}]},
        )
        self.runner._purge_exception_cases(str(self.job_dir))


class TestHarborRunnerInit(unittest.TestCase):
    def test_unexpected_kwargs_logged(self):
        runner = HarborRunner(task={"type": "X"}, extra_param=1)
        self.assertTrue(hasattr(runner, "logger"))


class TestHarborRunnerInline(unittest.TestCase):
    def _make_task(self):
        return {
            "work_dir": "/wd",
            "models": [{"abbr": "m"}],
            "datasets": [[{"abbr": "d"}]],
        }

    def _runner(self):
        runner = HarborRunner(task={"type": "X"})
        runner.logger = mock.MagicMock()
        return runner

    @mock.patch("ais_bench.benchmark.runners.harbor_monitor.HarborMonitorServer")
    @mock.patch("ais_bench.benchmark.runners.harbor_monitor.HarborMonitor")
    @mock.patch("ais_bench.benchmark.runners.harbor_runner.TASKS")
    def test_inline_success(self, mock_tasks, mock_mon, mock_srv):
        built = mock.MagicMock()
        built.name = "agenttask"
        mock_tasks.build.return_value = built
        srv = mock.MagicMock()
        srv.start.return_value = 8080
        mock_srv.return_value = srv
        name, code = self._runner()._launch_inline(self._make_task())
        self.assertEqual((name, code), ("agenttask", 0))
        built.run.assert_called_once()
        srv.stop.assert_called_once()

    @mock.patch("ais_bench.benchmark.runners.harbor_monitor.HarborMonitorServer")
    @mock.patch("ais_bench.benchmark.runners.harbor_monitor.HarborMonitor")
    @mock.patch("ais_bench.benchmark.runners.harbor_runner.TASKS")
    def test_inline_failure(self, mock_tasks, mock_mon, mock_srv):
        built = mock.MagicMock()
        built.name = "agenttask"
        built.run.side_effect = RuntimeError("boom")
        mock_tasks.build.return_value = built
        srv = mock.MagicMock()
        srv.start.return_value = 0  # port 0 -> no server log
        mock_srv.return_value = srv
        name, code = self._runner()._launch_inline(self._make_task())
        self.assertEqual((name, code), ("agenttask", 1))


class TestHarborRunnerMulti(unittest.TestCase):
    def _make_task(self):
        return {
            "work_dir": "/wd",
            "models": [{"abbr": "m"}],
            "datasets": [[{"abbr": "d"}]],
        }

    def _runner(self, **kw):
        runner = HarborRunner(task={"type": "X"}, **kw)
        runner.logger = mock.MagicMock()
        return runner

    @mock.patch("ais_bench.benchmark.runners.harbor_runner.TasksMonitor.rm_tmp_files")
    @mock.patch("ais_bench.benchmark.runners.harbor_monitor.HarborMonitorServer")
    @mock.patch("ais_bench.benchmark.runners.harbor_monitor.HarborMonitor")
    @mock.patch.object(HarborRunner, "_start_task_board")
    @mock.patch.object(HarborRunner, "_run_tasks")
    def test_multi_success(self, mock_run, mock_board, mock_mon, mock_srv, mock_rm):
        mock_board.return_value = None
        mock_run.return_value = [("t", 0)]
        srv = mock.MagicMock()
        srv.start.return_value = 8080
        mock_srv.return_value = srv
        tasks = [self._make_task(), self._make_task()]
        result = self._runner()._launch_multi(tasks)
        self.assertEqual(result, [("t", 0)])
        mock_rm.assert_called_once_with("/wd")
        srv.stop.assert_called_once()

    @mock.patch("ais_bench.benchmark.runners.harbor_runner.TasksMonitor.rm_tmp_files")
    @mock.patch("ais_bench.benchmark.runners.harbor_monitor.HarborMonitorServer")
    @mock.patch("ais_bench.benchmark.runners.harbor_monitor.HarborMonitor")
    @mock.patch.object(HarborRunner, "_start_task_board")
    @mock.patch.object(HarborRunner, "_wait_for_cleanup")
    @mock.patch.object(HarborRunner, "_run_tasks")
    def test_multi_interrupt(self, mock_run, mock_wait, mock_board, mock_mon, mock_srv, mock_rm):
        mock_board.return_value = None
        mock_run.side_effect = KeyboardInterrupt()
        srv = mock.MagicMock()
        srv.start.return_value = 0
        mock_srv.return_value = srv
        tasks = [self._make_task()]
        with self.assertRaises(SystemExit) as cm:
            self._runner()._launch_multi(tasks)
        self.assertEqual(cm.exception.code, 130)
        mock_wait.assert_called_once()


class TestHarborRunnerBoard(unittest.TestCase):
    def _runner(self, **kw):
        runner = HarborRunner(task={"type": "X"}, **kw)
        runner.logger = mock.MagicMock()
        return runner

    def test_start_board_debug_returns_none(self):
        self.assertIsNone(self._runner(debug=True)._start_task_board(["t"], "/wd"))

    @mock.patch("ais_bench.benchmark.runners.harbor_runner.TasksMonitor")
    def test_start_board_launches(self, mock_tm):
        inst = mock.MagicMock()
        mock_tm.return_value = inst
        runner = self._runner()
        board = runner._start_task_board(["t"], "/wd")
        self.assertIsNotNone(board)
        for _ in range(200):
            if getattr(board, "_holder", {}).get("monitor") is not None:
                break
            time.sleep(0.005)
        self.assertIs(inst, board._holder.get("monitor"))
        inst.launch_state_board.assert_called()
        board.join(timeout=2)

    def test_stop_board_none(self):
        self._runner()._stop_board(None)

    def test_stop_board_stops_monitor(self):
        mon = mock.MagicMock()
        board = mock.MagicMock()
        board._holder = {"monitor": mon}
        self._runner()._stop_board(board)
        mon.stop_state_board.assert_called()


class TestHarborRunnerRunTasks(unittest.TestCase):
    def _make_task(self):
        return {
            "work_dir": "/wd",
            "models": [{"abbr": "m"}],
            "datasets": [[{"abbr": "d"}]],
        }

    @mock.patch.object(HarborRunner, "_launch")
    def test_sequential(self, mock_launch):
        mock_launch.side_effect = [("a", 0), ("b", 1)]
        runner = HarborRunner(task={"type": "X"})
        runner.logger = mock.MagicMock()
        self.assertEqual(
            runner._run_tasks([self._make_task(), self._make_task()]),
            [("a", 0), ("b", 1)],
        )

    @mock.patch.object(HarborRunner, "_launch")
    def test_threadpool(self, mock_launch):
        mock_launch.side_effect = [("a", 0), ("b", 1)]
        runner = HarborRunner(task={"type": "X"}, max_num_workers=2)
        runner.logger = mock.MagicMock()
        self.assertEqual(
            runner._run_tasks([self._make_task(), self._make_task()]),
            [("a", 0), ("b", 1)],
        )


class TestHarborRunnerCleanup(unittest.TestCase):
    def setUp(self):
        self.runner = HarborRunner(task={"type": "X"})
        self.runner.logger = mock.MagicMock()

    @mock.patch("time.sleep")
    def test_heartbeat_then_exit(self, mock_sleep):
        calls = {"n": 0}

        def fake_poll():
            calls["n"] += 1
            return None if calls["n"] == 1 else 0

        class P:
            def poll(self):
                return fake_poll()

        self.runner._active_popens = [P()]
        with mock.patch("time.time", side_effect=lambda: 0.0):
            self.runner._wait_for_cleanup(timeout=100)
        self.assertEqual(mock_sleep.call_count, 1)


class TestHarborRunnerLaunchSubprocess(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir, True)

    def _make_task(self):
        return {
            "work_dir": "/wd",
            "models": [{"abbr": "m"}],
            "datasets": [[{"abbr": "d"}]],
        }

    def _built(self):
        built = mock.MagicMock()
        built.name = "agenttask"
        built.cfg.dump = mock.MagicMock()
        built.get_command.return_value = "echo hi"
        built.get_log_path.return_value = os.path.join(self.temp_dir, "t.out")
        return built

    def _run(self, **kw):
        runner = HarborRunner(task={"type": "X"}, **kw)
        runner.logger = mock.MagicMock()
        return runner, self._make_task()

    @mock.patch("subprocess.Popen")
    @mock.patch("ais_bench.benchmark.runners.harbor_runner.TASKS")
    @mock.patch("ais_bench.benchmark.runners.harbor_runner.mmengine.mkdir_or_exist")
    @mock.patch("os.remove")
    def test_success(self, mock_remove, mock_mkdir, mock_tasks, mock_popen):
        built = self._built()
        mock_tasks.build.return_value = built
        proc = mock.MagicMock()
        proc.wait.return_value = 0
        mock_popen.return_value = proc
        runner, task = self._run()
        name, code = runner._launch(task)
        self.assertEqual((name, code), ("agenttask", 0))
        mock_remove.assert_called_once()

    @mock.patch("subprocess.Popen")
    @mock.patch("ais_bench.benchmark.runners.harbor_runner.TASKS")
    @mock.patch("ais_bench.benchmark.runners.harbor_runner.mmengine.mkdir_or_exist")
    @mock.patch("os.remove")
    def test_failure(self, mock_remove, mock_mkdir, mock_tasks, mock_popen):
        built = self._built()
        mock_tasks.build.return_value = built
        proc = mock.MagicMock()
        proc.wait.return_value = 3
        mock_popen.return_value = proc
        runner, task = self._run()
        name, code = runner._launch(task)
        self.assertEqual((name, code), ("agenttask", 3))
        runner.logger.error.assert_called_once()

    @mock.patch("subprocess.Popen")
    @mock.patch("ais_bench.benchmark.runners.harbor_runner.TASKS")
    @mock.patch("ais_bench.benchmark.runners.harbor_runner.mmengine.mkdir_or_exist")
    @mock.patch("os.remove")
    def test_keep_tmp_file(self, mock_remove, mock_mkdir, mock_tasks, mock_popen):
        built = self._built()
        mock_tasks.build.return_value = built
        proc = mock.MagicMock()
        proc.wait.return_value = 0
        mock_popen.return_value = proc
        runner, task = self._run(keep_tmp_file=True)
        runner._launch(task)
        mock_remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()