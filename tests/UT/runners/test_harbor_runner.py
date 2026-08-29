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


if __name__ == "__main__":
    unittest.main()