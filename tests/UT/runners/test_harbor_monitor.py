import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from ais_bench.benchmark.runners.harbor_monitor import (
    HarborMonitor,
    HarborMonitorServer,
    _tail,
    _read_json,
    _fromiso,
)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class TestTail(unittest.TestCase):
    def test_missing_file(self):
        self.assertIsNone(_tail(Path("does/not/exist")))

    def test_returns_last_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.txt"
            p.write_text("\n".join(str(i) for i in range(100)), encoding="utf-8")
            tail = _tail(p)
            self.assertIsNotNone(tail)
            self.assertTrue("99" in tail)
            self.assertFalse("0\n" in tail)


class TestReadJson(unittest.TestCase):
    def test_missing(self):
        self.assertIsNone(_read_json(Path("does/not/exist.json")))

    def test_valid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            _write_json(p, {"a": 1})
            self.assertEqual(_read_json(p), {"a": 1})

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            p.write_text("{broken", encoding="utf-8")
            self.assertIsNone(_read_json(p))

    def test_non_dict(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            _write_json(p, [1, 2])
            self.assertIsNone(_read_json(p))


class TestFromIso(unittest.TestCase):
    def test_normal(self):
        self.assertIsNotNone(_fromiso("2026-01-01T00:00:00"))

    def test_none(self):
        self.assertIsNone(_fromiso(None))

    def test_invalid(self):
        self.assertIsNone(_fromiso("not-a-date"))


class TestHarborMonitor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.job_dir = Path(self.temp_dir) / "results" / "m" / "d" / "details"
        self.job_dir.mkdir(parents=True)
        self.monitor = HarborMonitor(self.temp_dir, refresh_interval=0.05)
        self.monitor.register_task("m/d", status_file=None, job_dir=str(self.job_dir))

    def tearDown(self):
        self.monitor.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _make_job_result(self):
        _write_json(
            self.job_dir / "result.json",
            {
                "n_total_trials": 3,
                "status": "done",
                "stats": {
                    "n_completed_trials": 2,
                    "n_errored_trials": 1,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_cancelled_trials": 0,
                    "n_retries": 0,
                    "evals": {
                        "eval1": {
                            "exception_stats": {"Timeout": ["caseA"]}
                        }
                    },
                },
            },
        )

    def _make_trial(self, name, result):
        tdir = self.job_dir / name
        _write_json(tdir / "result.json", result)
        return tdir

    def test_register_and_refresh(self):
        self._make_job_result()
        self.monitor.refresh()
        snap = self.monitor.snapshot("m/d")
        self.assertEqual(snap["task_name"], "m/d")
        self.assertEqual(snap["harbor"]["n_total_trials"], 3)

    def test_snapshot_all(self):
        self.monitor.refresh()
        self.assertIsInstance(self.monitor.snapshot(), list)

    def test_snapshot_missing_task(self):
        self.assertIsNone(self.monitor.snapshot("no/such"))

    def test_cases_empty_when_no_trials(self):
        self._make_job_result()
        self.monitor.refresh()
        self.assertEqual(self.monitor.cases("m/d"), [])
        self.assertEqual(self.monitor.cases("no/such"), [])

    def test_case_completed(self):
        self._make_trial(
            "trial_00000",
            {
                "trial_name": "caseA",
                "task_name": "taskA",
                "verifier_result": {"rewards": {"reward": 1.0}},
            },
        )
        self.monitor.refresh()
        cases = self.monitor.cases("m/d")
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["status"], "completed")
        self.assertEqual(cases[0]["reward"], 1.0)

    def test_case_errored_and_cancelled(self):
        self._make_trial(
            "trial_00000",
            {
                "trial_name": "errCase",
                "exception_info": {"exception_type": "Timeout", "exception_message": "boom"},
            },
        )
        self._make_trial(
            "trial_00001",
            {
                "trial_name": "cancCase",
                "exception_info": {"exception_type": "CancelledError"},
            },
        )
        self.monitor.refresh()
        states = {c["trial_name"]: c["status"] for c in self.monitor.cases("m/d")}
        self.assertEqual(states["errCase"], "errored")
        self.assertEqual(states["cancCase"], "cancelled")

    def test_case_running_by_files(self):
        tdir = self.job_dir / "trial_00000"
        tdir.mkdir()
        (tdir / "agent").mkdir()
        (tdir / "agent" / "trajectory.json").write_text("{}", encoding="utf-8")
        self.monitor.refresh()
        cases = self.monitor.cases("m/d")
        self.assertEqual(cases[0]["status"], "running")
        self.assertTrue(cases[0]["agent"]["has_trajectory"])

    def test_jobs_aggregation(self):
        self._make_trial(
            "trial_00000",
            {"trial_name": "a", "verifier_result": {"rewards": {"reward": 1.0}}},
        )
        self.monitor.refresh()
        jobs = self.monitor.jobs()
        self.assertEqual(jobs[0]["case_status_counts"], {"completed": 1})

    def test_task_info(self):
        info = self.monitor.task_info("m/d")
        self.assertEqual(info["job_dir"], str(self.job_dir))
        self.assertIsNone(self.monitor.task_info("no/such"))

    def test_raw_job_result(self):
        self._make_job_result()
        self.assertEqual(self.monitor.raw_job_result("m/d")["n_total_trials"], 3)
        self.assertIsNone(self.monitor.raw_job_result("no/such"))

    def test_raw_case_result_by_trial_dir(self):
        self._make_trial("trial_00000", {"trial_name": "a", "x": 1})
        self.assertEqual(
            self.monitor.raw_case_result("m/d", "trial_00000")["x"], 1
        )

    def test_raw_case_result_by_index(self):
        self._make_trial("trial_00002", {"trial_name": "a", "x": 2})
        self.assertEqual(self.monitor.raw_case_result("m/d", "2")["x"], 2)

    def test_raw_case_result_by_task_name(self):
        tdir = self.job_dir / "trial_00000"
        _write_json(tdir / "result.json", {"trial_name": "a", "x": 3})
        _write_json(tdir / "config.json", {"task": {"name": "astropy__123"}})
        self.assertEqual(
            self.monitor.raw_case_result("m/d", "astropy__123")["x"], 3
        )

    def test_raw_case_result_missing(self):
        self.assertIsNone(self.monitor.raw_case_result("m/d", "no_such"))
        self.assertIsNone(self.monitor.raw_case_result("no/task", "x"))

    def test_cache_reuse(self):
        self._make_trial("trial_00000", {"trial_name": "a", "x": 1})
        self.monitor.refresh()
        key = str(self.job_dir / "trial_00000" / "result.json")
        self.assertIn(key, self.monitor._case_cache)


class TestHarborMonitorServer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.job_dir = Path(self.temp_dir) / "job"
        self.job_dir.mkdir(parents=True)
        self.monitor = HarborMonitor(self.temp_dir)
        self.monitor.register_task("m/d", status_file=None, job_dir=str(self.job_dir))

    def tearDown(self):
        self.monitor.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @staticmethod
    def _free_port() -> int:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
        finally:
            s.close()

    def _get(self, port, path):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read().decode("utf-8")
        conn.close()
        return resp.status, body

    def test_start_disabled_by_port_zero(self):
        server = HarborMonitorServer(self.monitor, port=0)
        try:
            self.assertIsNone(server.start())
        finally:
            server.stop()

    def test_server_endpoints(self):
        server = HarborMonitorServer(self.monitor, port=self._free_port())
        port = server.start()
        self.assertIsNotNone(port)
        try:
            status, body = self._get(port, "/api/health")
            self.assertEqual(status, 200)
            self.assertIn("ok", body)

            status, body = self._get(port, "/api/tasks")
            self.assertEqual(status, 200)

            status, body = self._get(port, "/api/tasks")
            self.assertIn("tasks", body)

            status, _ = self._get(port, "/no/such")
            self.assertEqual(status, 404)

            # raw job result for a registered task -> 404 (no result.json)
            status, _ = self._get(port, "/api/tasks/m/d/")
            self.assertEqual(status, 404)
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()