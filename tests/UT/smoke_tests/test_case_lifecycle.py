"""Standalone lifecycle regressions; no benchmark, hardware, or logger initialization."""
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


class CaseLifecycleTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[3] / "smoke_tests/framework/run_test_case.py"
        spec = importlib.util.spec_from_file_location("_smoke_case_lifecycle_under_test", source)
        self.case = importlib.util.module_from_spec(spec)
        # Restore any existing run_log module; never install persistent production-name stubs.
        with mock.patch.dict(sys.modules, {"run_log": SimpleNamespace(log=mock.Mock())}):
            spec.loader.exec_module(self.case)

    def test_log_handles_success_open_failure_and_spawn_failure(self):
        process = mock.Mock(returncode=0)
        with mock.patch("builtins.open", mock.mock_open()) as opened, \
                mock.patch.object(self.case.subprocess, "Popen", return_value=process) as spawn:
            self.assertEqual(self.case.exec_script_file("case.py", "result.log", 1), 0)
            process.wait.assert_called_once_with(timeout=60)
            self.assertEqual(spawn.call_args.kwargs["start_new_session"], os.name == "posix")
            opened().close.assert_called_once()
        for failure in ("open", "spawn"):
            with self.subTest(failure=failure), \
                    mock.patch("builtins.open", mock.mock_open()) as opened, \
                    mock.patch.object(self.case.subprocess, "Popen") as spawn:
                (opened if failure == "open" else spawn).side_effect = OSError("primary error")
                with self.assertRaisesRegex(self.case.RunCaseException, "primary error"):
                    self.case.exec_script_file("case.py", "result.log")
                if failure == "spawn":
                    opened().close.assert_called_once()
                else:
                    spawn.assert_not_called()

    def test_timeout_and_interrupt_kill_group_even_after_leader_exits(self):
        for platform in ("posix", "nt"):
            for reason in (subprocess.TimeoutExpired("case", 0), KeyboardInterrupt()):
                for leader_stubborn in (False, True):
                    with self.subTest(platform=platform, reason=type(reason), stubborn=leader_stubborn):
                        process = mock.Mock(pid=123)
                        process.poll.return_value = None if leader_stubborn else 0
                        process.wait.side_effect = [reason, subprocess.TimeoutExpired("case", 3)
                                                    if leader_stubborn else 0, 0]
                        fake_os = SimpleNamespace(name=platform, killpg=mock.Mock())
                        with mock.patch.object(self.case, "os", fake_os), \
                                mock.patch.object(self.case, "signal", SimpleNamespace(SIGTERM=15, SIGKILL=9)), \
                                mock.patch.object(self.case.subprocess, "Popen", return_value=process):
                            expected = KeyboardInterrupt if isinstance(reason, KeyboardInterrupt) else self.case.RunCaseException
                            with self.assertRaises(expected):
                                self.case.exec_script_file("case.py", timeout_minutes=0)
                        self.assertEqual(process.wait.call_args_list, [mock.call(timeout=0),
                                                                      mock.call(timeout=3), mock.call()])
                        if platform == "posix":
                            self.assertEqual(fake_os.killpg.call_args_list, [mock.call(123, 15), mock.call(123, 9)])
                        else:
                            process.terminate.assert_called_once()
                            self.assertEqual(process.kill.call_count, int(leader_stubborn))

    def test_cleanup_timeout_defaults_override_and_failure_conversion(self):
        with mock.patch.object(self.case, "read_yaml_config", side_effect=lambda path, config: config):
            self.assertEqual(self.case.get_test_case_config("case.yml")["clean_timeout"], 2)
        for configured, expected in (({}, 2), ({"clean_timeout": None}, 2), ({"clean_timeout": 0.01}, 0.01)):
            for outcome in (0, 7, self.case.RunCaseException("timeout"), OSError("spawn")):
                with self.subTest(config=configured, outcome=outcome), \
                        mock.patch.object(self.case.os.path, "exists", return_value=True), \
                        mock.patch.object(self.case, "exec_script_file") as execute:
                    execute.side_effect = [outcome]
                    if outcome == 0:
                        self.case.post_processing(configured, "case")
                    else:
                        with self.assertRaises(self.case.CleanCaseException):
                            self.case.post_processing(configured, "case")
                    execute.assert_called_once_with(os.path.join("case", "clean.sh"), timeout_minutes=expected)
        with mock.patch.object(self.case.os.path, "exists", return_value=False), \
                mock.patch.object(self.case, "exec_script_file") as execute:
            self.case.post_processing({}, "case")
            execute.assert_not_called()

    def test_main_case_preserves_results_and_interrupt_despite_secondary_failures(self):
        config = {"case_type": "smoke", "case_name": "example", "clean_timeout": 0.01}
        for outcome in (0, 7, self.case.RunCaseException("primary timeout"), KeyboardInterrupt()):
            with self.subTest(outcome=outcome), \
                    mock.patch.object(self.case.os.path, "exists", return_value=True), \
                    mock.patch.object(self.case, "exec_script_file", side_effect=[outcome, self.case.RunCaseException("cleanup timeout")]) as execute, \
                    mock.patch.object(self.case, "get_last_lines", return_value=["primary failure"]), \
                    mock.patch.object(self.case, "record_and_analysis_metric") as metrics, \
                    mock.patch.object(self.case, "gather_logs", side_effect=OSError("collection failure")) as gather:
                if isinstance(outcome, KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        self.case.run_test_case(config, "test-case/group/case", "res", "workspace")
                else:
                    kind, name, result = self.case.run_test_case(config, "test-case/group/case", "res", "workspace")
                    self.assertEqual((kind, name), ("smoke", "example"))
                    self.assertEqual(result["success"], outcome == 0)
                    self.assertEqual(result["metric_key"], ["group", "case"])
                    if outcome == 0:
                        self.assertEqual(result["message"], "")
                        self.assertIsNotNone(result["execution_time"])
                    else:
                        self.assertIn("primary", result["message"])
                        self.assertNotIn("collection", result["message"])
                self.assertEqual(metrics.call_count, int(outcome == 0))
                self.assertEqual(gather.call_count, int(outcome != 0))
                self.assertEqual(execute.call_count, 2)
                self.assertEqual(execute.call_args.kwargs["timeout_minutes"], 0.01)
                self.assertTrue(any("WARRING" in str(call) for call in self.case.log.call_args_list))

    def test_metric_and_log_tail_failures_still_collect_and_clean(self):
        for code, expected in ((0, "metric failure"), (7, "run script path failed")):
            with self.subTest(code=code), \
                    mock.patch.object(self.case.os.path, "exists", return_value=True), \
                    mock.patch.object(self.case, "exec_script_file", return_value=code), \
                    mock.patch.object(self.case, "record_and_analysis_metric", side_effect=self.case.RunCaseException("metric failure")), \
                    mock.patch.object(self.case, "get_last_lines", side_effect=OSError("log unreadable")), \
                    mock.patch.object(self.case, "gather_logs") as gather, \
                    mock.patch.object(self.case, "post_processing") as clean:
                result = self.case.run_test_case({}, "test-case/group/case", "res", "workspace")[2]
            self.assertFalse(result["success"])
            self.assertIn(expected, result["message"])
            gather.assert_called_once()
            clean.assert_called_once()

    @unittest.skipUnless(os.name == "posix", "POSIX process groups required")
    def test_real_timeout_and_interrupt_kill_stubborn_descendant(self):
        real_wait = subprocess.Popen.wait
        for reason in (subprocess.TimeoutExpired("case", 1), KeyboardInterrupt()):
            with self.subTest(reason=type(reason)), tempfile.TemporaryDirectory() as directory:
                ready = Path(directory) / "pids"
                script = Path(directory) / "case.py"
                child = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"
                script.write_text("import os,subprocess,sys,time\n"
                                  f"child=subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.PIPE,text=True)\n"
                                  "child.stdout.readline()\n"
                                  f"open({str(ready)!r},'w').write(str(os.getpid())+' '+str(child.pid))\n"
                                  "time.sleep(60)\n", encoding="utf-8")
                leaders = []

                def interrupted_wait(process, timeout=None):
                    if not leaders:
                        leaders.append(process)
                        deadline = time.monotonic() + 5
                        while not ready.exists() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        self.assertTrue(ready.exists(), "descendant did not become ready")
                        if isinstance(reason, KeyboardInterrupt):
                            raise reason
                        return real_wait(process, timeout=0.05)  # Real timeout, after readiness.
                    return real_wait(process, timeout=timeout)

                try:
                    with mock.patch.object(subprocess.Popen, "wait", interrupted_wait):
                        expected = KeyboardInterrupt if isinstance(reason, KeyboardInterrupt) else self.case.RunCaseException
                        with self.assertRaises(expected):
                            self.case.exec_script_file(str(script), timeout_minutes=1)
                    self.assertIsNotNone(leaders[0].returncode)  # Leader was reaped.
                    descendant = int(ready.read_text().split()[1])
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        try:
                            os.kill(descendant, 0)
                        except ProcessLookupError:
                            break
                        stat = Path(f"/proc/{descendant}/stat")
                        if stat.exists() and stat.read_text().split()[2] == "Z":
                            break  # Orphan is dead; its new parent owns reaping it.
                        time.sleep(0.01)
                    else:
                        self.fail("stubborn descendant survived process-group shutdown")
                finally:
                    for process in leaders:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        real_wait(process, timeout=5)


if __name__ == "__main__":
    unittest.main()
