"""Cover probe failure paths without launching a Harness process."""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ais_bench.benchmark.tasks.custom_tasks import agentx_task as task


@pytest.fixture
def probe(monkeypatch):
    python = "fixture-python"
    env = {"PATH": "fixture-path"}
    run = Mock()
    monkeypatch.setattr(task.subprocess, "run", run)
    return python, env, run


def metadata_result(**overrides):
    fields = {
        "python_version": "3.11.9",
        "agentx_version": task.DEFAULT_HARNESS_VERSION,
        "upstream_aiperf": False,
    }
    return subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="optional interpreter banner\n" + json.dumps({**fields, **overrides}) + "\n",
        stderr="",
    )


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("missing executable"), subprocess.TimeoutExpired("fixture-python", 120)],
    ids=["missing-executable", "metadata-timeout"],
)
def test_metadata_probe_wraps_subprocess_errors(probe, error):
    python, env, run = probe
    run.side_effect = error

    with pytest.raises(RuntimeError, match="AGENTX_ENV_INVALID: cannot query") as raised:
        task.probe_harness_environment(python, env)

    assert raised.value.__cause__ is error
    assert python in str(raised.value)
    run.assert_called_once()


@pytest.mark.parametrize(
    ("stdout", "stderr", "diagnostic"),
    [
        ("not the diagnostic", "  distribution not found\n", "distribution not found"),
        ("  fallback diagnostic\n", " \n", "fallback diagnostic"),
    ],
)
def test_metadata_probe_propagates_diagnostic_without_running_help(probe, stdout, stderr, diagnostic):
    python, env, run = probe
    run.return_value = subprocess.CompletedProcess([], 1, stdout, stderr)

    with pytest.raises(RuntimeError, match="AGENTX_ENV_INVALID:.*unavailable") as raised:
        task.probe_harness_environment(python, env)

    assert str(raised.value).endswith(diagnostic)
    run.assert_called_once()


@pytest.mark.parametrize("stdout", ["  \n", "banner only\n", "{truncated\n"])
def test_metadata_probe_rejects_unreadable_result(probe, stdout):
    python, env, run = probe
    run.return_value = subprocess.CompletedProcess([], 0, stdout, "")

    with pytest.raises(RuntimeError, match="invalid environment probe output") as raised:
        task.probe_harness_environment(python, env)

    assert repr(stdout) in str(raised.value)
    run.assert_called_once()


@pytest.mark.parametrize("version", ["3.12.8", "3.13.1"])
def test_supported_metadata_reaches_help_probe_using_same_environment(probe, monkeypatch, version):
    python, env, run = probe
    entry = "custom agentx"
    monkeypatch.setattr(task.platform, "platform", lambda: "fixture-platform")
    run.side_effect = [metadata_result(python_version=version), subprocess.CompletedProcess([], 0, "Usage", "")]

    result = task.probe_harness_environment(python, env, agentx_bin=entry)

    assert result == {
        "python_version": version,
        "agentx_version": task.DEFAULT_HARNESS_VERSION,
        "upstream_aiperf": False,
        "distribution": "agentx-harness",
        "version": task.DEFAULT_HARNESS_VERSION,
        "python": python,
        "entry": [entry],
        "platform": "fixture-platform",
    }
    metadata_call, help_call = run.call_args_list
    assert metadata_call.args[0][:2] == [python, "-c"]
    assert metadata_call.kwargs["env"] is env
    assert metadata_call.kwargs["timeout"] == 120
    assert help_call.args == ([entry, "--help"],)
    assert help_call.kwargs["env"] is env
    assert help_call.kwargs["timeout"] == 30
    assert help_call.kwargs["stderr"] == subprocess.STDOUT


@pytest.mark.parametrize(
    "error",
    [PermissionError("not executable"), subprocess.TimeoutExpired("custom-agentx", 30)],
    ids=["entry-permission-error", "help-timeout"],
)
def test_help_probe_wraps_subprocess_errors(probe, error):
    python, env, run = probe
    run.side_effect = [metadata_result(), error]

    with pytest.raises(RuntimeError, match="AGENTX_ENV_INVALID: agentx entry self-check failed") as raised:
        task.probe_harness_environment(python, env, agentx_bin="custom-agentx")

    assert raised.value.__cause__ is error
    assert run.call_count == 2


def test_help_probe_keeps_tail_of_failed_command_output(probe):
    python, env, run = probe
    detail = "actionable error at end"
    run.side_effect = [
        metadata_result(),
        subprocess.CompletedProcess([], 2, "unneeded prefix" + "x" * 2000 + detail, None),
    ]

    with pytest.raises(RuntimeError, match="AGENTX_ENV_INVALID: agentx entry self-check failed") as raised:
        task.probe_harness_environment(python, env, agentx_bin="custom-agentx")

    assert str(raised.value).endswith(detail)
    assert "unneeded prefix" not in str(raised.value)
    assert run.call_count == 2


def test_entry_expands_home_before_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    [entry] = task.resolve_agentx_bin("~/bin/agentx", None)
    assert type(tmp_path)(entry) == tmp_path / "bin" / "agentx"


def test_entry_reports_no_interpreter_or_explicit_binary():
    with pytest.raises(RuntimeError, match="set agentx_bin or agentx_python"):
        task.resolve_agentx_bin(None, None)


@pytest.mark.parametrize("raw", ["NaN", "inf", "-inf", "-0", "1e309"])
def test_watchdog_setting_rejects_nonfinite_or_nonpositive_timeout(raw):
    with pytest.raises(ValueError, match=f"{task.PROCESS_TIMEOUT_ENV} must be a positive number"):
        task._process_timeout({task.PROCESS_TIMEOUT_ENV: raw})


@pytest.mark.parametrize("raw", ["  ", "\t\n"])
def test_whitespace_timeout_does_not_enable_watchdog(raw):
    assert task._process_timeout({task.PROCESS_TIMEOUT_ENV: raw}) is None


def test_invalid_timeout_retains_conversion_error():
    with pytest.raises(ValueError, match="positive number of seconds") as raised:
        task._process_timeout({task.PROCESS_TIMEOUT_ENV: "ten seconds"})
    assert isinstance(raised.value.__cause__, ValueError)


def test_watchdog_does_not_signal_already_finished_process(monkeypatch):
    kill_group = Mock()
    monkeypatch.setattr(task.os, "killpg", kill_group, raising=False)
    completed = Mock()
    completed.wait.return_value = True
    timed_out = Mock()

    task.AgentXTask._watchdog(SimpleNamespace(pid=123), 0.25, completed, timed_out)

    completed.wait.assert_called_once_with(0.25)
    timed_out.set.assert_not_called()
    kill_group.assert_not_called()


@pytest.mark.parametrize(
    "error", [None, ProcessLookupError("already exited"), PermissionError("cannot signal")],
    ids=["kills-group", "group-exited", "signal-denied"],
)
def test_watchdog_records_timeout_before_attempting_to_stop_group(monkeypatch, error):
    completed = Mock()
    completed.wait.return_value = False
    timed_out = Mock()
    calls = Mock()
    calls.attach_mock(timed_out.set, "mark_timeout")
    calls.kill_group.side_effect = error
    monkeypatch.setattr(task.os, "killpg", calls.kill_group, raising=False)
    monkeypatch.setattr(task.signal, "SIGKILL", 9, raising=False)

    task.AgentXTask._watchdog(SimpleNamespace(pid=123), 0.25, completed, timed_out)

    completed.wait.assert_called_once_with(0.25)
    assert [record[0] for record in calls.mock_calls] == ["mark_timeout", "kill_group"]
    calls.kill_group.assert_called_once_with(123, 9)


def test_watchdog_does_not_swallow_unexpected_os_error(monkeypatch):
    completed = Mock()
    completed.wait.return_value = False
    timed_out = Mock()
    monkeypatch.setattr(task.os, "killpg", Mock(side_effect=OSError("unexpected")), raising=False)
    monkeypatch.setattr(task.signal, "SIGKILL", 9, raising=False)

    with pytest.raises(OSError, match="unexpected"):
        task.AgentXTask._watchdog(SimpleNamespace(pid=123), 0.25, completed, timed_out)
    timed_out.set.assert_called_once_with()
