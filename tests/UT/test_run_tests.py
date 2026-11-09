"""Regression tests for explicit UT selection."""

import importlib.util
import io
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("selected", [None, ["tests/UT/runners/test_harbor_runner.py"]])
def test_runner_selects_files_without_adding_entire_directory(
    tmp_path, monkeypatch, selected
):
    spec = importlib.util.spec_from_file_location(
        "ut_runner_script", Path(__file__).parents[1] / "run_tests.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runner = module.TestRunner(str(tmp_path), str(tmp_path / "reports"))
    monkeypatch.setattr(runner, "clean_cache", lambda: None)
    process = Mock(stdout=io.StringIO("1 passed in 0.01s\n"), returncode=0)
    process.poll.return_value = 0
    start = Mock(return_value=process)
    monkeypatch.setattr(module.subprocess, "Popen", start)

    assert runner.run_tests(specific_tests=selected)
    command = start.call_args.args[0]
    targets = command[3 : command.index("--tb=short")]
    assert targets == (selected or [str(tmp_path.resolve())])
