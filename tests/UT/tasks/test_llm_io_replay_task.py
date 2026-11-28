"""Unit tests for replay task orchestration and report persistence."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from mmengine.config import ConfigDict

from ais_bench.benchmark.tasks.llm_io_replay import llm_io_replay as replay


@pytest.fixture(autouse=True)
def clean_replay_environment(monkeypatch):
    for name in replay.os.environ.copy():
        if name.startswith("AISBENCH_REPLAY_"):
            monkeypatch.delenv(name)


def make_task(work_dir, *datasets, **model_overrides):
    model = {"abbr": "model", "url": "http://example.invalid", "model": "chat"}
    return replay.LLMIOReplayTask(
        ConfigDict(
            models=[{**model, **model_overrides}],
            datasets=[list(datasets) or [{"abbr": "trace"}]],
            work_dir=str(work_dir),
            cli_args={"mode": "perf", "debug": False},
        )
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"url": "https:///missing-host"}, "Invalid replay URL"),
        ({"concurrency": 0}, "concurrency must be greater than zero"),
        ({"requests": -1}, "requests cannot be negative"),
        ({"timeout": 0}, "timeout must be greater than zero"),
        ({"max_tokens": 0}, "max_tokens must be greater than zero"),
        ({"model": ""}, "model must not be empty"),
    ],
)
def test_replay_settings_reject_invalid_limits(overrides, reason):
    settings = replay.ReplaySettings(
        **{"url": "https://example.invalid", "model": "chat", **overrides}
    )
    with pytest.raises(ValueError, match=reason):
        settings.validate()


def test_worker_command_quotes_executable_script_and_config(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    config_path = tmp_path / "path with spaces" / "task.py"
    task = make_task(tmp_path)

    command = task.get_command(str(config_path), "runner -- {task_cmd}")

    assert command == f'runner -- "{sys.executable}" "{replay.__file__}" "{config_path}"'


def test_run_loads_and_repairs_records_and_persists_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "trace input.jsonl"
    repaired = r'{"messages":[{"role":"user","content":"\q"}],"threshold":1[ID_a]2}'
    conversation = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "ok"}]
    source.write_text(
        "\n".join([
            "bad outer JSON",
            json.dumps({"payload": repaired, "request_id": "repaired"}),
            json.dumps({"payload": {"messages": conversation}, "request_id": "intact"}),
            json.dumps({"payload": {"messages": conversation}, "request_id": "beyond-limit"}),
        ]),
        encoding="utf-8",
    )
    monkeypatch.setenv("AISBENCH_REPLAY_LOG_FILE", "not-the-configured-path.jsonl")
    monkeypatch.setenv("AISBENCH_REPLAY_MAX_RECORDS", "2")
    monkeypatch.setenv("AISBENCH_REPLAY_REPAIR_REDACTED", " YeS ")
    task = make_task(
        tmp_path,
        {"abbr": "trace", "args": {"input_log_file": source.name, "on_error": "skip"}},
        concurrent=3,
        add_timestamp_prefix=False,
        x_app_key="example-key-do-not-persist",
    )
    manager = object()
    details = [
        {"request_index": 1, "source_request_id": "repaired", "success": True,
         "input_tokens": 9, "output_tokens": 3, "latency": 1},
        {"request_index": 2, "source_request_id": "intact", "success": False,
         "status_code": 429},
    ]

    async def fake_run(records, settings):
        assert task.task_state_manager is manager
        assert settings.concurrency == 3
        assert settings.requests == 0
        assert settings.add_timestamp_prefix is False
        assert [record["source_line"] for record in records] == [2, 3]
        assert [record["request_id"] for record in records] == ["repaired", "intact"]
        assert records[0]["payload"]["messages"][0]["content"] == r"\q"
        assert records[1]["payload"]["messages"] == conversation
        return details, 2.0

    monkeypatch.setattr(task, "_run_requests", fake_run)
    task.run(manager)

    prediction = tmp_path / "predictions" / "model" / "trace.json"
    report_text = prediction.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert report["task"] == "llm_io_replay"
    assert report["dataset"] == "trace"
    assert Path(report["input_log_file"]) == source
    assert report["loaded_records"] == 2
    assert report["repaired_records"] == 1
    assert report["repair_operations"] == 2
    assert report["settings"]["x_app_key"] == "***"
    assert "example-key-do-not-persist" not in report_text
    assert report["details"] == details
    summary = report["summary"]
    assert summary["max_concurrency"] == 3
    assert summary["successful_requests"] == summary["failed_requests"] == 1
    assert summary["total_tokens"] == 12
    assert summary["request_throughput_rps"] == 0.5
    assert summary["errors"]["HTTP 429"]["count"] == 1
    markdown = prediction.with_suffix(".md").read_text(encoding="utf-8")
    assert "HTTP 429" in markdown
    assert "example-key-do-not-persist" not in markdown

    task.display_results()

    terminal = capsys.readouterr().out
    assert "Performance Parameters" in terminal
    assert "Total Token Throughput" in terminal


@pytest.mark.parametrize("contents", ["\n \n", "bad outer JSON\n{}"])
def test_task_refuses_to_schedule_an_empty_or_all_skipped_log(tmp_path, monkeypatch, contents):
    source = tmp_path / "empty.jsonl"
    source.write_text(contents, encoding="utf-8")
    monkeypatch.setenv("AISBENCH_REPLAY_LOG_FILE", str(source))
    task = make_task(tmp_path, {"abbr": "trace", "args": {"on_error": "skip"}})
    run_requests = Mock()
    monkeypatch.setattr(task, "_run_requests", run_requests)

    with pytest.raises(ValueError, match="No replayable requests found") as err:
        task.run()

    assert str(source) in str(err.value)
    run_requests.assert_not_called()
    assert not (tmp_path / "predictions").exists()


def test_task_requires_source_configuration(tmp_path):
    task = make_task(tmp_path)

    with pytest.raises(ValueError, match="input_log_file must not be empty"):
        task.run()


def test_display_and_summary_warn_without_creating_missing_artifacts(tmp_path, capsys):
    task = make_task(tmp_path)
    summarizer = replay.LLMIOReplayPerfSummarizer(
        ConfigDict(models=[task.model_cfg], datasets=task.dataset_cfgs, work_dir=str(tmp_path))
    )
    task.logger = Mock()
    summarizer.logger = Mock()

    task.display_results()
    summarizer.summarize()

    missing = tmp_path / "predictions" / "model" / "trace.json"
    assert Path(task.logger.warning.call_args.args[1]) == missing
    assert Path(summarizer.logger.warning.call_args_list[0].args[1]) == missing
    assert summarizer.logger.warning.call_args.args == ("No LLMIO replay reports were summarized.",)
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "performances").exists()


def test_scheduler_cycles_records_limits_inflight_and_returns_input_order(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    settings = replay.ReplaySettings(
        url="http://example.invalid", model="chat", concurrency=2, requests=5,
        timeout=12, add_timestamp_prefix=False,
    )
    session = object()
    context = AsyncMock()
    context.__aenter__.return_value = session
    connector = object()
    connector_factory = Mock(return_value=connector)
    session_factory = Mock(return_value=context)
    monkeypatch.setattr(replay.aiohttp, "TCPConnector", connector_factory)
    monkeypatch.setattr(replay.aiohttp, "ClientSession", session_factory)
    active = set()
    peak = 0
    finished = []

    async def schedule():
        later_request_started = asyncio.Event()

        async def fake_send(client, current_session, index, record):
            nonlocal peak
            assert client.timestamp_prefix == ""
            assert current_session is session
            active.add(index)
            peak = max(peak, len(active))
            try:
                if index == 1:
                    await later_request_started.wait()
                if index == 3:
                    later_request_started.set()
                finished.append(index)
                return {"request_index": index, "request_id": record["request_id"]}
            finally:
                active.remove(index)

        monkeypatch.setattr(replay.ReplayClient, "send", fake_send)
        records = [{"request_id": "a"}, {"request_id": "b"}]
        return await asyncio.wait_for(task._run_requests(records, settings), timeout=2)

    details, duration = asyncio.run(schedule())

    assert duration >= 0
    assert peak == 2
    assert not active
    assert finished.index(1) > finished.index(3)
    assert [item["request_index"] for item in details] == [1, 2, 3, 4, 5]
    assert [item["request_id"] for item in details] == ["a", "b", "a", "b", "a"]
    connector_factory.assert_called_once_with(limit=2, ssl=False)
    assert session_factory.call_args.kwargs["connector"] is connector
    assert session_factory.call_args.kwargs["timeout"].total == 12
    context.__aexit__.assert_awaited_once()
