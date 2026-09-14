import asyncio

import pytest
from mmengine.config import ConfigDict

from ais_bench.benchmark.tasks.llm_io_replay import (
    LLMIOReplayTask,
    ReplayClient,
    ReplaySettings,
    build_request_headers,
    convert_to_teleagi_format,
    normalize_endpoint,
    parse_payload,
    summarize_results,
)


def test_parse_payload_repairs_only_redacted_numeric_tokens():
    raw = (
        '{"messages":[{"role":"user","content":"keep [PHONE_ab12]"}],'
        '"schema":{"minimum":-9007[PHONE_ab12]1}}'
    )
    payload, repairs = parse_payload(raw, repair_redacted=True)

    assert payload["schema"]["minimum"] == 0
    assert payload["messages"][0]["content"] == "keep [PHONE_ab12]"
    assert repairs == 1


def test_parse_payload_strict_mode_rejects_damaged_json():
    raw = '{"messages":[{"role":"user","content":"x"}],"value":1[PHONE_a]2}'
    with pytest.raises(ValueError):
        parse_payload(raw, repair_redacted=False)


def test_request_conversion_matches_glm_replay_semantics():
    payload = {
        "model": "chat-pro",
        "temperature": 0.2,
        "max_tokens": 65536,
        "tool_stream": True,
        "reasoning_effort": "high",
        "tools": [{"type": "function", "function": {"name": "search"}}],
        "tool_choice": "auto",
        "messages": [
            {"role": "system", "content": "system"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "drop me",
                "tool_calls": [{"id": "call-1", "type": "function"}],
            },
            {"role": "tool", "content": None, "tool_call_id": "call-1"},
        ],
    }

    body = convert_to_teleagi_format(
        payload,
        model="glm51",
        timestamp_prefix="[2026-09-14 10:00:00] ",
        stream=True,
    )

    assert body["model"] == "glm51"
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 65536
    assert body["stream_options"] == {"include_usage": True}
    assert "tool_stream" not in body
    assert "reasoning_effort" not in body
    assert body["messages"][0]["content"][0]["text"].endswith("system")
    assert body["messages"][1]["content"] == []
    assert "reasoning_content" not in body["messages"][1]
    assert body["messages"][2]["content"] == ""


def test_endpoint_and_headers_match_observable_script_behavior():
    url = "http://172.27.13.87:8900/v1/chat/completions"
    assert normalize_endpoint(url) == url
    assert normalize_endpoint("http://172.27.13.87:8900") == url

    headers = build_request_headers(
        x_app_id="1111",
        session_id="session-1",
        send_legacy_app_headers=True,
    )
    assert headers["X-APP-ID"] == "1111"
    assert headers["Authorization"] == "1111"
    assert headers["session-id"] == "session-1"
    assert headers["traceparent"].startswith("00-")


def test_settings_validation_and_summary():
    ReplaySettings(
        url="http://127.0.0.1:8900/v1/chat/completions",
        model="glm51",
    ).validate()
    details = [
        {
            "success": True,
            "latency": 2.0,
            "ttft": 0.5,
            "input_tokens": 10,
            "output_tokens": 20,
        },
        {
            "success": False,
            "latency": 1.0,
            "ttft": None,
            "input_tokens": 0,
            "output_tokens": 0,
        },
    ]
    summary = summarize_results(details, duration=4.0)
    assert summary["successful_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["request_throughput_rps"] == 0.25
    assert summary["output_token_throughput"] == 5.0


def test_task_reads_runtime_environment_overrides(monkeypatch):
    monkeypatch.setenv("AISBENCH_REPLAY_URL", "http://127.0.0.1:9999/v1/chat/completions")
    monkeypatch.setenv("AISBENCH_REPLAY_CONCURRENCY", "17")
    monkeypatch.setenv("AISBENCH_REPLAY_REQUESTS", "23")
    cfg = ConfigDict(
        models=[
            ConfigDict(
                abbr="model",
                url="http://default.invalid",
                model="glm51",
            )
        ],
        datasets=[[ConfigDict(abbr="data", args=ConfigDict(requests=0))]],
        work_dir="outputs/test",
        cli_args=ConfigDict(mode="infer", debug=True),
    )
    task = LLMIOReplayTask(cfg)
    settings = task._settings(task.dataset_cfgs[0])

    assert settings.url == "http://127.0.0.1:9999/v1/chat/completions"
    assert settings.concurrency == 17
    assert settings.requests == 23


def test_replay_client_parses_framed_sse_without_network():
    class FakeContent:
        def __init__(self, lines):
            self.lines = iter(lines)

        async def readline(self):
            return next(self.lines, b"")

    class FakeResponse:
        status = 200

        def __init__(self):
            self.content = FakeContent(
                [
                    b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
                    b"\n",
                    b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n',
                    b"\n",
                    b"data: [DONE]\n",
                    b"\n",
                ]
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class FakeSession:
        def __init__(self):
            self.request = None

        def post(self, url, json, headers):
            self.request = {"url": url, "json": json, "headers": headers}
            return FakeResponse()

    settings = ReplaySettings(
        url="http://127.0.0.1:8900/v1/chat/completions",
        model="glm51",
        stream=True,
    )
    client = ReplayClient(settings, timestamp_prefix="")
    session = FakeSession()
    record = {
        "payload": {"messages": [{"role": "user", "content": "hello"}]},
        "session_id": "session-1",
        "source_line": 9,
    }

    result = asyncio.run(client.send(session, 1, record))

    assert result["success"] is True
    assert result["input_tokens"] == 3
    assert result["output_tokens"] == 1
    assert result["content_length"] == 2
    assert result["source_line"] == 9
    assert session.request["url"].endswith("/v1/chat/completions")
    assert len(session.request["json"]["messages"]) == 1
