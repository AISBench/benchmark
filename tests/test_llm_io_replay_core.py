import asyncio

import pytest
from mmengine.config import ConfigDict

from ais_bench.benchmark.tasks.llm_io_replay import (
    LLMIOReplayTask,
    ReplayClient,
    ReplaySettings,
    build_markdown_report,
    build_request_headers,
    convert_to_teleagi_format,
    format_terminal_report,
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


def test_complete_summary_matches_script_and_aisbench_metrics():
    details = [
        {
            "success": True,
            "latency": 2.0,
            "ttft": 0.5,
            "input_tokens": 10,
            "output_tokens": 20,
            "reasoning_tokens": 5,
            "prompt_cache_hit_tokens": 2,
            "request_body_size": 100,
            "token_source": "api",
            "status_code": 200,
            "error": "",
        },
        {
            "success": True,
            "latency": 4.0,
            "ttft": 1.0,
            "input_tokens": 30,
            "output_tokens": 10,
            "reasoning_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "request_body_size": 300,
            "token_source": "estimated_prompt",
            "status_code": 200,
            "error": "",
        },
        {
            "success": False,
            "latency": 0.2,
            "ttft": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "request_body_size": 50,
            "token_source": "none",
            "status_code": 429,
            "error": '{"error":"rate limit"}',
        },
    ]

    summary = summarize_results(details, duration=5.0, max_concurrency=100)

    assert summary["successful_requests"] == 2
    assert summary["failed_requests"] == 1
    assert summary["request_throughput_rps"] == 0.4
    assert summary["request_throughput_rpm"] == 24.0
    assert summary["average_concurrency"] == 1.2
    assert summary["max_concurrency"] == 100
    assert summary["total_input_tokens"] == 40
    assert summary["total_output_tokens"] == 30
    assert summary["total_tokens"] == 70
    assert summary["input_token_throughput"] == 8.0
    assert summary["output_token_throughput"] == 6.0
    assert summary["total_token_throughput"] == 14.0
    assert summary["latency_seconds"]["min"] == 2.0
    assert summary["latency_seconds"]["max"] == 4.0
    assert summary["latency_seconds"]["p75"] == 3.5
    assert summary["latency_seconds"]["n"] == 2
    assert summary["tps_tokens_per_second"]["mean"] == 6.25
    assert summary["cache"]["hit_requests"] == 1
    assert summary["cache"]["miss_requests"] == 1
    assert summary["cache"]["hit_rate_percent"] == 5.0
    assert summary["token_sources"] == {"api": 1, "estimated_prompt": 1}
    assert summary["errors"]["HTTP 429"]["count"] == 1
    assert summary["errors"]["HTTP 429"]["percentage"] == 100.0


def test_terminal_and_markdown_reports_include_complete_metrics():
    details = [
        {
            "request_index": 1,
            "success": True,
            "status_code": 200,
            "error": "",
            "latency": 2.0,
            "ttft": 0.5,
            "input_tokens": 10,
            "output_tokens": 20,
            "reasoning_tokens": 5,
            "prompt_cache_hit_tokens": 2,
            "cache_hit": True,
            "request_body_size": 1024,
            "token_source": "api",
            "traceparent": "00-abc-def-01",
        },
        {
            "request_index": 2,
            "success": False,
            "status_code": 429,
            "error": "rate limit",
            "latency": 0.2,
            "ttft": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "cache_hit": False,
            "request_body_size": 512,
            "token_source": "none",
            "traceparent": "00-ghi-jkl-01",
        },
    ]
    report = {
        "task": "llm_io_replay",
        "dataset": "llm-io-replay",
        "input_log_file": "/data/replay_fix.txt",
        "loaded_records": 2500,
        "settings": {
            "url": "http://127.0.0.1:8005/v1/chat/completions",
            "model": "qwen3.6",
            "concurrency": 100,
            "requests": 2,
            "timeout": 900,
            "temperature": 0.7,
            "stream": True,
        },
        "summary": summarize_results(details, 5.0, max_concurrency=100),
        "details": details,
    }

    terminal = format_terminal_report(report)
    markdown = build_markdown_report(report)

    assert "Performance Parameters" in terminal
    assert "Common Metric" in terminal
    assert "E2EL" in terminal
    assert "Benchmark Duration" in terminal
    assert "HTTP 429" in terminal
    for heading in (
        "测试概览",
        "性能参数统计",
        "端到端汇总指标",
        "Token统计来源",
        "缓存命中统计",
        "错误分析",
        "成功请求详情",
        "失败请求详情",
    ):
        assert heading in markdown


def test_terminal_report_upgrades_legacy_summary_from_details():
    details = [
        {
            "request_index": 1,
            "success": True,
            "status_code": 200,
            "error": "",
            "latency": 2.0,
            "ttft": 0.5,
            "input_tokens": 10,
            "output_tokens": 20,
            "request_body_size": 100,
        }
    ]
    report = {
        "summary": {
            "duration_seconds": 2.0,
            "latency_seconds": {"mean": 2.0, "p50": 2.0},
        },
        "settings": {"concurrency": 100},
        "details": details,
    }

    terminal = format_terminal_report(report)

    assert "Performance Parameters" in terminal
    assert "Max Concurrency" in terminal


def test_task_prefers_config_to_runtime_environment(monkeypatch):
    monkeypatch.setenv("AISBENCH_REPLAY_URL", "http://127.0.0.1:9999/v1/chat/completions")
    monkeypatch.setenv("AISBENCH_REPLAY_CONCURRENCY", "17")
    monkeypatch.setenv("AISBENCH_REPLAY_REQUESTS", "23")
    cfg = ConfigDict(
        models=[
            ConfigDict(
                abbr="model",
                url="http://default.invalid",
                model="glm51",
                concurrent=100,
            )
        ],
        datasets=[[ConfigDict(abbr="data", args=ConfigDict(requests=2500))]],
        work_dir="outputs/test",
        cli_args=ConfigDict(mode="infer", debug=True),
    )
    task = LLMIOReplayTask(cfg)
    settings = task._settings(task.dataset_cfgs[0])

    assert settings.url == "http://default.invalid"
    assert settings.concurrency == 100
    assert settings.requests == 2500


def test_task_uses_runtime_environment_as_missing_config_fallback(monkeypatch):
    monkeypatch.setenv("AISBENCH_REPLAY_URL", "http://127.0.0.1:9999/v1/chat/completions")
    monkeypatch.setenv("AISBENCH_REPLAY_CONCURRENCY", "17")
    monkeypatch.setenv("AISBENCH_REPLAY_REQUESTS", "23")
    cfg = ConfigDict(
        models=[ConfigDict(abbr="model", model="glm51")],
        datasets=[[ConfigDict(abbr="data", args=ConfigDict())]],
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


def test_replay_client_collects_extended_usage_metrics():
    class FakeContent:
        def __init__(self):
            self.lines = iter(
                [
                    b'data: {"choices":[{"delta":{"content":"okay"}}]}\n',
                    b"\n",
                    (
                        b'data: {"choices":[],"usage":{"total_tokens":9,'
                        b'"prompt_tokens":5,"completion_tokens":4,'
                        b'"completion_tokens_details":{"reasoning_tokens":2},'
                        b'"prompt_tokens_details":{"cached_tokens":3}}}\n'
                    ),
                    b"\n",
                    b"data: [DONE]\n",
                    b"\n",
                ]
            )

        async def readline(self):
            return next(self.lines, b"")

    class FakeResponse:
        status = 200
        content = FakeContent()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class FakeSession:
        def post(self, url, json, headers):
            return FakeResponse()

    settings = ReplaySettings(
        url="http://127.0.0.1:8900/v1/chat/completions",
        model="glm51",
    )
    result = asyncio.run(
        ReplayClient(settings, timestamp_prefix="").send(
            FakeSession(),
            1,
            {
                "payload": {"messages": [{"role": "user", "content": "hi"}]},
                "source_line": 1,
            },
        )
    )

    assert result["input_tokens"] == 5
    assert result["output_tokens"] == 4
    assert result["total_tokens"] == 9
    assert result["reasoning_tokens"] == 2
    assert result["prompt_cache_hit_tokens"] == 3
    assert result["cache_hit"] is True
    assert result["token_source"] == "api"
    assert result["tokens_per_second"] > 0
    assert result["typing_speed"] >= 0
