"""Exercise replay request conversion and response failures without networking."""

import asyncio
import copy
import json
import re
from types import SimpleNamespace

import aiohttp
import pytest

from ais_bench.benchmark.tasks.llm_io_replay import llm_io_replay as replay


class Response:
    def __init__(self, *, body="", status=200, lines=()):
        self.status = status
        self.body = body
        self.content = self
        self.lines = iter(lines)
        self.exited = False
        self.text_reads = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.exited = True

    async def text(self):
        self.text_reads += 1
        return self.body

    async def readline(self):
        return next(self.lines, b"")


class Session:
    def __init__(self, response=None, *, error=None):
        self.response = response
        self.error = error
        self.posts = []

    def post(self, url, *, json, headers):
        self.posts.append((url, json, headers))
        if self.error is not None:
            raise self.error
        return self.response


def settings(**overrides):
    return replay.ReplaySettings(
        **{"url": "https://example.invalid/prefix/", "model": "replay", **overrides}
    )


@pytest.fixture
def record():
    return {
        "payload": {"messages": [{"role": "user", "content": "你好"}]},
        "source_line": 17,
        "request_id": "req-old",
        "trace_id": "trace-old",
        "session_id": 0,
        "repair_count": 2,
    }


def test_multimodal_conversion_copies_nested_data_and_prefixes_first_text():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
    payload = {
        "messages": [
            {"content": [image, {"type": "text", "text": 12}, {"type": "text", "text": "later"}]},
            {"role": "user", "content": [image]},
            {"role": "tool", "content": 42, "tool_call_id": "call-1", "name": "calc"},
        ],
        "tools": [{"type": "function", "function": {"name": "calc"}}],
    }
    original = copy.deepcopy(payload)

    converted = replay.convert_to_teleagi_format(
        payload, model="model", timestamp_prefix="[stamp] ", stream=False
    )

    assert converted["messages"] == [
        {"role": "user", "content": [image, {"type": "text", "text": "[stamp] 12"},
                                     {"type": "text", "text": "later"}]},
        {"role": "user", "content": [{"type": "text", "text": "[stamp]"}, image]},
        {"role": "tool", "content": [{"type": "text", "text": "[stamp] 42"}],
         "tool_call_id": "call-1", "name": "calc"},
    ]
    converted["messages"][0]["content"][0]["image_url"]["url"] = "changed"
    converted["tools"][0]["function"]["name"] = "changed"
    assert payload == original
    assert "stream" not in converted
    assert "stream_options" not in converted


@pytest.mark.parametrize(
    ("messages", "error", "reason"),
    [([], ValueError, "messages must be a non-empty list"),
     (["not an object"], TypeError, "item must be an object")],
)
def test_conversion_rejects_invalid_message_shapes(messages, error, reason):
    with pytest.raises(error, match=reason):
        replay.convert_to_teleagi_format({"messages": messages}, model="model")


def test_nonstream_request_metadata_and_latency_metrics(record, monkeypatch):
    reply = {
        "choices": [{"message": {
            "reasoning": "think",
            "content": "done",
            "tool_calls": [{"function": {"name": "calc", "arguments": "{}"}},
                           {"function": {"name": "lookup"}}],
        }}],
        "output": {"text": "fallback"},
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 20},
    }
    response = Response(body=json.dumps(reply))
    session = Session(response)
    clock = iter([10.0, 12.0, 16.0])
    monkeypatch.setattr(replay, "time", SimpleNamespace(time=lambda: 100, perf_counter=clock.__next__))
    client = replay.ReplayClient(settings(stream=False, x_app_id="app"), "[stamp] ")

    result = asyncio.run(client.send(session, 3, record))

    [(url, body, headers)] = session.posts
    assert url == "https://example.invalid/prefix/v1/chat/completions"
    assert body["model"] == "replay"
    assert body["timestamp"] == 100
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "[stamp] 你好"}]}]
    assert headers["session-id"] == "0"
    assert headers["Authorization"] == "app"
    assert re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-01", result["traceparent"])
    assert headers["traceparent"] == result["traceparent"]
    assert response.exited and response.text_reads == 1
    assert result["success"] is True
    assert result["error"] == ""
    assert result["status_code"] == 200
    assert result["request_index"] == 3
    assert result["source_line"] == 17
    assert result["source_request_id"] == "req-old"
    assert result["source_trace_id"] == "trace-old"
    assert result["session_id"] == 0
    assert result["repair_count"] == 2
    assert result["request_body_size"] == len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
    assert result["content_length"] == len("thinkdone{}lookupfallback")
    assert result["token_source"] == "api"
    assert result["total_tokens"] == 20
    assert result["latency"] == 6
    assert result["ttft"] == 2
    assert result["tokens_per_second"] == pytest.approx(5 / 6)
    assert result["typing_speed"] == 1.25
    assert result["prefill_token_throughput"] == 6
    assert result["tpot"] == 1


@pytest.mark.parametrize(
    ("usage", "text", "expected_source", "input_tokens", "output_tokens"),
    [
        ({}, "x", "estimated_all", None, 1),
        ({"prompt_tokens": 0}, "hello", "estimated_output", 0, 2),
        ({"completion_tokens": 0}, "hello", "estimated_prompt", None, 0),
        ({"prompt_tokens": None, "completion_tokens": None}, "", "estimated_all", None, 0),
    ],
    ids=["missing-both", "explicit-zero-prompt", "explicit-zero-output", "null-usage-empty-text"],
)
def test_nonstream_estimates_only_missing_usage(
    record, usage, text, expected_source, input_tokens, output_tokens
):
    reply = {"choices": [{"message": {"content": text}}], "usage": usage}
    response = Response(body=json.dumps(reply))
    client = replay.ReplayClient(settings(stream=False), "")

    result = asyncio.run(client.send(Session(response), 1, record))

    if input_tokens is None:
        input_tokens = result["request_body_size"] // 2
    assert result["success"] is True
    assert result["input_tokens"] == input_tokens
    assert result["output_tokens"] == output_tokens
    assert result["total_tokens"] == input_tokens + output_tokens
    assert result["token_source"] == expected_source
    assert result["content_length"] == len(text)
    if output_tokens < 2:
        assert result["tpot"] is None


def test_http_error_is_truncated_and_does_not_enter_stream_parser(record):
    response = Response(status=429, body="a" * 4000 + "private suffix", lines=[b"bad"])
    client = replay.ReplayClient(settings(), "")

    result = asyncio.run(client.send(Session(response), 1, record))

    assert response.exited and response.text_reads == 1
    assert next(response.lines) == b"bad"
    assert result["status_code"] == 429
    assert result["success"] is False
    assert result["error"] == "a" * 4000
    assert result["ttft"] is None
    assert result["token_source"] == "none"
    assert result["total_tokens"] == result["content_length"] == 0


@pytest.mark.parametrize(
    ("error", "label", "message"),
    [
        (TimeoutError("deadline"), "Timeout", "TimeoutError: deadline"),
        (aiohttp.ClientConnectionError("reset"), "Connection Error", "ClientConnectionError: reset"),
        (RuntimeError("unexpected"), "RuntimeError", "RuntimeError: unexpected"),
    ],
    ids=["timeout", "connection", "unexpected"],
)
def test_transport_errors_remain_individual_failed_results(record, error, label, message):
    client = replay.ReplayClient(settings(), "")

    result = asyncio.run(client.send(Session(error=error), 5, record))

    assert result["request_index"] == 5
    assert result["source_request_id"] == "req-old"
    assert result["status_code"] == 0
    assert result["success"] is False
    assert result["error"] == message
    assert result["ttft"] is None
    assert result["token_source"] == "none"
    assert replay.summarize_results([result], 1)["errors"] == {
        label: {"count": 1, "percentage": 100, "request_percentage": 100}
    }


@pytest.mark.parametrize("lines", [[], [b": keepalive\r\n", b"data: [DONE]\r\n", b"\r\n"]])
def test_stream_without_data_events_is_not_a_success(record, lines):
    response = Response(lines=lines)
    client = replay.ReplayClient(settings(), "")

    result = asyncio.run(client.send(Session(response), 1, record))

    assert response.exited and response.text_reads == 0
    assert result["status_code"] == 200
    assert result["success"] is False
    assert result["error"] == "empty response stream"
    assert result["ttft"] is None
    assert result["token_source"] == "none"


def test_invalid_sse_after_valid_data_preserves_partial_usage_but_not_success(record):
    response = Response(lines=[
        b'data: {"choices":[{"delta":{"content":"ok"}}],"usage":{"prompt_tokens":7}}\n',
        b"\n",
        b"data: invalid\n",
    ])
    client = replay.ReplayClient(settings(), "")

    result = asyncio.run(client.send(Session(response), 1, record))

    assert response.exited
    assert result["status_code"] == 200
    assert result["success"] is False
    assert result["error"] == "ValueError: invalid SSE JSON event: invalid"
    assert result["ttft"] is not None
    assert result["content_length"] == 2
    assert result["input_tokens"] == result["total_tokens"] == 7
    assert result["output_tokens"] == 0
    assert result["token_source"] == "none"


def test_stream_supports_raw_json_lines_interleaved_with_sse(record):
    response = Response(lines=[
        b'data: {"choices":[{"delta":{"content":"left"}}]}\r\n',
        b'{"choices":[{"delta":{"content":"right"}}]}\r\n',
        b'data: {"usage":{"prompt_tokens":5,"completion_tokens":3}}\r\n',
    ])
    client = replay.ReplayClient(settings(), "")

    result = asyncio.run(client.send(Session(response), 1, record))

    assert response.exited
    assert result["success"] is True
    assert result["content_length"] == len("leftright")
    assert result["total_tokens"] == 8
    assert result["token_source"] == "api"
