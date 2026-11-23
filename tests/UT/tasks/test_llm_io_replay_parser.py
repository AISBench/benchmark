"""Unit tests for replay log ingestion and conservative payload repair."""

import json

import pytest

from ais_bench.benchmark.tasks.llm_io_replay.parser import (
    LLMIOReplayParseError,
    iter_llm_io_records,
    parse_payload,
)


@pytest.mark.parametrize(
    ("escape", "decoded"),
    [(r"\q", r"\q"), (r"\u12XY", r"\u12XY")],
    ids=["unknown-escape", "malformed-unicode-escape"],
)
def test_repair_preserves_invalid_escapes_as_literal_text(escape, decoded):
    raw = (
        r'{"messages":[{"content":"quote: \"[PHONE_a1]\"; escaped newline: \n; '
        + escape
        + r'"}],"threshold":1.5e-[PHONE_b2]3}'
    )

    payload, repairs = parse_payload(raw, repair_redacted=True)

    assert payload["messages"][0]["content"] == (
        'quote: "[PHONE_a1]"; escaped newline: \n; ' + decoded
    )
    assert payload["threshold"] == 0
    assert repairs == 2


def test_escape_repair_refuses_non_escape_json_errors():
    raw = r'{"messages":[{"content":"hello\q"}],}'

    with pytest.raises(LLMIOReplayParseError, match="invalid payload JSON") as err:
        parse_payload(raw, repair_redacted=True)

    assert isinstance(err.value.__cause__, json.JSONDecodeError)
    assert err.value.__cause__.msg == "Expecting property name enclosed in double quotes"


def test_payload_requiring_too_many_escape_repairs_is_rejected():
    raw = '{"messages":[{"content":"' + r"\q" * 257 + '"}]}'

    with pytest.raises(LLMIOReplayParseError, match="more than 256 escape repairs"):
        parse_payload(raw, repair_redacted=True)


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        (None, "must be a JSON string or object, got NoneType"),
        ("[]", "must decode to a JSON object"),
        ({}, "messages must be a non-empty list"),
        ({"messages": []}, "messages must be a non-empty list"),
        ({"messages": "hello"}, "messages must be a non-empty list"),
    ],
    ids=["missing", "array", "no-messages", "empty-messages", "string-messages"],
)
def test_rejects_invalid_payload_shape(raw, error):
    with pytest.raises(LLMIOReplayParseError, match=error):
        parse_payload(raw)


def test_record_iterator_keeps_full_conversation_and_source_metadata(tmp_path):
    source = tmp_path / "trace.jsonl"
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "早上好"},
    ]
    payload = {"messages": messages, "temperature": 0.4}
    metadata = {
        "request_id": "request-7",
        "trace_id": "trace-9",
        "session_id": 0,
        "message_id": "message-3",
        "user_id": "user-2",
    }
    source.write_text(
        json.dumps({"payload": payload, **metadata}, ensure_ascii=False)
        + "\n \t\n"
        + json.dumps({"payload": json.dumps(payload, ensure_ascii=False)}),
        encoding="utf-8-sig",
    )

    assert list(iter_llm_io_records(source)) == [
        {"payload": payload, **metadata, "source_line": 1, "repair_count": 0},
        {
            "payload": payload,
            **dict.fromkeys(metadata),
            "source_line": 3,
            "repair_count": 0,
        },
    ]


def test_skip_policy_counts_only_yielded_records_toward_limit(tmp_path):
    source = tmp_path / "mixed.jsonl"
    ordinary = {"messages": [{"role": "user", "content": "hello"}]}
    repairable = r'{"messages":[{"content":"\q"}],"n":123[ID_abc]4}'
    source.write_text(
        "\n".join(
            [
                "{bad outer JSON",
                "[]",
                json.dumps({"payload": {"messages": []}}),
                json.dumps({"payload": repairable}),
                "",
                json.dumps({"payload": ordinary}),
                json.dumps({"payload": ordinary}),
            ]
        ),
        encoding="utf-8",
    )

    records = list(
        iter_llm_io_records(source, on_error="skip", repair_redacted=True, max_records=2)
    )

    assert [record["source_line"] for record in records] == [4, 6]
    assert [record["repair_count"] for record in records] == [2, 0]
    assert records[0]["payload"] == {"messages": [{"content": r"\q"}], "n": 0}
    assert records[1]["payload"] == ordinary


@pytest.mark.parametrize(
    ("invalid_entry", "cause", "reason"),
    [
        ("{broken", json.JSONDecodeError, "Expecting property name"),
        ("null", LLMIOReplayParseError, "outer log entry is not an object"),
        ('{"payload": {}}', LLMIOReplayParseError, "non-empty list"),
    ],
    ids=["invalid-json", "invalid-entry-shape", "invalid-payload"],
)
def test_raise_policy_includes_path_physical_line_and_cause(
    tmp_path, invalid_entry, cause, reason
):
    source = tmp_path / "trace with spaces.jsonl"
    source.write_text(
        '{"payload":{"messages":[{"content":"ok"}]}}\n\n' + invalid_entry,
        encoding="utf-8",
    )
    records = iter_llm_io_records(str(source))

    assert next(records)["source_line"] == 1
    with pytest.raises(LLMIOReplayParseError) as err:
        next(records)

    assert str(err.value).startswith(f"{source}:3: ")
    assert reason in str(err.value)
    assert isinstance(err.value.__cause__, cause)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"on_error": "ignore"}, "on_error must be 'raise' or 'skip'"),
        ({"max_records": 0}, "max_records must be greater than zero"),
        ({"max_records": -1}, "max_records must be greater than zero"),
    ],
)
def test_record_options_are_checked_before_opening_file(tmp_path, kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        list(iter_llm_io_records(tmp_path / "does-not-exist.jsonl", **kwargs))
