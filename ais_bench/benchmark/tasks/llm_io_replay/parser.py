"""Parser and conservative repair for line-oriented ``llm_io`` logs."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any


REDACTION_PLACEHOLDER = re.compile(r"\[[A-Z][A-Z0-9_]*_[0-9A-Za-z]+\]")
_NUMBER_CHARS = frozenset("+-0123456789.eE")
_REPAIRABLE_ESCAPE_ERRORS = (
    "Invalid \\escape",
    "Invalid \\uXXXX escape",
)


class LLMIOReplayParseError(ValueError):
    """Raised when an ``llm_io`` record cannot become a replay request."""


def _replace_redacted_numbers(raw: str) -> tuple[str, int]:
    """Replace redaction placeholders only outside JSON strings.

    Redaction can turn a number into an invalid token such as
    ``-9007[PHONE_ab12]1``. The entire damaged numeric fragment is replaced
    with zero. Placeholders in prompts and tool descriptions stay unchanged.
    """

    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    repairs = 0

    while index < len(raw):
        char = raw[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue

        match = REDACTION_PLACEHOLDER.match(raw, index)
        if match is None:
            output.append(char)
            index += 1
            continue

        while output and output[-1] in _NUMBER_CHARS:
            output.pop()
        end = match.end()
        while end < len(raw) and raw[end] in _NUMBER_CHARS:
            end += 1
        output.append("0")
        index = end
        repairs += 1

    return "".join(output), repairs


def _loads_with_escape_repair(raw: str, repairs: int) -> tuple[Any, int]:
    """Decode JSON, doubling only a backslash reported as invalid."""

    candidate = raw
    for _ in range(256):
        try:
            return json.loads(candidate), repairs
        except json.JSONDecodeError as error:
            if not error.msg.startswith(_REPAIRABLE_ESCAPE_ERRORS):
                raise
            candidate = candidate[: error.pos] + "\\" + candidate[error.pos :]
            repairs += 1
    raise LLMIOReplayParseError("payload requires more than 256 escape repairs")


def parse_payload(
    raw_payload: str | dict[str, Any],
    *,
    repair_redacted: bool = False,
) -> tuple[dict[str, Any], int]:
    """Parse one payload and return ``(payload, repair_count)``."""

    if isinstance(raw_payload, dict):
        payload = raw_payload
        repairs = 0
    elif isinstance(raw_payload, str):
        try:
            if repair_redacted:
                candidate, repairs = _replace_redacted_numbers(raw_payload)
                payload, repairs = _loads_with_escape_repair(candidate, repairs)
            else:
                payload = json.loads(raw_payload)
                repairs = 0
        except (json.JSONDecodeError, LLMIOReplayParseError) as error:
            raise LLMIOReplayParseError(f"invalid payload JSON: {error}") from error
    else:
        raise LLMIOReplayParseError(
            "payload must be a JSON string or object, got "
            f"{type(raw_payload).__name__}"
        )

    if not isinstance(payload, dict):
        raise LLMIOReplayParseError("payload must decode to a JSON object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise LLMIOReplayParseError("payload.messages must be a non-empty list")
    return payload, repairs


def iter_llm_io_records(
    path: str | Path,
    *,
    repair_redacted: bool = False,
    on_error: str = "raise",
    max_records: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield complete replay records without expanding their message turns."""

    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be greater than zero")

    loaded = 0
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                outer = json.loads(line)
                if not isinstance(outer, dict):
                    raise LLMIOReplayParseError("outer log entry is not an object")
                payload, repair_count = parse_payload(
                    outer.get("payload"), repair_redacted=repair_redacted
                )
                record = {
                    "payload": payload,
                    "session_id": outer.get("session_id"),
                    "request_id": outer.get("request_id"),
                    "trace_id": outer.get("trace_id"),
                    "message_id": outer.get("message_id"),
                    "user_id": outer.get("user_id"),
                    "source_line": line_number,
                    "repair_count": repair_count,
                }
            except (json.JSONDecodeError, LLMIOReplayParseError) as error:
                if on_error == "skip":
                    continue
                raise LLMIOReplayParseError(
                    f"{source}:{line_number}: {error}"
                ) from error

            yield record
            loaded += 1
            if max_records is not None and loaded >= max_records:
                break

