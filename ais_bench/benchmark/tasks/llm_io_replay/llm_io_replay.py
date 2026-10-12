"""AISBench task for replaying complete ``llm_io`` chat requests.

The task follows the custom ``BaseTask`` lifecycle used by ``OneIGEvalTask``:
the runner creates a task config, ``run`` owns the complete workflow, and the
task writes its JSON/Markdown result into the AISBench work directory.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import os.path as osp
import statistics
import sys
import threading
import time
import urllib.parse
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime

import aiohttp
import mmengine
import tabulate
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.tasks.llm_io_replay.parser import iter_llm_io_records
from ais_bench.benchmark.utils.core.abbr import (
    dataset_abbr_from_cfg,
    get_infer_output_path,
    model_abbr_from_cfg,
    task_abbr_from_cfg,
)
from ais_bench.benchmark.utils.logging import AISLogger


@dataclass(frozen=True)
class ReplaySettings:
    url: str
    model: str
    concurrency: int = 100
    requests: int = 0
    timeout: float = 900
    temperature: float = 0.7
    max_tokens: int | None = None
    ignore_eos: bool | None = None
    stream: bool = True
    x_app_id: str = ""
    x_app_key: str = ""
    send_legacy_app_headers: bool = True
    add_timestamp_prefix: bool = True

    def validate(self) -> None:
        parsed = urllib.parse.urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid replay URL: {self.url!r}")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be greater than zero")
        if self.requests < 0:
            raise ValueError("requests cannot be negative")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if not self.model:
            raise ValueError("model must not be empty")


def generate_traceparent() -> str:
    trace_id = uuid.uuid4().hex
    parent_id = uuid.uuid4().hex[:16]
    return f"00-{trace_id}-{parent_id}-01"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _config_or_env(config, key: str, env_name: str, default):
    """Read config first, retaining environment variables as a legacy fallback."""

    if key in config and config.get(key) is not None:
        return config.get(key)
    return os.getenv(env_name, default)


def _config_bool_or_env(
    config,
    key: str,
    env_name: str,
    default: bool,
) -> bool:
    if key in config and config.get(key) is not None:
        return bool(config.get(key))
    return _env_bool(env_name, default)


def _config_optional_bool_or_env(
    config,
    key: str,
    env_name: str,
) -> bool | None:
    if key in config and config.get(key) is not None:
        return bool(config.get(key))
    value = os.getenv(env_name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_endpoint(url: str) -> str:
    """Keep an exact chat-completions URL or append the endpoint once."""

    clean_url = url.rstrip("/")
    if clean_url.endswith("/v1/chat/completions"):
        return clean_url
    return urllib.parse.urljoin(clean_url + "/", "v1/chat/completions")


def _convert_content(message: dict, timestamp_prefix: str) -> object:
    content = message.get("content")
    if content is None:
        return "" if message.get("role") == "tool" else []
    if isinstance(content, str):
        return [{"type": "text", "text": timestamp_prefix + content}]
    if isinstance(content, list):
        converted = copy.deepcopy(content)
        for item in converted:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and "text" in item
            ):
                item["text"] = timestamp_prefix + str(item["text"])
                break
        else:
            converted.insert(0, {"type": "text", "text": timestamp_prefix.strip()})
        return converted
    return [{"type": "text", "text": timestamp_prefix + str(content)}]


def convert_to_teleagi_format(
    log_payload: dict,
    *,
    model: str,
    default_temperature: float = 0.7,
    max_tokens: int | None = None,
    ignore_eos: bool | None = None,
    timestamp_prefix: str = "",
    stream: bool = True,
) -> dict:
    """Mirror ``glm51_replay_v3._convert_to_teleagi_format``."""

    messages = log_payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("replay payload.messages must be a non-empty list")

    converted_messages = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("every payload.messages item must be an object")
        converted = {
            "role": message.get("role", "user"),
            "content": _convert_content(message, timestamp_prefix),
        }
        for key in ("tool_calls", "tool_call_id", "name"):
            if key in message:
                converted[key] = copy.deepcopy(message[key])
        converted_messages.append(converted)

    request_body = {
        "traceId": uuid.uuid4().hex.upper(),
        "timestamp": int(time.time()),
        "model": model,
        "n": 1,
        "temperature": log_payload.get("temperature", default_temperature),
        "messages": converted_messages,
    }
    for key in ("tools", "tool_choice", "max_tokens", "ignore_eos"):
        if key in log_payload:
            request_body[key] = copy.deepcopy(log_payload[key])
    if max_tokens is not None:
        request_body["max_tokens"] = max_tokens
    if ignore_eos is not None:
        request_body["ignore_eos"] = ignore_eos
    if stream:
        request_body["stream"] = True
        request_body["stream_options"] = {"include_usage": True}
    return request_body


def build_request_headers(
    *,
    x_app_id: str,
    session_id: object,
    send_legacy_app_headers: bool,
) -> dict[str, str]:
    """Build the observable headers sent by the original replay script.

    ``glm51_replay_v3`` computes an HMAC and then discards it, assigning the
    app id to ``Authorization``. ``x_app_key`` is therefore intentionally not
    part of this function.
    """

    headers = {"Content-Type": "application/json"}
    if send_legacy_app_headers and x_app_id:
        headers["X-APP-ID"] = x_app_id
        headers["Authorization"] = x_app_id
    if session_id is not None:
        headers["session-id"] = str(session_id)
    headers["traceparent"] = generate_traceparent()
    return headers


async def iter_sse_data(content):
    """Yield complete SSE data fields without relying on TCP chunk boundaries."""

    data_lines: list[str] = []
    while True:
        raw_line = await content.readline()
        if not raw_line:
            if data_lines:
                yield "\n".join(data_lines)
            return
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        if data_lines:
            yield "\n".join(data_lines)
            data_lines.clear()
        yield line


def _extract_response_text(event: dict) -> str:
    fragments: list[str] = []
    for choice in event.get("choices", []):
        message = choice.get("delta") or choice.get("message") or {}
        for key in ("reasoning_content", "reasoning", "content"):
            value = message.get(key)
            if value:
                fragments.append(str(value))
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            if function.get("arguments"):
                fragments.append(str(function["arguments"]))
            elif function.get("name"):
                fragments.append(str(function["name"]))
    fallback = event.get("output")
    if isinstance(fallback, dict) and fallback.get("text"):
        fragments.append(str(fallback["text"]))
    return "".join(fragments)


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: list[float]) -> dict:
    values = [float(value) for value in values if value is not None]
    return {
        "mean": statistics.fmean(values) if values else None,
        "max": max(values) if values else None,
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _error_type(item: dict) -> str:
    status_code = int(item.get("status_code") or 0)
    if status_code:
        return f"HTTP {status_code}"
    error = str(item.get("error") or "")
    lowered = error.lower()
    if "timeout" in lowered:
        return "Timeout"
    if "clientconnector" in lowered or "connection" in lowered:
        return "Connection Error"
    if ":" in error:
        return error.split(":", 1)[0]
    return error or "Unknown Error"


def summarize_results(
    details: list[dict],
    duration: float,
    max_concurrency: int = 0,
) -> dict:
    """Aggregate standalone-script and AISBench performance metrics."""

    successes = [item for item in details if item.get("success")]
    failures = [item for item in details if not item.get("success")]
    total_input_tokens = sum(int(item.get("input_tokens") or 0) for item in successes)
    total_output_tokens = sum(
        int(item.get("output_tokens") or 0) for item in successes
    )
    total_tokens = sum(
        int(item.get("total_tokens") or 0)
        or int(item.get("input_tokens") or 0)
        + int(item.get("output_tokens") or 0)
        for item in successes
    )

    latencies = [float(item.get("latency") or 0) for item in successes]
    ttfts = [float(item["ttft"]) for item in successes if item.get("ttft") is not None]
    tpot_values = []
    tps_values = []
    typing_values = []
    prefill_values = []
    for item in successes:
        latency = float(item.get("latency") or 0)
        ttft = item.get("ttft")
        ttft_value = float(ttft) if ttft is not None else 0.0
        output_tokens = int(item.get("output_tokens") or 0)
        input_tokens = int(item.get("input_tokens") or 0)
        generation_time = max(latency - ttft_value, 0.0)
        if latency > 0:
            tps_values.append(output_tokens / latency)
        if generation_time > 0:
            typing_values.append(output_tokens / generation_time)
        if generation_time > 0 and output_tokens > 1:
            tpot_values.append(generation_time / (output_tokens - 1))
        if ttft_value > 0:
            prefill_values.append(input_tokens / ttft_value)

    token_sources: dict[str, int] = {}
    for item in successes:
        source = str(item.get("token_source") or "unknown")
        token_sources[source] = token_sources.get(source, 0) + 1

    error_counts: dict[str, int] = {}
    for item in failures:
        label = _error_type(item)
        error_counts[label] = error_counts.get(label, 0) + 1
    errors = {
        label: {
            "count": count,
            "percentage": count / len(failures) * 100 if failures else 0.0,
            "request_percentage": count / len(details) * 100 if details else 0.0,
        }
        for label, count in sorted(error_counts.items())
    }

    cache_hit_tokens = sum(
        int(item.get("prompt_cache_hit_tokens") or 0) for item in successes
    )
    cache_hit_requests = sum(
        int(item.get("prompt_cache_hit_tokens") or 0) > 0 for item in successes
    )
    total_reasoning_tokens = sum(
        int(item.get("reasoning_tokens") or 0) for item in successes
    )
    total_request_body_size = sum(
        int(item.get("request_body_size") or 0) for item in successes
    )
    request_throughput = len(successes) / duration if duration > 0 else 0.0
    input_throughput = total_input_tokens / duration if duration > 0 else 0.0
    output_throughput = total_output_tokens / duration if duration > 0 else 0.0
    total_throughput = total_tokens / duration if duration > 0 else 0.0

    return {
        "total_requests": len(details),
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "success_rate": len(successes) / len(details) if details else 0.0,
        "duration_seconds": duration,
        "request_throughput_rps": request_throughput,
        "request_throughput_rpm": request_throughput * 60,
        "average_concurrency": sum(latencies) / duration if duration > 0 else 0.0,
        "max_concurrency": max_concurrency,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_request_body_size_bytes": total_request_body_size,
        "input_token_throughput": input_throughput,
        "output_token_throughput": output_throughput,
        "total_token_throughput": total_throughput,
        "latency_seconds": _distribution(latencies),
        "ttft_seconds": _distribution(ttfts),
        "tpot_seconds": _distribution(tpot_values),
        "tps_tokens_per_second": _distribution(tps_values),
        "typing_speed_tokens_per_second": _distribution(typing_values),
        "prefill_token_throughput": _distribution(prefill_values),
        "input_tokens_per_request": _distribution(
            [int(item.get("input_tokens") or 0) for item in successes]
        ),
        "output_tokens_per_request": _distribution(
            [int(item.get("output_tokens") or 0) for item in successes]
        ),
        "total_tokens_per_request": _distribution(
            [
                int(item.get("total_tokens") or 0)
                or int(item.get("input_tokens") or 0)
                + int(item.get("output_tokens") or 0)
                for item in successes
            ]
        ),
        "reasoning_tokens_per_request": _distribution(
            [int(item.get("reasoning_tokens") or 0) for item in successes]
        ),
        "request_body_size_bytes": _distribution(
            [int(item.get("request_body_size") or 0) for item in successes]
        ),
        "cache": {
            "hit_requests": cache_hit_requests,
            "miss_requests": len(successes) - cache_hit_requests,
            "hit_request_rate_percent": (
                cache_hit_requests / len(successes) * 100 if successes else 0.0
            ),
            "hit_tokens": cache_hit_tokens,
            "average_hit_tokens": (
                cache_hit_tokens / cache_hit_requests if cache_hit_requests else 0.0
            ),
            "hit_rate_percent": (
                cache_hit_tokens / total_input_tokens * 100
                if total_input_tokens
                else 0.0
            ),
        },
        "token_sources": token_sources,
        "errors": errors,
    }


_PERFORMANCE_ROWS = (
    ("E2EL", "latency_seconds", "s"),
    ("TTFT", "ttft_seconds", "s"),
    ("TPOT", "tpot_seconds", "s/token"),
    ("TPS", "tps_tokens_per_second", "token/s"),
    ("Typing Speed", "typing_speed_tokens_per_second", "token/s"),
    ("Prefill Throughput", "prefill_token_throughput", "token/s"),
    ("Input Tokens", "input_tokens_per_request", "token"),
    ("Output Tokens", "output_tokens_per_request", "token"),
    ("Total Tokens", "total_tokens_per_request", "token"),
    ("Reasoning Tokens", "reasoning_tokens_per_request", "token"),
    ("Request Body Size", "request_body_size_bytes", "byte"),
)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _report_summary(report: dict) -> dict:
    """Return a complete summary, upgrading legacy replay reports in memory."""

    summary = report.get("summary") or {}
    latency = summary.get("latency_seconds") or {}
    required = {
        "request_throughput_rpm",
        "average_concurrency",
        "tpot_seconds",
        "total_reasoning_tokens",
        "total_request_body_size_bytes",
        "cache",
        "errors",
    }
    cache = summary.get("cache") or {}
    if (
        required.issubset(summary)
        and {"miss_requests", "average_hit_tokens"}.issubset(cache)
        and "min" in latency
        and "p75" in latency
    ):
        return summary
    settings = report.get("settings") or {}
    return summarize_results(
        report.get("details") or [],
        float(summary.get("duration_seconds") or 0),
        max_concurrency=int(settings.get("concurrency") or 0),
    )


def format_terminal_report(report: dict) -> str:
    """Render replay metrics with AISBench's native ``fancy_grid`` style."""

    summary = _report_summary(report)
    performance = [[
        "Stage",
        "Unit",
        "Average",
        "Max",
        "Min",
        "Median",
        "P75",
        "P90",
        "P95",
        "P99",
        "N",
    ]]
    for label, key, unit in _PERFORMANCE_ROWS:
        values = summary[key]
        performance.append(
            [
                label,
                unit,
                values["mean"],
                values["max"],
                values["min"],
                values["p50"],
                values["p75"],
                values["p90"],
                values["p95"],
                values["p99"],
                values["n"],
            ]
        )

    common = [
        ["Common Metric", "Value", "Unit"],
        ["Benchmark Duration", summary["duration_seconds"], "s"],
        ["Total Requests", summary["total_requests"], "request"],
        ["Successful Requests", summary["successful_requests"], "request"],
        ["Failed Requests", summary["failed_requests"], "request"],
        ["Success Rate", summary["success_rate"] * 100, "%"],
        ["Average Concurrency", summary["average_concurrency"], "request"],
        ["Max Concurrency", summary["max_concurrency"], "request"],
        ["Request Throughput", summary["request_throughput_rps"], "request/s"],
        ["Request Throughput", summary["request_throughput_rpm"], "request/min"],
        ["Total Input Tokens", summary["total_input_tokens"], "token"],
        ["Total Output Tokens", summary["total_output_tokens"], "token"],
        ["Total Tokens", summary["total_tokens"], "token"],
        ["Total Reasoning Tokens", summary["total_reasoning_tokens"], "token"],
        [
            "Total Request Body Size",
            summary["total_request_body_size_bytes"],
            "byte",
        ],
        ["Input Token Throughput", summary["input_token_throughput"], "token/s"],
        ["Output Token Throughput", summary["output_token_throughput"], "token/s"],
        ["Total Token Throughput", summary["total_token_throughput"], "token/s"],
    ]
    options = dict(
        headers="firstrow",
        tablefmt="fancy_grid",
        floatfmt=".3f",
        numalign="center",
        stralign="left",
        missingval="N/A",
    )
    sections = [
        "Performance Parameters\n" + tabulate.tabulate(performance, **options),
        "Common Metric\n" + tabulate.tabulate(common, **options),
    ]
    return "\n\n".join(sections)


def _markdown_distribution_table(summary: dict) -> list[str]:
    lines = [
        "| 指标 | 单位 | 平均 | 最小 | 最大 | 中位数 | P75 | P90 | P95 | P99 | 标准差 | N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key, unit in _PERFORMANCE_ROWS:
        values = summary[key]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    unit,
                    _fmt(values["mean"]),
                    _fmt(values["min"]),
                    _fmt(values["max"]),
                    _fmt(values["p50"]),
                    _fmt(values["p75"]),
                    _fmt(values["p90"]),
                    _fmt(values["p95"]),
                    _fmt(values["p99"]),
                    _fmt(values["stddev"]),
                    _fmt(values["n"], 0),
                ]
            )
            + " |"
        )
    return lines


def build_markdown_report(report: dict) -> str:
    """Build the complete persisted report without performing file I/O."""

    settings = report["settings"]
    summary = _report_summary(report)
    cache = summary["cache"]
    lines = [
        "# AISBench llm_io 回放压测报告",
        "",
        "## 测试概览",
        "",
        f"- 输入文件：`{report['input_log_file']}`",
        f"- 数据集：`{report['dataset']}`",
        f"- 加载记录数：{report['loaded_records']}",
        f"- 请求数：{summary['total_requests']}",
        f"- 成功/失败：{summary['successful_requests']} / {summary['failed_requests']}",
        f"- 成功率：{summary['success_rate'] * 100:.2f}%",
        "",
        "## 测试配置",
        "",
        f"- Endpoint：`{settings.get('url', '')}`",
        f"- Model：`{settings.get('model', '')}`",
        f"- 并发：{settings.get('concurrency', 0)}",
        f"- 超时：{settings.get('timeout', 0)} 秒",
        f"- 流式：{settings.get('stream', True)}",
        "",
        "## 性能参数统计",
        "",
        *_markdown_distribution_table(summary),
        "",
        "## 端到端汇总指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| Benchmark Duration (s) | {_fmt(summary['duration_seconds'])} |",
        f"| Request Throughput (req/s) | {_fmt(summary['request_throughput_rps'])} |",
        f"| Request Throughput (req/min) | {_fmt(summary['request_throughput_rpm'])} |",
        f"| Average Concurrency | {_fmt(summary['average_concurrency'])} |",
        f"| Max Concurrency | {summary['max_concurrency']} |",
        f"| Total Input Tokens | {summary['total_input_tokens']} |",
        f"| Total Output Tokens | {summary['total_output_tokens']} |",
        f"| Total Tokens | {summary['total_tokens']} |",
        f"| Total Reasoning Tokens | {summary['total_reasoning_tokens']} |",
        f"| Total Request Body Size (bytes) | {summary['total_request_body_size_bytes']} |",
        f"| Input Token Throughput (token/s) | {_fmt(summary['input_token_throughput'])} |",
        f"| Output Token Throughput / OTPS (token/s) | {_fmt(summary['output_token_throughput'])} |",
        f"| Total Token Throughput (token/s) | {_fmt(summary['total_token_throughput'])} |",
        "",
        "## Token统计来源",
        "",
        "| 来源 | 请求数 |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {source} | {count} |"
        for source, count in summary["token_sources"].items()
    )
    lines.extend(
        [
            "",
            "## 缓存命中统计",
            "",
            f"- 命中请求数：{cache['hit_requests']}",
            f"- 未命中请求数：{cache['miss_requests']}",
            f"- 请求命中率：{cache['hit_request_rate_percent']:.2f}%",
            f"- 命中 Token 数：{cache['hit_tokens']}",
            f"- 平均每次命中 Token 数：{cache['average_hit_tokens']:.2f}",
            f"- Prompt Token 命中率：{cache['hit_rate_percent']:.2f}%",
            "",
            "## 错误分析",
            "",
            "| 错误类型 | 数量 | 占失败请求 | 占全部请求 |",
            "|---|---:|---:|---:|",
        ]
    )
    if summary["errors"]:
        lines.extend(
            f"| {label} | {values['count']} | {values['percentage']:.2f}% | "
            f"{values['request_percentage']:.2f}% |"
            for label, values in summary["errors"].items()
        )
    else:
        lines.append("| 无 | 0 | 0.00% | 0.00% |")

    for success, heading in ((True, "成功请求详情"), (False, "失败请求详情")):
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| # | HTTP | TTFT(s) | E2EL(s) | Body(KB) | Input | Output | Reasoning | Cache | Total | TPS | 打字率 | Token来源 | Trace | 错误 |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
            ]
        )
        rows = [item for item in report["details"] if bool(item.get("success")) is success]
        if not rows:
            lines.append(
                "| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |"
            )
        for item in rows:
            error = str(item.get("error") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("request_index", "")),
                        str(item.get("status_code", "")),
                        _fmt(item.get("ttft")),
                        _fmt(item.get("latency")),
                        _fmt((item.get("request_body_size") or 0) / 1024),
                        str(item.get("input_tokens", 0)),
                        str(item.get("output_tokens", 0)),
                        str(item.get("reasoning_tokens", 0)),
                        str(item.get("prompt_cache_hit_tokens", 0)),
                        str(
                            item.get("total_tokens")
                            or int(item.get("input_tokens") or 0)
                            + int(item.get("output_tokens") or 0)
                        ),
                        _fmt(
                            item.get("tokens_per_second")
                            if item.get("tokens_per_second") is not None
                            else (
                                int(item.get("output_tokens") or 0)
                                / float(item.get("latency") or 1)
                            )
                        ),
                        _fmt(
                            item.get("typing_speed")
                            if item.get("typing_speed") is not None
                            else (
                                int(item.get("output_tokens") or 0)
                                / max(
                                    float(item.get("latency") or 0)
                                    - float(item.get("ttft") or 0),
                                    1e-12,
                                )
                            )
                        ),
                        str(item.get("token_source", "")),
                        str(
                            item.get("source_trace_id")
                            or item.get("traceparent")
                            or ""
                        ),
                        error[:200],
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## 线上流量回放",
            "",
            f"- 修复记录数：{report.get('repaired_records', 0)}",
            f"- 修复操作数：{report.get('repair_operations', 0)}",
            "- JSON 报告保留每个请求的 source_line、source_request_id、source_trace_id、session_id 与 traceparent。",
            "",
            "> ITL 需要服务端提供逐 Token 时间戳；当前回放协议仅有首包和结束时间，因此不伪造 ITL。",
            "",
        ]
    )
    return "\n".join(lines)


class ReplayClient:
    """Asynchronous HTTP runner kept separate for focused unit testing."""

    def __init__(self, settings: ReplaySettings, timestamp_prefix: str):
        settings.validate()
        self.settings = settings
        self.url = normalize_endpoint(settings.url)
        self.timestamp_prefix = timestamp_prefix

    async def send(self, session, request_index: int, record: dict) -> dict:
        request_body = convert_to_teleagi_format(
            record["payload"],
            model=self.settings.model,
            default_temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            ignore_eos=self.settings.ignore_eos,
            timestamp_prefix=self.timestamp_prefix,
            stream=self.settings.stream,
        )
        headers = build_request_headers(
            x_app_id=self.settings.x_app_id,
            session_id=record.get("session_id"),
            send_legacy_app_headers=self.settings.send_legacy_app_headers,
        )
        request_body_size = len(
            json.dumps(request_body, ensure_ascii=False).encode("utf-8")
        )
        started = time.perf_counter()
        first_event_time = None
        status_code = 0
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        reasoning_tokens = 0
        prompt_cache_hit_tokens = 0
        input_tokens_from_api = False
        output_tokens_from_api = False
        response_fragments: list[str] = []
        event_count = 0
        error_text = ""

        def capture_usage(usage: dict) -> None:
            nonlocal input_tokens
            nonlocal output_tokens
            nonlocal total_tokens
            nonlocal reasoning_tokens
            nonlocal prompt_cache_hit_tokens
            nonlocal input_tokens_from_api
            nonlocal output_tokens_from_api
            if usage.get("prompt_tokens") is not None:
                input_tokens = int(usage["prompt_tokens"] or 0)
                input_tokens_from_api = True
            if usage.get("completion_tokens") is not None:
                output_tokens = int(usage["completion_tokens"] or 0)
                output_tokens_from_api = True
            if usage.get("total_tokens") is not None:
                total_tokens = int(usage["total_tokens"] or 0)
            completion_details = usage.get("completion_tokens_details") or {}
            if completion_details.get("reasoning_tokens") is not None:
                reasoning_tokens = int(
                    completion_details["reasoning_tokens"] or 0
                )
            prompt_details = usage.get("prompt_tokens_details") or {}
            cached_tokens = usage.get("prompt_cache_hit_tokens")
            if cached_tokens is None:
                cached_tokens = prompt_details.get("cached_tokens")
            if cached_tokens is not None:
                prompt_cache_hit_tokens = int(cached_tokens or 0)

        try:
            async with session.post(
                self.url,
                json=request_body,
                headers=headers,
            ) as response:
                status_code = response.status
                if response.status != 200:
                    error_text = (await response.text())[:4000]
                elif self.settings.stream:
                    async for event_text in iter_sse_data(response.content):
                        if event_text == "[DONE]":
                            break
                        try:
                            event = json.loads(event_text)
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                f"invalid SSE JSON event: {event_text[:500]}"
                            ) from error
                        event_count += 1
                        if first_event_time is None:
                            first_event_time = time.perf_counter()
                        response_fragments.append(_extract_response_text(event))
                        usage = event.get("usage") or {}
                        capture_usage(usage)
                else:
                    event = json.loads(await response.text())
                    event_count = 1
                    first_event_time = time.perf_counter()
                    response_fragments.append(_extract_response_text(event))
                    usage = event.get("usage") or {}
                    capture_usage(usage)
        except (asyncio.TimeoutError, aiohttp.ClientError, ValueError) as error:
            error_text = f"{type(error).__name__}: {error}"
        except Exception as error:  # keep every request represented in results
            error_text = f"{type(error).__name__}: {error}"

        finished = time.perf_counter()
        response_text = "".join(response_fragments)
        success = status_code == 200 and not error_text and event_count > 0
        if status_code == 200 and not error_text and event_count == 0:
            error_text = "empty response stream"
        if success and not output_tokens_from_api and response_text:
            output_tokens = max(1, len(response_text) // 2)
        if success and not input_tokens_from_api:
            input_tokens = max(1, request_body_size // 2)
        if total_tokens <= 0:
            total_tokens = input_tokens + output_tokens
        if success:
            if input_tokens_from_api and output_tokens_from_api:
                token_source = "api"
            elif input_tokens_from_api:
                token_source = "estimated_output"
            elif output_tokens_from_api:
                token_source = "estimated_prompt"
            else:
                token_source = "estimated_all"
        else:
            token_source = "none"
        latency = finished - started
        ttft = first_event_time - started if first_event_time else None
        generation_time = max(latency - (ttft or 0.0), 0.0)

        return {
            "request_index": request_index,
            "source_line": record.get("source_line"),
            "source_request_id": record.get("request_id"),
            "source_trace_id": record.get("trace_id"),
            "session_id": record.get("session_id"),
            "repair_count": record.get("repair_count", 0),
            "success": success,
            "status_code": status_code,
            "error": error_text,
            "latency": latency,
            "ttft": ttft,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
            "reasoning_tokens": int(reasoning_tokens),
            "prompt_cache_hit_tokens": int(prompt_cache_hit_tokens),
            "cache_hit": prompt_cache_hit_tokens > 0,
            "token_source": token_source,
            "tokens_per_second": output_tokens / latency if latency > 0 else 0.0,
            "typing_speed": (
                output_tokens / generation_time if generation_time > 0 else 0.0
            ),
            "prefill_token_throughput": (
                input_tokens / ttft if ttft and ttft > 0 else None
            ),
            "tpot": (
                generation_time / (output_tokens - 1)
                if generation_time > 0 and output_tokens > 1
                else None
            ),
            "content_length": len(response_text),
            "request_body_size": request_body_size,
            "traceparent": headers["traceparent"],
        }


@TASKS.register_module()
class LLMIOReplayTask(BaseTask):
    """Replay each source record as one complete chat-completions POST."""

    name_prefix = "LLMIOReplay"
    log_subdir = "logs/infer"
    output_subdir = "predictions"

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        self.num_gpus = 0
        self.task_state_manager = None

    def get_command(self, cfg_path, template):
        sys.path.insert(0, os.getcwd())
        command = f'"{sys.executable}" "{__file__}" "{cfg_path}"'
        return template.format(task_cmd=command)

    def _settings(self, dataset_cfg: ConfigDict) -> ReplaySettings:
        data_args = dataset_cfg.get("args", {})
        max_tokens = _config_or_env(
            self.model_cfg,
            "max_tokens",
            "AISBENCH_REPLAY_MAX_TOKENS",
            None,
        )
        return ReplaySettings(
            url=_config_or_env(
                self.model_cfg, "url", "AISBENCH_REPLAY_URL", ""
            ),
            model=_config_or_env(
                self.model_cfg, "model", "AISBENCH_REPLAY_MODEL", ""
            ),
            concurrency=int(
                _config_or_env(
                    self.model_cfg,
                    "concurrent",
                    "AISBENCH_REPLAY_CONCURRENCY",
                    100,
                )
            ),
            requests=int(
                _config_or_env(
                    data_args,
                    "requests",
                    "AISBENCH_REPLAY_REQUESTS",
                    0,
                )
            ),
            timeout=float(
                _config_or_env(
                    self.model_cfg,
                    "timeout",
                    "AISBENCH_REPLAY_TIMEOUT",
                    900,
                )
            ),
            temperature=float(
                _config_or_env(
                    self.model_cfg,
                    "temperature",
                    "AISBENCH_REPLAY_TEMPERATURE",
                    0.7,
                )
            ),
            max_tokens=(
                int(max_tokens) if max_tokens not in (None, "") else None
            ),
            ignore_eos=_config_optional_bool_or_env(
                self.model_cfg,
                "ignore_eos",
                "AISBENCH_REPLAY_IGNORE_EOS",
            ),
            stream=(
                _config_or_env(
                    self.model_cfg,
                    "mode",
                    "AISBENCH_REPLAY_MODE",
                    "stream",
                )
                == "stream"
            ),
            x_app_id=_config_or_env(
                self.model_cfg, "x_app_id", "AISBENCH_REPLAY_X_APP_ID", ""
            ),
            x_app_key=_config_or_env(
                self.model_cfg, "x_app_key", "AISBENCH_REPLAY_X_APP_KEY", ""
            ),
            send_legacy_app_headers=_config_bool_or_env(
                self.model_cfg,
                "send_legacy_app_headers",
                "AISBENCH_REPLAY_SEND_LEGACY_APP_HEADERS",
                True,
            ),
            add_timestamp_prefix=_config_bool_or_env(
                self.model_cfg,
                "add_timestamp_prefix",
                "AISBENCH_REPLAY_ADD_TIMESTAMP_PREFIX",
                True,
            ),
        )

    async def _run_requests(
        self,
        records: list[dict],
        settings: ReplaySettings,
    ) -> tuple[list[dict], float]:
        request_count = settings.requests or len(records)
        if self.task_state_manager is not None:
            self.task_state_manager.update_task_state(
                {
                    "status": "running",
                    "total_count": request_count,
                    "finish_count": 0,
                    "progress_description": "",
                }
            )
        timestamp_prefix = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            if settings.add_timestamp_prefix
            else ""
        )
        client = ReplayClient(settings, timestamp_prefix)
        semaphore = asyncio.Semaphore(settings.concurrency)
        connector = aiohttp.TCPConnector(limit=settings.concurrency, ssl=False)
        timeout = aiohttp.ClientTimeout(total=settings.timeout)

        async def bounded_send(session, request_index, record):
            async with semaphore:
                return await client.send(session, request_index, record)

        started = time.perf_counter()
        results: list[dict] = []
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        ) as session:
            pending = [
                asyncio.create_task(
                    bounded_send(session, index + 1, records[index % len(records)])
                )
                for index in range(request_count)
            ]
            for completed, future in enumerate(asyncio.as_completed(pending), 1):
                results.append(await future)
                if self.task_state_manager is not None:
                    self.task_state_manager.update_task_state(
                        {"finish_count": completed}
                    )
                if completed % 100 == 0 or completed == request_count:
                    self.logger.info(
                        "LLMIO replay progress: %d/%d", completed, request_count
                    )
        duration = time.perf_counter() - started
        results.sort(key=lambda item: item["request_index"])
        return results, duration

    def _write_report(
        self,
        dataset_cfg: ConfigDict,
        settings: ReplaySettings,
        source_path: str,
        records: list[dict],
        details: list[dict],
        duration: float,
    ) -> str:
        output_path = get_infer_output_path(
            self.model_cfg,
            dataset_cfg,
            osp.join(self.work_dir, self.output_subdir),
        )
        mkdir_or_exist(osp.dirname(output_path))
        summary = summarize_results(
            details,
            duration,
            max_concurrency=settings.concurrency,
        )
        report = {
            "task": "llm_io_replay",
            "generated_at": datetime.now().astimezone().isoformat(),
            "dataset": dataset_abbr_from_cfg(dataset_cfg),
            "input_log_file": source_path,
            "loaded_records": len(records),
            "repaired_records": sum(item.get("repair_count", 0) > 0 for item in records),
            "repair_operations": sum(item.get("repair_count", 0) for item in records),
            "settings": {
                **asdict(settings),
                "x_app_key": "***" if settings.x_app_key else "",
            },
            "summary": summary,
            "details": details,
        }
        mmengine.dump(report, output_path, ensure_ascii=False, indent=2)

        markdown_path = osp.splitext(output_path)[0] + ".md"
        with open(markdown_path, "w", encoding="utf-8") as stream:
            stream.write(build_markdown_report(report))
        return output_path

    def display_results(self) -> None:
        """Print persisted metrics in the parent AISBench CLI process."""

        for dataset_cfg in self.dataset_cfgs:
            output_path = get_infer_output_path(
                self.model_cfg,
                dataset_cfg,
                osp.join(self.work_dir, self.output_subdir),
            )
            if not osp.isfile(output_path):
                self.logger.warning(
                    "LLMIO replay report not found for terminal display: %s",
                    output_path,
                )
                continue
            report = mmengine.load(output_path)
            print(format_terminal_report(report), flush=True)
            self.logger.info("Detailed replay report: %s", output_path)

    def run(self, task_state_manager=None):
        self.task_state_manager = task_state_manager
        for dataset_cfg in self.dataset_cfgs:
            data_args = dataset_cfg.get("args", {})
            configured_source = _config_or_env(
                data_args,
                "input_log_file",
                "AISBENCH_REPLAY_LOG_FILE",
                "",
            )
            if not configured_source:
                raise ValueError("input_log_file must not be empty")
            source_path = osp.abspath(osp.expanduser(configured_source))
            settings = self._settings(dataset_cfg)
            settings.validate()
            self.logger.info("Loading llm_io replay log: %s", source_path)
            max_records = int(
                _config_or_env(
                    data_args,
                    "max_records",
                    "AISBENCH_REPLAY_MAX_RECORDS",
                    0,
                )
            )
            records = list(
                iter_llm_io_records(
                    source_path,
                    repair_redacted=_config_bool_or_env(
                        data_args,
                        "repair_redacted",
                        "AISBENCH_REPLAY_REPAIR_REDACTED",
                        False,
                    ),
                    on_error=_config_or_env(
                        data_args,
                        "on_error",
                        "AISBENCH_REPLAY_ON_ERROR",
                        "raise",
                    ),
                    max_records=max_records if max_records > 0 else None,
                )
            )
            if not records:
                raise ValueError(f"No replayable requests found in {source_path}")
            self.logger.info("Loaded %d complete replay records", len(records))
            details, duration = asyncio.run(self._run_requests(records, settings))
            output_path = self._write_report(
                dataset_cfg,
                settings,
                source_path,
                records,
                details,
                duration,
            )
            self.logger.info("LLMIO replay report saved to %s", output_path)


class LLMIOReplayPerfSummarizer:
    """Render replay reports during AISBench's standard ``perf`` workflow."""

    def __init__(self, config: ConfigDict, calculator=None) -> None:
        self.cfg = config
        self.logger = AISLogger()

    def summarize(self) -> None:
        found_report = False
        for model_cfg in self.cfg["models"]:
            model_abbr = model_abbr_from_cfg(model_cfg)
            performance_dir = osp.join(
                self.cfg["work_dir"], "performances", model_abbr
            )
            for dataset_cfg in self.cfg["datasets"]:
                report_path = get_infer_output_path(
                    model_cfg,
                    dataset_cfg,
                    osp.join(self.cfg["work_dir"], "predictions"),
                )
                if not osp.isfile(report_path):
                    self.logger.warning(
                        "LLMIO replay report not found for performance summary: %s",
                        report_path,
                    )
                    continue

                found_report = True
                report = mmengine.load(report_path)
                print(format_terminal_report(report), flush=True)

                mkdir_or_exist(performance_dir)
                dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
                performance_path = osp.join(
                    performance_dir, f"{dataset_abbr}.json"
                )
                details_path = osp.join(
                    performance_dir, f"{dataset_abbr}_details.json"
                )
                markdown_path = osp.join(
                    performance_dir, f"{dataset_abbr}.md"
                )
                mmengine.dump(
                    report,
                    performance_path,
                    ensure_ascii=False,
                    indent=2,
                )
                mmengine.dump(
                    report.get("details", []),
                    details_path,
                    ensure_ascii=False,
                    indent=2,
                )
                with open(markdown_path, "w", encoding="utf-8") as stream:
                    stream.write(build_markdown_report(report))
                self.logger.info(
                    "Performance Result files located in %s.", performance_dir
                )

        if not found_report:
            self.logger.warning("No LLMIO replay reports were summarized.")


def parse_args():
    parser = argparse.ArgumentParser(description="AISBench llm_io replay task")
    parser.add_argument("config", help="Task config file")
    return parser.parse_args()


if __name__ == "__main__":
    logger = AISLogger()
    args = parse_args()
    cfg = Config.fromfile(args.config)
    status_dir = os.path.join(cfg["work_dir"], "status_tmp")
    mkdir_or_exist(status_dir)
    task_state_manager = TaskStateManager(
        tmp_path=status_dir,
        task_name=task_abbr_from_cfg(cfg),
        is_debug=cfg.get("cli_args", {}).get("debug", False),
    )
    manager_thread = threading.Thread(target=task_state_manager.launch)
    manager_thread.start()
    task_state_manager.update_task_state(
        {
            "status": "start",
            "task_log_path": os.path.join(
                "logs/infer", f"{task_abbr_from_cfg(cfg)}.out"
            ),
        }
    )
    started = time.perf_counter()
    try:
        task = LLMIOReplayTask(cfg)
        task.run(task_state_manager)
    except Exception:
        task_state_manager.update_task_state({"status": "error"})
        raise
    finally:
        if task_state_manager.task_state.get("status") != "error":
            task_state_manager.update_task_state({"status": "finish"})
        manager_thread.join()
    logger.info(
        "LLMIO replay task elapsed: %.2fs", time.perf_counter() - started
    )
