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
import sys
import threading
import time
import urllib.parse
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime

import aiohttp
import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.tasks.llm_io_replay.parser import iter_llm_io_records
from ais_bench.benchmark.utils.core.abbr import (
    dataset_abbr_from_cfg,
    get_infer_output_path,
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
    for key in ("tools", "tool_choice", "max_tokens"):
        if key in log_payload:
            request_body[key] = copy.deepcopy(log_payload[key])
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


def summarize_results(details: list[dict], duration: float) -> dict:
    successes = [item for item in details if item["success"]]
    latencies = [item["latency"] for item in successes]
    ttfts = [item["ttft"] for item in successes if item["ttft"] is not None]
    total_output_tokens = sum(item["output_tokens"] for item in successes)
    total_input_tokens = sum(item["input_tokens"] for item in successes)

    def stats(values: list[float]) -> dict:
        return {
            "mean": sum(values) / len(values) if values else None,
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values) if values else None,
        }

    return {
        "total_requests": len(details),
        "successful_requests": len(successes),
        "failed_requests": len(details) - len(successes),
        "success_rate": len(successes) / len(details) if details else 0,
        "duration_seconds": duration,
        "request_throughput_rps": len(successes) / duration if duration > 0 else 0,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "output_token_throughput": (
            total_output_tokens / duration if duration > 0 else 0
        ),
        "latency_seconds": stats(latencies),
        "ttft_seconds": stats(ttfts),
    }


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
        response_fragments: list[str] = []
        event_count = 0
        error_text = ""

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
                        input_tokens = usage.get("prompt_tokens", input_tokens) or input_tokens
                        output_tokens = (
                            usage.get("completion_tokens", output_tokens) or output_tokens
                        )
                else:
                    event = json.loads(await response.text())
                    event_count = 1
                    first_event_time = time.perf_counter()
                    response_fragments.append(_extract_response_text(event))
                    usage = event.get("usage") or {}
                    input_tokens = usage.get("prompt_tokens", 0) or 0
                    output_tokens = usage.get("completion_tokens", 0) or 0
        except (asyncio.TimeoutError, aiohttp.ClientError, ValueError) as error:
            error_text = f"{type(error).__name__}: {error}"
        except Exception as error:  # keep every request represented in results
            error_text = f"{type(error).__name__}: {error}"

        finished = time.perf_counter()
        response_text = "".join(response_fragments)
        if output_tokens <= 0 and response_text:
            output_tokens = max(1, len(response_text) // 2)
        success = status_code == 200 and not error_text and event_count > 0
        if status_code == 200 and not error_text and event_count == 0:
            error_text = "empty response stream"

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
            "latency": finished - started,
            "ttft": first_event_time - started if first_event_time else None,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
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
        return ReplaySettings(
            url=os.getenv("AISBENCH_REPLAY_URL", self.model_cfg.get("url", "")),
            model=os.getenv(
                "AISBENCH_REPLAY_MODEL", self.model_cfg.get("model", "")
            ),
            concurrency=int(
                os.getenv(
                    "AISBENCH_REPLAY_CONCURRENCY",
                    str(self.model_cfg.get("concurrent", 100)),
                )
            ),
            requests=int(
                os.getenv(
                    "AISBENCH_REPLAY_REQUESTS",
                    str(data_args.get("requests", 0)),
                )
            ),
            timeout=float(
                os.getenv(
                    "AISBENCH_REPLAY_TIMEOUT",
                    str(self.model_cfg.get("timeout", 900)),
                )
            ),
            temperature=float(
                os.getenv(
                    "AISBENCH_REPLAY_TEMPERATURE",
                    str(self.model_cfg.get("temperature", 0.7)),
                )
            ),
            stream=(
                os.getenv(
                    "AISBENCH_REPLAY_MODE",
                    self.model_cfg.get("mode", "stream"),
                )
                == "stream"
            ),
            x_app_id=os.getenv(
                "AISBENCH_REPLAY_X_APP_ID",
                str(self.model_cfg.get("x_app_id", "")),
            ),
            x_app_key=os.getenv(
                "AISBENCH_REPLAY_X_APP_KEY",
                str(self.model_cfg.get("x_app_key", "")),
            ),
            send_legacy_app_headers=_env_bool(
                "AISBENCH_REPLAY_SEND_LEGACY_APP_HEADERS",
                bool(self.model_cfg.get("send_legacy_app_headers", True)),
            ),
            add_timestamp_prefix=_env_bool(
                "AISBENCH_REPLAY_ADD_TIMESTAMP_PREFIX",
                bool(self.model_cfg.get("add_timestamp_prefix", True)),
            ),
        )

    async def _run_requests(
        self,
        records: list[dict],
        settings: ReplaySettings,
    ) -> tuple[list[dict], float]:
        request_count = settings.requests or len(records)
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
                        {"completed": completed, "total": request_count}
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
        summary = summarize_results(details, duration)
        report = {
            "task": "llm_io_replay",
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
        latency = summary["latency_seconds"]
        ttft = summary["ttft_seconds"]
        with open(markdown_path, "w", encoding="utf-8") as stream:
            stream.write("# AISBench llm_io replay report\n\n")
            stream.write(f"- Input: `{source_path}`\n")
            stream.write(f"- Endpoint: `{settings.url}`\n")
            stream.write(f"- Model: `{settings.model}`\n")
            stream.write(f"- Concurrency: {settings.concurrency}\n")
            stream.write(f"- Requests: {summary['total_requests']}\n")
            stream.write(f"- Success: {summary['successful_requests']}\n")
            stream.write(f"- Failed: {summary['failed_requests']}\n")
            stream.write(
                f"- Throughput: {summary['request_throughput_rps']:.3f} req/s\n"
            )
            stream.write(
                f"- Output throughput: {summary['output_token_throughput']:.3f} token/s\n"
            )
            if latency["p50"] is not None:
                stream.write(
                    "- Latency p50/p95/p99: "
                    f"{latency['p50']:.3f} / {latency['p95']:.3f} / "
                    f"{latency['p99']:.3f} s\n"
                )
            if ttft["p50"] is not None:
                stream.write(
                    "- TTFT p50/p95/p99: "
                    f"{ttft['p50']:.3f} / {ttft['p95']:.3f} / "
                    f"{ttft['p99']:.3f} s\n"
                )
        return output_path

    def run(self, task_state_manager=None):
        self.task_state_manager = task_state_manager
        for dataset_cfg in self.dataset_cfgs:
            data_args = dataset_cfg.get("args", {})
            configured_source = os.getenv(
                "AISBENCH_REPLAY_LOG_FILE", data_args["input_log_file"]
            )
            source_path = osp.abspath(osp.expanduser(configured_source))
            settings = self._settings(dataset_cfg)
            settings.validate()
            self.logger.info("Loading llm_io replay log: %s", source_path)
            max_records = int(
                os.getenv(
                    "AISBENCH_REPLAY_MAX_RECORDS",
                    str(data_args.get("max_records", 0) or 0),
                )
            )
            records = list(
                iter_llm_io_records(
                    source_path,
                    repair_redacted=_env_bool(
                        "AISBENCH_REPLAY_REPAIR_REDACTED",
                        bool(data_args.get("repair_redacted", False)),
                    ),
                    on_error=os.getenv(
                        "AISBENCH_REPLAY_ON_ERROR",
                        data_args.get("on_error", "raise"),
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
