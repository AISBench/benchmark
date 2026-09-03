from __future__ import annotations

import json
import time
from urllib.parse import urlsplit, urlunsplit
from typing import Any

from ais_bench.benchmark.models.api_models.vllm_custom_api import VLLMCustomAPI
from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.logging.error_codes import MODEL_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchValueError
from ais_bench.benchmark.utils.logging.logger import AISLogger


_DP_KEY = "_aisbench_prefix_cache_dp_rank"
logger = AISLogger()


@MODELS.register_module()
class VLLMPrefixCacheAPI(VLLMCustomAPI):
    """vLLM completions model with concurrency-safe per-request DP routing."""

    def __init__(self, inference_url: str, *args, **kwargs):
        # 归一化推理地址：若 URL 已带 /v1/completions 后缀则剥掉，基类会自行拼接，
        # 同时保留原始地址供直接 POST 使用。
        parsed = urlsplit(inference_url)
        endpoint_suffix = "/v1/completions"
        if parsed.path.rstrip("/").endswith(endpoint_suffix):
            base_path = parsed.path.rstrip("/")[: -len(endpoint_suffix)] or "/"
            kwargs["url"] = urlunsplit((parsed.scheme, parsed.netloc, base_path, "", ""))
        else:
            kwargs["url"] = inference_url
        super().__init__(*args, **kwargs)
        self.url = inference_url
        logger.debug(
            "[aisbench-model] initialized inference_url=%s model=%s stream=%s max_out_len=%s retry=%s batch_size=%s api_key_configured=%s",
            self._safe_url(inference_url),
            getattr(self, "model", None),
            getattr(self, "stream", None),
            getattr(self, "max_out_len", None),
            getattr(self, "retry", None),
            getattr(self, "batch_size", None),
            bool(kwargs.get("api_key")),
        )

    @staticmethod
    def _safe_url(value: str) -> str:
        try:
            parsed = urlsplit(value)
            hostname = parsed.hostname or ""
            if parsed.port is not None:
                hostname = f"{hostname}:{parsed.port}"
            return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))
        except (TypeError, ValueError):
            return "<invalid-url>"

    async def get_request_body(self, input_data, max_out_len, output, dp_rank=None, **args):
        # 把该请求应路由到的 DP rank 藏进请求体，供后续构造请求头使用。
        body = await super().get_request_body(input_data, max_out_len, output, **args)
        body[_DP_KEY] = dp_rank
        logger.debug(
            "[aisbench-model] request_body built dp_rank=%s input_chars=%d max_out_len=%d stream=%s model=%s generation_keys=%s",
            dp_rank,
            len(input_data) if isinstance(input_data, str) else len(str(input_data)),
            max_out_len,
            body.get("stream"),
            body.get("model"),
            sorted(key for key in body if key not in {"prompt", _DP_KEY}),
        )
        return body

    def _payload_and_headers(self, request_body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        """拆分请求体与请求头：去掉内部 DP 标记，并把 rank 写入自定义请求头。"""
        payload = {key: value for key, value in request_body.items() if key != _DP_KEY}
        headers = dict(self.headers)
        rank = request_body.get(_DP_KEY)
        if rank is not None:
            # vLLM 依据该请求头把请求固定路由到指定 DP 卡，保证同组请求落在同一张卡的缓存上。
            headers["X-data-parallel-rank"] = str(rank)
        logger.debug(
            "[aisbench-model] request route prepared dp_rank=%s routed=%s payload_keys=%s prompt_chars=%d max_tokens=%s stream=%s",
            rank,
            rank is not None,
            sorted(payload),
            len(payload.get("prompt", "")) if isinstance(payload.get("prompt"), str) else len(str(payload.get("prompt", ""))),
            payload.get("max_tokens"),
            payload.get("stream"),
        )
        return payload, headers

    async def text_infer(self, request_body, output):
        # 非流式推理：发 POST，非 200 记为失败；成功则解析 JSON 并写入 output。
        payload, headers = self._payload_and_headers(request_body)
        started = time.perf_counter()
        logger.debug(
            "[aisbench-model] text_infer start url=%s dp_rank=%s max_tokens=%s",
            self._safe_url(self.url),
            request_body.get(_DP_KEY),
            payload.get("max_tokens"),
        )
        await output.record_time_point()
        async with self.session.post(url=self.url, json=payload, headers=headers) as response:
            if response.status != 200:
                output.error_info = response.reason
                output.success = False
                logger.debug(
                    "[aisbench-model] text_infer failed status=%s reason=%s dp_rank=%s elapsed_seconds=%.6f",
                    response.status,
                    response.reason,
                    request_body.get(_DP_KEY),
                    time.perf_counter() - started,
                )
                return
            raw_data = await response.text()
            await output.record_time_point()
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError as exc:
                output.success = False
                output.error_info = f"Unexpected response format: {raw_data}"
                raise AISBenchValueError(MODEL_CODES.PARSE_TEXT_RSP_INVALID_FORMAT, output.error_info) from exc
            await self.parse_text_response(data, output)
            self._record_response_anomaly_payload(data, output)
            output.success = True
            logger.debug(
                "[aisbench-model] text_infer complete status=%s dp_rank=%s input_tokens=%s output_tokens=%s elapsed_seconds=%.6f",
                response.status,
                request_body.get(_DP_KEY),
                getattr(output, "input_tokens", None),
                getattr(output, "output_tokens", None),
                time.perf_counter() - started,
            )

    async def stream_infer(self, request_body, output):
        # 流式推理：逐行解析 SSE 数据，忽略注释行与 [DONE]，边收边写 output。
        payload, headers = self._payload_and_headers(request_body)
        started = time.perf_counter()
        first_chunk_seconds = None
        chunk_count = 0
        logger.debug(
            "[aisbench-model] stream_infer start url=%s dp_rank=%s max_tokens=%s",
            self._safe_url(self.url),
            request_body.get(_DP_KEY),
            payload.get("max_tokens"),
        )
        await output.record_time_point()
        async with self.session.post(url=self.url, json=payload, headers=headers) as response:
            if response.status != 200:
                output.error_info = response.reason
                output.success = False
                logger.debug(
                    "[aisbench-model] stream_infer failed status=%s reason=%s dp_rank=%s elapsed_seconds=%.6f",
                    response.status,
                    response.reason,
                    request_body.get(_DP_KEY),
                    time.perf_counter() - started,
                )
                return
            async for raw_chunk in self.iter_lines(response.content):
                chunk = raw_chunk.strip().decode("utf-8")
                if not chunk or chunk.startswith(":"):
                    continue
                chunk = chunk.removeprefix("data:").strip()
                if chunk == "[DONE]":
                    break
                chunk_count += 1
                if first_chunk_seconds is None:
                    first_chunk_seconds = time.perf_counter() - started
                await output.record_time_point()
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    logger.debug(
                        "[aisbench-model] stream_infer invalid_chunk chunk_index=%d dp_rank=%s chunk_chars=%d",
                        chunk_count,
                        request_body.get(_DP_KEY),
                        len(chunk),
                        exc_info=True,
                    )
                    raise
                await self.parse_stream_response(data, output)
                self._accumulate_response_anomaly_payload(data, output)
            output.success = True
            logger.debug(
                "[aisbench-model] stream_infer complete status=%s dp_rank=%s chunks=%d first_chunk_seconds=%s input_tokens=%s output_tokens=%s elapsed_seconds=%.6f",
                response.status,
                request_body.get(_DP_KEY),
                chunk_count,
                first_chunk_seconds,
                getattr(output, "input_tokens", None),
                getattr(output, "output_tokens", None),
                time.perf_counter() - started,
            )
