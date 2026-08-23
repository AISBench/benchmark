from __future__ import annotations

import json
from urllib.parse import urlsplit, urlunsplit
from typing import Any

from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.models.api_models.vllm_custom_api import VLLMCustomAPI
from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.logging.error_codes import MODEL_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchValueError


_DP_KEY = "_aisbench_prefix_cache_dp_rank"


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

    async def get_request_body(self, input_data, max_out_len, output, dp_rank=None, **args):
        # 把该请求应路由到的 DP rank 藏进请求体，供后续构造请求头使用。
        body = await super().get_request_body(input_data, max_out_len, output, **args)
        body[_DP_KEY] = dp_rank
        return body

    def _payload_and_headers(self, request_body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        """拆分请求体与请求头：去掉内部 DP 标记，并把 rank 写入自定义请求头。"""
        payload = {key: value for key, value in request_body.items() if key != _DP_KEY}
        headers = dict(self.headers)
        rank = request_body.get(_DP_KEY)
        if rank is not None:
            # vLLM 依据该请求头把请求固定路由到指定 DP 卡，保证同组请求落在同一张卡的缓存上。
            headers["X-data-parallel-rank"] = str(rank)
        return payload, headers

    async def text_infer(self, request_body, output):
        # 非流式推理：发 POST，非 200 记为失败；成功则解析 JSON 并写入 output。
        payload, headers = self._payload_and_headers(request_body)
        await output.record_time_point()
        async with self.session.post(url=self.url, json=payload, headers=headers) as response:
            if response.status != 200:
                output.error_info = response.reason
                output.success = False
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
            output.success = True

    async def stream_infer(self, request_body, output):
        # 流式推理：逐行解析 SSE 数据，忽略注释行与 [DONE]，边收边写 output。
        payload, headers = self._payload_and_headers(request_body)
        await output.record_time_point()
        async with self.session.post(url=self.url, json=payload, headers=headers) as response:
            if response.status != 200:
                output.error_info = response.reason
                output.success = False
                return
            async for raw_chunk in self.iter_lines(response.content):
                chunk = raw_chunk.strip().decode("utf-8")
                if not chunk or chunk.startswith(":"):
                    continue
                chunk = chunk.removeprefix("data:").strip()
                if chunk == "[DONE]":
                    break
                await output.record_time_point()
                await self.parse_stream_response(json.loads(chunk), output)
            output.success = True
