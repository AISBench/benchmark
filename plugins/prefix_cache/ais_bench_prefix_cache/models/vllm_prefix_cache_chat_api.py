from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from ais_bench.benchmark.models.api_models.vllm_custom_api_chat import VLLMCustomAPIChat
from ais_bench.benchmark.registry import MODELS


@MODELS.register_module()
class VLLMPrefixCacheChatAPI(VLLMCustomAPIChat):
    """OpenAI-compatible vLLM Chat API using an exact configured endpoint."""

    def __init__(self, inference_url: str, *args, **kwargs):
        parsed = urlsplit(inference_url)
        endpoint_suffix = "/v1/chat/completions"
        if parsed.path.rstrip("/").endswith(endpoint_suffix):
            base_path = parsed.path.rstrip("/")[: -len(endpoint_suffix)] or "/"
            kwargs["url"] = urlunsplit((parsed.scheme, parsed.netloc, base_path, "", ""))
        else:
            kwargs["url"] = inference_url
        super().__init__(*args, **kwargs)
        self.url = inference_url
