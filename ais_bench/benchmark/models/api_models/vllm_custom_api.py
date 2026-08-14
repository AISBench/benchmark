import json
import os
import urllib
from typing import Dict, Optional, Union
from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList

from ais_bench.benchmark.models import BaseAPIModel, LMTemplateParser
from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.ppl_inferencer_output_handler import PPLRequestOutput

PromptType = Union[PromptList, str]

@MODELS.register_module()
class VLLMCustomAPI(BaseAPIModel):
    """Model wrapper around OpenAI's models. vllm 0.6 +

    Args:
        path (str, optional): Model path or identifier for the specific API model. Defaults to empty string.
        model (str, optional): Name of the model to use for inference. If not provided, will be auto-detected from service. Defaults to empty string.
        stream (bool, optional): Whether to enable streaming output. Defaults to False.
        max_out_len (int, optional): Maximum output length, controlling the maximum number of tokens for generated text. Defaults to 4096.
        retry (int, optional): Number of retry attempts when request fails. Defaults to 2.
        api_key (str, optional): API key for the API service. Defaults to empty string.
        host_ip (str, optional): Host IP address of the API service. Defaults to "localhost".
        host_port (int, optional): Port number of the API service. Defaults to 8080.
        url (str, optional): Complete URL address of the API service. Defaults to empty string.
        trust_remote_code (bool, optional): Whether to trust remote code when loading tokenizer. Defaults to False.
        generation_kwargs (Dict, optional): Generation parameters configuration, additional parameters passed to the API service. Defaults to None.
        meta_template (Dict, optional): Meta template configuration for the model, used to define conversation format and roles. Defaults to None.
        enable_ssl (bool, optional): Whether to enable SSL connection. Defaults to False.
        verbose (bool, optional): Whether to enable verbose logging output. Defaults to False.
    """

    is_api: bool = True

    def __init__(
        self,
        path: str = "",
        model: str = "",
        stream: bool = False,
        max_out_len: int = 4096,
        retry: int = 2,
        api_key: str = "",
        host_ip: str = "localhost",
        host_port: int = 8080,
        url: str = "",
        trust_remote_code: bool = False,
        generation_kwargs: Optional[Dict] = None,
        meta_template: Optional[Dict] = None,
        enable_ssl: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            path=path,
            stream=stream,
            max_out_len=max_out_len,
            retry=retry,
            api_key=api_key,
            host_ip=host_ip,
            host_port=host_port,
            url=url,
            generation_kwargs=generation_kwargs,
            meta_template=meta_template,
            enable_ssl=enable_ssl,
            verbose=verbose,
        )
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
            self.logger.info(f"API key is set")
        self.model = model if model else self._get_service_model_path()
        self.url = self._get_url()
        self.template_parser = LMTemplateParser(meta_template)
        # For non-chat APIs, the actual prompt is passed as a plain string (just like with offline models), so LMTemplateParser is used.
        # Multi-LoRA: load data_id -> lora adapter name map from generation_kwargs (optional).
        self.lora_data_map = self._load_lora_data_map(generation_kwargs)

    def _get_url(self) -> str:
        endpoint = "v1/completions"
        url = urllib.parse.urljoin(self.base_url, endpoint)
        self.logger.debug(f"Request url: {url}")
        return url

    @staticmethod
    def _load_lora_data_map(generation_kwargs):
        """Load data_id -> LoRA adapter name mapping JSON file.

        Returns None when ``lora_data_map_file`` is absent/invalid; the caller
        then falls back to the base model without error.
        """
        if not generation_kwargs:
            return None
        lora_data_map_file = generation_kwargs.get("lora_data_map_file")
        if not (isinstance(lora_data_map_file, str) and lora_data_map_file):
            return None
        file_path = os.path.abspath(lora_data_map_file)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _resolve_lora_model_name(self, output: Output):
        """Look up LoRA adapter name for the current sample's data_id."""
        if not self.lora_data_map:
            return None
        data_id = getattr(output, "data_id", None)
        if data_id is None:
            return None
        return self.lora_data_map.get(f"{data_id}")

    async def get_request_body(
        self, input_data: PromptType, max_out_len: int, output: Output, **args
    ):
        output.input = input_data
        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update({"max_tokens": max_out_len})
        # Multi-LoRA: override model field with the resolved LoRA adapter name.
        lora_model_name = self._resolve_lora_model_name(output)
        lora_active = lora_model_name is not None
        actual_model_in_body = lora_model_name if lora_active else self.model
        generation_kwargs.update({"model": actual_model_in_body})
        # Debug: log LoRA routing decision for observability.
        self.logger.debug(
            f"[Multi-LoRA] data_id={getattr(output, 'data_id', None)} "
            f"lora_model_name={lora_model_name} "
            f"model_in_request_body={actual_model_in_body}"
        )
        request_body = dict(
            prompt=input_data,
            stream=self.stream,
        )
        request_body = request_body | generation_kwargs
        if self.stream:
            request_body["stream_options"] = {"include_usage": True}
        return request_body

    async def _parse_usage(self, json_content: dict, output: Output):
        if json_content.get("usage"):
            output.input_tokens = json_content["usage"].get("prompt_tokens", 0)
            output.output_tokens = json_content["usage"].get("completion_tokens", 0)

    async def _parse_logprobs(self, choice: dict, output: Output) -> None:
        # completions API 格式：并行数组 {tokens, token_logprobs, top_logprobs}
        # 转换为与 chat API 一致的嵌套结构：[{token, logprob, top_logprobs}, ...]
        lp = choice.get("logprobs")
        if not lp:
            if self._logprobs_enabled():
                output.extra_details_data["logprobs_warning"] = (
                    "logprobs is enabled in generation_kwargs but missing in response"
                )
            return
        tokens = lp.get("tokens", []) or []
        token_logprobs = lp.get("token_logprobs", []) or []
        top_logprobs = lp.get("top_logprobs", []) or []
        result = []
        for i in range(len(tokens)):
            # token_logprobs 首项可能为 None（predefined token），保留为 None 项以对齐位置
            if token_logprobs[i] is None:
                result.append(None)
                continue
            item = {
                "token": tokens[i],
                "logprob": token_logprobs[i],
                "top_logprobs": top_logprobs[i] if i < len(top_logprobs) else [],
            }
            result.append(item)
        output.origin_logprobs = result

    async def parse_text_response(self, api_response: dict, output: Output):
        generated_text = api_response.get("choices", [{}])[0].get("text", "")
        output.content = generated_text
        await self._parse_usage(api_response, output)
        await self._parse_logprobs(api_response.get("choices", [{}])[0], output)
        self.logger.debug(f"Output content: {output.content}")

    async def parse_stream_response(self, api_response: dict, output: Output):
        generated_text = ""
        if len(api_response.get("choices", [])) > 0:
            generated_text = api_response["choices"][0]["text"]
        if generated_text:
            output.content += generated_text
        await self._parse_usage(api_response, output)

    async def get_ppl_request_body(self, input_data:PromptType, max_out_len: int, output: PPLRequestOutput, **args):
        request_body = await self.get_request_body(input_data, max_out_len, output, **args)
        request_body.update({"prompt_logprobs": 0})
        return request_body

    def get_prompt_logprobs(self, data: dict):
        choices = data.get("choices", [])
        prompt_logprobs = [item.get("prompt_logprobs", {}) for item in choices if item is not None][0]
        return prompt_logprobs

class VLLMCustomAPIStream(VLLMCustomAPI):

    def __init__(self, *args, **kwargs):
        kwargs['stream'] = True
        super().__init__(*args, **kwargs)
        self.logger.warning("VLLMCustomAPIStream is deprecated, please use VLLMCustomAPI with stream=True instead.")
