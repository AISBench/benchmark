import json
import os
import urllib
from typing import Dict, Optional, Union

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.models import BaseAPIModel, APITemplateParser
from ais_bench.benchmark.models.output import RequestOutput, Output
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.ppl_inferencer_output_handler import PPLRequestOutput

PromptType = Union[PromptList, str]

# Role mapping for converting internal role names to API role names
ROLE_MAP = {
    "HUMAN": "user",
    "BOT": "assistant",
    "SYSTEM": "system",
    "TOOL": "tool",
}


@MODELS.register_module()
class VLLMCustomAPIChat(BaseAPIModel):
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
    is_chat_api: bool = True

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
            self.logger.info("API key is set")
        self.meta_template = (
            dict(
                round=[
                    dict(role="HUMAN", api_role="HUMAN"),
                    dict(role="BOT", api_role="BOT", generate=True),
                ],
                reserved_roles=[dict(role="SYSTEM", api_role="SYSTEM")],
            )
            if not meta_template
            else meta_template
        )
        self.model = model if model else self._get_service_model_path()
        self.url = self._get_url()
        self.template_parser = APITemplateParser(self.meta_template)
        self.session = None
        # Multi-LoRA: load data_id -> lora adapter name map from generation_kwargs (optional).
        self.lora_data_map = self._load_lora_data_map(generation_kwargs)

    def _get_url(self) -> str:
        endpoint = "v1/chat/completions"
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

    def _resolve_lora_model_name(self, output: RequestOutput):
        """Look up LoRA adapter name for the current sample's data_id.

        ``output.data_id`` is set by ``GenInferencer.do_request``; if not
        available or not in the map, returns None and the caller falls back
        to the base model.
        """
        if not self.lora_data_map:
            return None
        data_id = getattr(output, "data_id", None)
        if data_id is None:
            return None
        return self.lora_data_map.get(f"{data_id}")

    async def get_request_body(
        self, input: PromptType, max_out_len: int, output: RequestOutput, **args
    ):
        if max_out_len <= 0:
            return ""
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        else:
            messages = []
            for item in input:
                msg = {"content": item["prompt"]}
                # Use hash table (dict) driven approach for role mapping
                role = item.get("role", "")
                msg["role"] = ROLE_MAP.get(role, role)  # Use original role if not in map
                for key, value in item.items(): # copy all other items to msg
                    if key not in ["role", "prompt"]:
                        msg[key] = value
                messages.append(msg)
        output.input = messages
        generation_kwargs = self.generation_kwargs.copy()
        if self.response_anomaly_enabled:
            # vLLM returns OpenAI-style logprobs with token text by default.
            # Token ids are required to convert them into msProbe input.
            generation_kwargs['return_token_ids'] = True
            generation_kwargs['return_tokens_as_token_ids'] = True
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
        if args.get("tools"):
            generation_kwargs.update({"tools": args["tools"]})

        request_body = dict(
            stream=self.stream,
            messages=messages,
        )
        if self.stream:
            request_body["stream_options"] = {"include_usage": True}
        request_body = request_body | generation_kwargs
        return request_body

    async def _parse_usage(self, json_content: dict, output: Output):
        if json_content.get("usage"):
            output.input_tokens = json_content["usage"].get("prompt_tokens", 0)
            output.output_tokens = json_content["usage"].get("completion_tokens", 0)

    async def parse_stream_response(self, json_content, output):
        for item in json_content.get("choices", []):
            if item["delta"].get("content"):
                output.content += item["delta"]["content"]
            if reasoning := item["delta"].get("reasoning_content") or item["delta"].get("reasoning"):
                output.reasoning_content += reasoning
        await self._parse_usage(json_content, output)

    async def parse_text_response(self, json_content, output):
        for item in json_content.get("choices", []):
            if content:=item["message"].get("content"):
                output.content += content
            if reasoning_content:=item["message"].get("reasoning_content") or item["message"].get("reasoning"):
                output.reasoning_content += reasoning_content
        await self._parse_usage(json_content, output)
        output.update_extra_details_data_from_text_response(json_content)
        self.logger.debug(f"Output content: {output.content}")
        self.logger.debug(f"Output reasoning content: {output.reasoning_content}")

    async def get_ppl_request_body(self, input_data:PromptType, max_out_len: int, output: PPLRequestOutput, **args):
        request_body = await self.get_request_body(input_data, max_out_len, output, **args)
        request_body.update({"prompt_logprobs": 0})
        return request_body

    def get_prompt_logprobs(self, data: dict):
        return data.get("prompt_logprobs", [])

@MODELS.register_module()
class VLLMFunctionCallAPIChat(VLLMCustomAPIChat):

    def __init__(self, *args, **kwargs):
        kwargs['stream'] = False
        super().__init__(*args, **kwargs)
        self.logger.warning("VLLMFunctionCallAPIChat is deprecated, please use VLLMCustomAPIChat instead.")

@MODELS.register_module()
class VLLMCustomAPIChatStream(VLLMCustomAPIChat):

    def __init__(self, *args, **kwargs):
        kwargs['stream'] = True
        super().__init__(*args, **kwargs)
        self.logger.warning("VLLMCustomAPIChatStream is deprecated, please use VLLMCustomAPIChat with stream=True instead.")

@MODELS.register_module()
class VllmMultiturnAPIChatStream(VLLMCustomAPIChat):

    def __init__(self, *args, **kwargs):
        kwargs.pop("custom_client", None)
        kwargs['stream'] = True
        super().__init__(*args, **kwargs)
        self.logger.warning("VllmMultiturnAPIChatStream is deprecated, please use VLLMCustomAPIChat with stream=True instead.")
