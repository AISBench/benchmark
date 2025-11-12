import os
from typing import Dict, List, Optional, Union, Tuple
from transformers import AutoTokenizer

from openai import OpenAI

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList

from ais_bench.benchmark.models import BaseAPIModel, APITemplateParser
from ais_bench.benchmark.models.output import RequestOutput

PromptType = Union[PromptList, str]


@MODELS.register_module()
class VLLMCustomAPIChat(BaseAPIModel):
    """Model wrapper around OpenAI's models. vllm 0.6 +

    Args:
        path (str, optional): Model path or identifier for the specific API model. Defaults to empty string.
        model (str, optional): Name of the model to use for inference. If not provided, will be auto-detected from service. Defaults to empty string.
        stream (bool, optional): Whether to enable streaming output. Defaults to False.
        max_out_len (int, optional): Maximum output length, controlling the maximum number of tokens for generated text. Defaults to 4096.
        retry (int, optional): Number of retry attempts when request fails. Defaults to 2.
        headers (Dict, optional): Headers for the API request. Defaults to {"Content-Type": "application/json"}.
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
        headers: Dict = {"Content-Type": "application/json"},
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
            headers=headers,
            host_ip=host_ip,
            host_port=host_port,
            url=url,
            generation_kwargs=generation_kwargs,
            meta_template=meta_template,
            enable_ssl=enable_ssl,
            verbose=verbose,
        )
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

    def _get_url(self) -> str:
        endpoint = "v1/chat/completions"
        url = f"{self.base_url}{endpoint}"
        self.logger.debug(f"Request url: {url}")
        return url

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
                if item["role"] == "HUMAN":
                    msg["role"] = "user"
                elif item["role"] == "BOT":
                    msg["role"] = "assistant"
                elif item["role"] == "SYSTEM":
                    msg["role"] = "system"
                messages.append(msg)
        output.input = messages
        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update({"max_tokens": max_out_len})
        generation_kwargs.update({"model": self.model})

        request_body = dict(
            stream=self.stream,
            messages=messages,
        )
        if self.stream:
            request_body["stream_options"] = {"include_usage": True}
        request_body = request_body | generation_kwargs
        return request_body

    async def parse_stream_response(self, json_content, output):
        for item in json_content.get("choices", []):
            if item["delta"].get("content"):
                output.content += item["delta"]["content"]
            if item["delta"].get("reasoning_content"):
                output.reasoning_content += item["delta"]["reasoning_content"]
        if json_content.get("usage"):
            output.output_tokens = json_content["usage"]["completion_tokens"]

    async def parse_text_response(self, json_content, output):
        for item in json_content.get("choices", []):
            if content:=item["message"].get("content"):
                output.content += content
            if reasoning_content:=item["message"].get("reasoning_content"):
                output.reasoning_content += reasoning_content
        if json_content.get("usage"):
            output.output_tokens = json_content["usage"]["completion_tokens"]
        self.logger.debug(f"Output content: {output.content}")
        self.logger.debug(f"Output reasoning content: {output.reasoning_content}")