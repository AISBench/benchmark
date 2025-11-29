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

    def _get_url(self) -> str:
        endpoint = "v1/completions"
        url = urllib.parse.urljoin(self.base_url, endpoint)
        self.logger.debug(f"Request url: {url}")
        return url

    async def get_request_body(
        self, input_data: PromptType, max_out_len: int, output: Output, **args
    ):
        output.input = input_data
        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update({"max_tokens": max_out_len})
        generation_kwargs.update({"model": self.model})
        request_body = dict(
            prompt=input_data,
            stream=self.stream,
        )
        request_body = request_body | generation_kwargs
        return request_body

    async def parse_text_response(self, api_response: dict, output: Output):
        generated_text = api_response.get("choices", [{}])[0].get("text", "")
        output.content = generated_text
        self.logger.debug(f"Output content: {output.content}")

    async def parse_stream_response(self, api_response: dict, output: Output):
        if len(api_response.get("choices", [])) > 0:
            generated_text = api_response["choices"][0]["text"]
        if generated_text:
            output.content += generated_text

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
