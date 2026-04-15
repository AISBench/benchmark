import urllib
from typing import Dict, Optional, Union

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.models import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class VITAGenerateAPI(BaseAPIModel):
    """Model wrapper for VITA multimodal models served via a custom /generate endpoint.

    Supports text, image, video, and audio inputs in a flat prompt array format.

    Args:
        path (str, optional): Model path or identifier. Defaults to empty string.
        model (str, optional): Name of the model. Defaults to empty string.
        stream (bool, optional): Whether to enable streaming output. Defaults to False.
        max_out_len (int, optional): Maximum output length in tokens. Defaults to 4096.
        retry (int, optional): Number of retry attempts. Defaults to 2.
        api_key (str, optional): API key. Defaults to empty string.
        host_ip (str, optional): Host IP address. Defaults to "localhost".
        host_port (int, optional): Port number. Defaults to 8080.
        url (str, optional): Complete URL. Defaults to empty string.
        trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
        generation_kwargs (Dict, optional): Extra generation parameters. Defaults to None.
        meta_template (Dict, optional): Meta template configuration. Defaults to None.
        enable_ssl (bool, optional): Whether to enable SSL. Defaults to False.
        verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
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
        self.model = model
        self.url = self._get_url()

    def _get_url(self) -> str:
        endpoint = "generate"
        url = urllib.parse.urljoin(self.base_url, endpoint)
        self.logger.debug(f"Request url: {url}")
        return url

    async def get_request_body(
        self, input_data: PromptType, max_out_len: int, output: Output, **args
    ):
        output.input = input_data

        prompt = []
        if isinstance(input_data, str):
            prompt.append({"type": "text", "text": input_data})
        else:
            for item in input_data:
                if item["type"] == "image_url":
                    image_url = item["image_url"]
                    if isinstance(image_url, dict) and "url" in image_url:
                        image_url = image_url["url"]
                    if isinstance(image_url, str) and image_url.startswith("file://"):
                        image_url = image_url[len("file://"):]
                    prompt.append({
                        "type": "image_url",
                        "image_url": image_url
                    })

                elif item["type"] == "video_url":
                    video_url = item["video_url"]
                    if isinstance(video_url, dict) and "url" in video_url:
                        video_url = video_url["url"]
                    if isinstance(video_url, str) and video_url.startswith("file://"):
                        video_url = video_url[len("file://"):]
                    prompt.append({
                        "type": "video_url",
                        "video_url": video_url
                    })

                elif item["type"] == "audio_url":
                    audio_url = item["audio_url"]
                    if isinstance(audio_url, dict) and "url" in audio_url:
                        audio_url = audio_url["url"]
                    if isinstance(audio_url, str) and audio_url.startswith("file://"):
                        audio_url = audio_url[len("file://"):]
                    prompt.append({
                        "type": "audio_url",
                        "audio_url": audio_url
                    })

                elif item["type"] == "text":
                    prompt.append({
                        "type": "text",
                        "text": item["text"]
                    })

        generation_kwargs = self.generation_kwargs.copy() if self.generation_kwargs else {}
        request_body = {
            "prompt": prompt,
            "max_tokens": max_out_len,
            "stream": self.stream,
            "model": self.model,
            **generation_kwargs,
        }
        return request_body

    async def parse_text_response(self, api_response: dict, output: Output):
        texts = api_response.get("text", [])
        output.content = texts[0] if texts else ""

    async def parse_stream_response(self, api_response: dict, output: Output):
        texts = api_response.get("text", [])
        if texts:
            output.content += texts[0]
