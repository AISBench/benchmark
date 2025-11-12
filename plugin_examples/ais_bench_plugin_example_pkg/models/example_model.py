import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union, Tuple

from tqdm import tqdm

from openai import OpenAI

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList, is_mm_prompt

from ais_bench.benchmark.models.base_api import BaseAPIModel, handle_synthetic_input
from ais_bench.benchmark.models.performance_api import PerformanceAPIModel
from ais_bench.benchmark.clients import OpenAIChatStreamClient, OpenAIChatTextClient
from ais_bench.benchmark.utils.results import MiddleData
from ais_bench.benchmark.utils.build import build_client_from_cfg

PromptType = Union[PromptList, str, dict]


@MODELS.register_module()
class ExampleModel(PerformanceAPIModel):
    """Model wrapper around OpenAI's models. vllm 0.6 +

    Args:
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        request_rate (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        host_ip (str): The  host ip of custom service, default "localhost".
        host_port (int): The host port of custom service, default "8080".
        enable_ssl (bool, optional): .
    """

    is_api: bool = True
    is_chat_api: bool = True

    def __init__(self,
                 path: str = "",
                 model: str = "",
                 max_seq_len: int = 4096,
                 request_rate: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 verbose: bool = False,
                 host_ip: str = "localhost",
                 host_port: int = 8080,
                 enable_ssl: bool = False,
                 custom_client = dict(type=OpenAIChatTextClient),
                 generation_kwargs: Optional[Dict] = None,
                 trust_remote_code: bool = False):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         request_rate=request_rate,
                         rpm_verbose=rpm_verbose,
                         retry=retry,
                         verbose=verbose,
                         generation_kwargs=generation_kwargs,
                         trust_remote_code=trust_remote_code)
        self.host_ip = host_ip
        self.host_port = host_port
        self.enable_ssl = enable_ssl
        self.base_url = self._get_base_url()
        self.endpoint_url = os.path.join(self.base_url, "chat/completions")
        self.model= model if model else self._get_service_model_path()
        self.is_multi_modal = False
        self.init_client(custom_client)

    def init_client(self, custom_client):
        if not isinstance(custom_client, dict):
            self.logger.warning(f"Value of custom_client: {custom_client} is not a dict! Use Default")
            custom_client = dict(type=OpenAIChatTextClient)
        custom_client['url'] = self.endpoint_url
        custom_client['retry'] = self.retry
        self.client = build_client_from_cfg(custom_client)

    def encode_input(self, prompt: list) -> Tuple[float, List[int]]:
        """Encode a string into tokens, measuring processing time."""
        if not self.tokenizer:
            self.logger.error("Tokenizer is not initialized.")
            return 0.0, []

        assert len(prompt)>0 and isinstance(prompt[0], dict)
        if "content" in prompt[0] and isinstance(prompt[0]['content'], list):
            self.logger.warning(f"Input type: expected a string, got list, InputTokens will be 0.")
            return 0.0, []

        messages = self.tokenizer.tokenizer.tokenizer_model.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        time_start = time.perf_counter()
        tokens = self.tokenizer.encode(messages)
        time_cost = (time.perf_counter() - time_start) * 1000  # Convert to milliseconds
        return time_cost, tokens

    def _input_decode(self, tokens: List):
        if not self.tokenizer:
            self.logger.error("Tokenizer is not initialized.")
            return []
        return self.tokenizer.decode(tokens)

    def prepare_input_data(self, inputs: list, data_id: int = -1) -> MiddleData:
        """Prepare input data, tokenize if performance mode is enabled."""
        rrid = uuid.uuid4().hex
        cache_data = self.result_cache[rrid]
        cache_data.data_id = data_id
        cache_data.request_id = rrid
        cache_data.input_data = inputs
        return cache_data

    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int = 512,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in AISBench'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        batch_size = kwargs.get("batch_size", len(inputs))
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(
                tqdm(executor.map(self._generate, inputs,
                                  [max_out_len] * len(inputs)),
                     total=len(inputs),
                     desc='Inferencing'))
        return results

    @handle_synthetic_input
    def _generate(self, input: PromptType, max_out_len: int) -> str:
        """Generate results given a list of inputs.

        Args:
            inputs (PromptType): A string or PromptDict.
                The PromptDict should be organized in AISBench'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        if isinstance(input, dict):
            data_id = input.get('data_id')
            input = input.get('prompt')
        else:
            data_id = -1
        if max_out_len <= 0:
            return ''

        if isinstance(input, str) or self.is_multi_modal:
            messages = [{'role': 'user', 'content': input}]
        elif is_mm_prompt(input):
            self.is_multi_modal = True
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)

        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update({"max_tokens": max_out_len})
        generation_kwargs.update({"model": self.model})
        cache_data = self.prepare_input_data(messages, data_id)

        response = self.client.request(cache_data, generation_kwargs)
        self.set_result(cache_data)

        return ''.join(response)

    def _get_base_url(self):
        if self.enable_ssl:
            return f"https://{self.host_ip}:{self.host_port}/v1"
        return f"http://{self.host_ip}:{self.host_port}/v1"

    def _get_service_model_path(self):
        client = OpenAI(api_key="EMPTY", base_url=self.base_url)
        return client.models.list().data[0].id