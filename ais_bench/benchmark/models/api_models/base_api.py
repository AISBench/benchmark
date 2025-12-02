import sys
import json
import warnings
import asyncio
import os.path as osp
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import requests
import aiohttp

from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import MODEL_CODES
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchNotImplementedError, AISBenchValueError, AISBenchKeyError,
    AISBenchTypeError, AISBenchRuntimeError, AISBenchImplementationError)
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.models import BaseModel
from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.ppl_inferencer_output_handler import PPLRequestOutput
from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES
from ais_bench.benchmark.global_consts import REQUEST_TIME_OUT


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=REQUEST_TIME_OUT)

PromptType = Union[PromptList, str]

class MockAISLogger(AISLogger):
    """Mock logger for API model. Because model will init in task for warmup and init in infer process,
    so we mock the logger to avoid the print each log in init process twice
    """
    def info(self, msg, *args, **kwargs):
        pass

    def debug(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        pass

class BaseAPIModel(BaseModel):
    """Base class for API model wrapper.

    Args:
        path (str): Model path or identifier for the specific API model.
        stream (bool, optional): Whether to enable streaming output. Defaults to False.
        max_out_len (int, optional): Maximum output length, controlling the maximum number of tokens for generated text. Defaults to 2048.
        retry (int, optional): Number of retry attempts when request fails. Defaults to 2.
        api_key (str, optional): API key for the API service. Defaults to empty string.
        host_ip (str, optional): Host IP address of the API service. Defaults to "localhost".
        host_port (int, optional): Port number of the API service. Defaults to 8080.
        url (str, optional): Complete URL address of the API service. Defaults to empty string.
        meta_template (Dict, optional): Meta template configuration for the model, used to define conversation format and roles. Defaults to None.
        generation_kwargs (Dict, optional): Generation parameters configuration, additional parameters passed to the API service. Defaults to empty dict.
        enable_ssl (bool, optional): Whether to enable SSL connection. Defaults to False.
        verbose (bool, optional): Whether to enable verbose logging output. Defaults to False.
    """

    is_api: bool = True
    init_flag: bool = False

    def __init__(
        self,
        path: str,
        stream: bool = False,
        max_out_len: int = 2048,
        retry: int = 2,
        host_ip: str = "localhost",
        host_port: int = 8080,
        url: str = "",
        meta_template: Optional[Dict] = None,
        generation_kwargs: Dict = dict(),
        enable_ssl: bool = False,
        verbose: bool = False,
        api_key: str = "",
    ):
        if not BaseAPIModel.init_flag:
            self.logger = MockAISLogger()
            BaseAPIModel.init_flag = True
        else:
            self.logger = AISLogger()
        self.path = path
        self.stream = stream
        self.max_out_len = max_out_len
        self.retry = retry
        self.headers = {"Content-Type": "application/json"}
        self.meta_template = meta_template if meta_template else None
        self.host_ip = host_ip
        self.host_port = host_port
        self.url = url
        self.enable_ssl = enable_ssl
        self.template_parser = APITemplateParser(self.meta_template)
        self.generation_kwargs = generation_kwargs
        self.verbose = verbose
        self.session = None
        self.base_url = self._get_base_url()

    @abstractmethod
    def _get_url(self) -> str:
        raise AISBenchNotImplementedError(
            MODEL_CODES.UNKNOWN_ERROR,
            f"{self.__class__.__name__} does not supported"
            " to be called in base classes"
        )

    def _get_base_url(self) -> str:
        protocol = "https" if self.enable_ssl else "http"
        if self.url:
            self.logger.info(f"Using custom URL: [{self.url}], [host_ip: {self.host_ip}] and [host_port: {self.host_port}] will be ignored")
            # Check if URL already contains protocol
            if self.url.startswith("http://") or self.url.startswith("https://"):
                return self.url
            return f"{protocol}://{self.url}"
        base_url = f"{protocol}://{self.host_ip}:{self.host_port}/"
        return base_url

    def _get_service_model_path(self) -> str:
        try:
            url = osp.join(self.base_url, "v1/models")
            headers = self.headers
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                data = response.json()
                model_id = data['data'][0]['id']
                self.logger.debug(f"Service Model ID: {model_id}")
                return model_id
            else:
                response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise AISBenchRuntimeError(
                MODEL_CODES.GET_SERVICE_MODEL_PATH_FAILED,
                f"Failed to get service model path from {self.base_url}. Error: {e}"
            )

    async def iter_lines(self, stream):
        """
        Split the input stream into lines based on multiple delimiters:
        - "\n\n" (LF LF)
        - "\r\n\r\n" (CRLF CRLF)
        - "\r\r" (CR CR)

        If the received packet does not encounter any of these delimiters,
        cache it and concatenate it with the subsequent stream.
        """
        pending = None
        async for chunk in stream:
            if pending is not None:
                chunk = pending + chunk
            lines = [
                d
                for d in chunk.replace(b"\r\n\r\n", b"\n\n")
                .replace(b"\r\r", b"\n\n")
                .split(b"\n\n")
                if d
            ]
            # If there are no lines or the chunk is empty, clear pending
            if not lines or not chunk:
                pending = None
            # If the last line's last byte matches the chunk's last byte,
            # it means the chunk did not end with '\n\n', so the last segment is incomplete
            elif lines[-1][-1] == chunk[-1]:
                pending = lines.pop()
            else:
                pending = None
            for line in lines:
                yield line
        # After the stream ends, yield any remaining incomplete data
        if pending is not None:
            yield pending

    @abstractmethod
    async def get_request_body(
        self, input_data: PromptType, max_out_len: int, output: Output, **args
    ):
        raise AISBenchNotImplementedError(
            MODEL_CODES.UNKNOWN_ERROR,
            f"{self.__class__.__name__} does not supported"
            " to be called in base classes"
        )

    async def parse_text_response(self, data, output):
        raise AISBenchNotImplementedError(
            MODEL_CODES.PARSE_TEXT_RSP_NOT_IMPLEMENTED,
            f"{self.__class__.__name__} should be implemented if stream is False"
        )

    async def parse_stream_response(self, data, output):
        raise AISBenchNotImplementedError(
            MODEL_CODES.PARSE_STREAM_RSP_NOT_IMPLEMENTED,
            f"{self.__class__.__name__} should be implemented if stream is True"
        )

    async def generate(
        self,
        input_data: PromptType,
        max_out_len: int,
        output: Output,
        session: aiohttp.ClientSession = None,
        **args,
    ):
        if not session:
            self.session = aiohttp.ClientSession(
                trust_env=True, timeout=AIOHTTP_TIMEOUT
            )
            close_session = True
        else:
            self.session = session
            close_session = False
        request_body = await self.get_request_body(
            input_data, max_out_len, output, **args
        )
        self.logger.debug(f"Request body: {request_body}")
        retry_count = 0
        for _ in range(self.retry):
            try:
                if self.stream:
                    await self.stream_infer(request_body, output)
                else:
                    await self.text_infer(request_body, output)
                # break retry loop when request is successful
                break
            except asyncio.exceptions.CancelledError as e:
                output.success = False
                output.error_info = "Request cancelled by user"
                break
            except json.JSONDecodeError:
                break
            except Exception as e:
                # increase retry count and set output to failed
                retry_count += 1
                output.success = False
                # Only print the last exception (the actual error), not the full traceback
                exc_type = type(e).__name__
                exc_msg = str(e)
                output.error_info = (
                    f"After {retry_count} retries, request failed with exception: "
                    f"{exc_type}: {exc_msg}"
                )
                await output.clear_time_points()
                continue
        if close_session:
            self.logger.debug(f"Waiting for session close ...")
            await self.session.close()
        return output

    async def stream_infer(self, request_body: dict, output: Output):
        await output.record_time_point()
        async with self.session.post(
            url=self.url, json=request_body, headers=self.headers
        ) as response:
            if response.status == 200:
                async for raw_chunk in self.iter_lines(response.content):
                    chunk = raw_chunk.strip()
                    if not chunk:
                        continue
                    chunk = chunk.decode("utf-8")
                    if chunk.startswith(":"):
                        continue
                    chunk = chunk.removeprefix("data:").strip()
                    if chunk == "[DONE]":
                        break
                    await output.record_time_point()
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError as e:
                        output.success = False
                        output.error_info = f"Unexpected response format: {raw_chunk}. Please check if server is working correctly."
                        raise AISBenchValueError(
                            MODEL_CODES.PARSE_TEXT_RSP_INVALID_FORMAT,
                            f"Unexpected response format. Please check 'error_info' in ***_failed.jsonl for more information."
                        )
                    await self.parse_stream_response(data, output)
                output.success = True
            else:
                output.error_info = response.reason
                output.success = False

    async def text_infer(self, request_body, output: Output):
        await output.record_time_point()
        async with self.session.post(
            url=self.url, json=request_body, headers=self.headers
        ) as response:
            if response.status == 200:
                raw_data = await response.text()
                await output.record_time_point()
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError as e:
                    output.success = False
                    output.error_info = f"Unexpected response format: {raw_data}. Please check if server is working correctly."
                    raise AISBenchValueError(
                        MODEL_CODES.PARSE_TEXT_RSP_INVALID_FORMAT,
                        f"Unexpected response format. Please check ***_details.jsonl for more information."
                    )
                await self.parse_text_response(data, output)
                output.success = True
            else:
                output.error_info = response.reason
                output.success = False

    async def get_ppl(self,
        input_data: PromptType,
        max_out_len: int,
        output: PPLRequestOutput,
        session: aiohttp.ClientSession = None,
        **args
        ):
        """Compute perplexity for a given prompt via the remote API.
        Args:
            input_data: Prompt text or list structure the backend expects.
            max_out_len: Maximum completion tokens; forwarded to the API.
            output: PPLRequestOutput to fill with upstream results (ppl, logprobs, etc.).
            session: Optional aiohttp session to reuse across calls.
            **args: Subclass-specific extra parameters.
        Subclasses must:
            • Compose the request and call the API.
            • Parse returned prompt_logprobs, compute negative average logprob as PPL.
            • Store results in `output` (ppl value, raw logprobs, success flag, error info).
            • Respect session lifecycle: only close if they created it.
        """
        if session is None:
            self.session = aiohttp.ClientSession(trust_env=True, timeout=AIOHTTP_TIMEOUT)
            close_session = True
        else:
            self.session = session
            close_session = False
        request_body = await self.get_ppl_request_body(input_data, max_out_len, output, **args)
        retry_count = 0
        for _ in range(self.retry):
            try:
                async with self.session.post(
                    url=self.url, json=request_body, headers=self.headers
                ) as response:
                    if response.status == 200:
                        raw_data = await response.text()
                        try:
                            data = json.loads(raw_data)
                        except json.JSONDecodeError as e:
                            output.success = False
                            output.error_info = f"Unexpected response format: {raw_data}. Please check if server is working correctly."
                            continue
                        prompt_logprobs = self.get_prompt_logprobs(data)
                        output.origin_prompt_logprobs = prompt_logprobs
                        loss = self._calc_ppl(prompt_logprobs)
                        output.ppl = loss
                        output.success = True
                        break
                    else:
                        output.error_info = response.reason
                        output.success = False
                        continue
            except asyncio.exceptions.CancelledError as e:
                output.success = False
                output.error_info = "Request cancelled by user"
                break
            except Exception as e:
                retry_count += 1
                output.success = False
                # Only print the last exception (the actual error), not the full traceback
                exc_type = type(e).__name__
                exc_msg = str(e)
                output.error_info = (
                    f"After {retry_count} retries, request failed with exception: "
                    f"{exc_type}: {exc_msg}"
                )
                continue
        if close_session:
            await self.session.close()

    async def get_ppl_request_body(self, input_data:PromptType, max_out_len: int, output: PPLRequestOutput, **args):
        raise AISBenchNotImplementedError(ICLI_CODES.IMPLEMENTATION_ERROR_PPL_METHOD_NOT_IMPLEMENTED, f"PPL is not supported for this model.")

    def get_prompt_logprobs(self, data: dict):
        raise AISBenchNotImplementedError(ICLI_CODES.IMPLEMENTATION_ERROR_PPL_METHOD_NOT_IMPLEMENTED, f"PPL is not supported for this model.")

    def _calc_ppl(self, prompt_logprobs: list):
        logprobs = [list(item.values())[0]['logprob'] for item in prompt_logprobs if item is not None]
        tokenids = [list(item.keys())[0] for item in prompt_logprobs if item is not None]
        if len(tokenids) == 0:
            raise AISBenchImplementationError(
                ICLI_CODES.PPL_COMPUTE_ERROR_NO_VALID_TOKENS,
                "No valid tokens with log probabilities found for PPL computation."
            )
        loss = -sum(logprobs) / len(tokenids)
        return loss

class APITemplateParser:
    """Intermidate prompt template parser, specifically for API models.

    Args:
        meta_template (Dict): The meta template for the model.
    """

    def __init__(self, meta_template: Optional[Dict] = None):
        self.logger = AISLogger()
        self.meta_template = meta_template
        # Check meta template
        if meta_template:
            if "round" not in meta_template:
                raise AISBenchTypeError(
                    MODEL_CODES.MISS_REQUIRED_PARAM_IN_META_TEMPLATE,
                    "round is required in meta template"
                )
            if not isinstance(meta_template["round"], list):
                raise AISBenchTypeError(
                    MODEL_CODES.INVALID_TYPE_OF_PARAM_IN_META_TEMPLATE,
                    "round must be a list in meta template"
                )
            keys_to_check = ["round"]

            if "reserved_roles" in meta_template:
                if not isinstance(meta_template["reserved_roles"], list):
                    raise AISBenchTypeError(
                        MODEL_CODES.INVALID_TYPE_OF_PARAM_IN_META_TEMPLATE,
                        "reserved_roles must be a list in meta template"
                    )
                keys_to_check.append("reserved_roles")

            self.roles: Dict[str, dict] = dict()  # maps role name to config
            for meta_key in keys_to_check:
                for item in meta_template[meta_key]:
                    if not isinstance(item, (str, dict)):
                        raise AISBenchTypeError(
                            MODEL_CODES.INVALID_TYPE_OF_PARAM_IN_META_TEMPLATE,
                            f"each item in {meta_key} must be a string or a dict in meta template"
                        )
                    if isinstance(item, dict):
                        if item["role"] in self.roles:
                            raise AISBenchTypeError(
                                MODEL_CODES.ROLE_IN_META_TEMPLATE_IS_NOT_UNIQUE,
                                f"role {item['role']} in meta prompt must be unique!"
                            )
                        self.roles[item["role"]] = item.copy()

    def parse_template(self, prompt_template: PromptType, mode: str) -> PromptType:
        """Parse the intermidate prompt template, and wrap it with meta
        template if applicable. When the meta template is set and the input is
        a PromptList, the return value will be a PromptList containing the full
        conversation history. Each item looks like:

        .. code-block:: python

            {'role': 'user', 'prompt': '...'}).

        Args:
            prompt_template (List[PromptType]): An intermidate prompt
                template (potentially before being wrapped by meta template).
            mode (str): Parsing mode. Choices are 'ppl' and 'gen'.

        Returns:
            List[PromptType]: The finalized prompt or a conversation.
        """
        if not isinstance(prompt_template, (str, list, PromptList, tuple)):
            raise AISBenchTypeError(
                MODEL_CODES.PARSE_TEMPLATE_INVALID_TYPE,
                f"prompt_template must be a string, list of strings, PromptList, or tuple of strings, but got {type(prompt_template)}"
            )

        if not isinstance(prompt_template, (str, PromptList)):
            return [self.parse_template(p, mode=mode) for p in prompt_template]

        if not mode in ["ppl", "gen"]:
            raise AISBenchTypeError(
                MODEL_CODES.PARSE_TEMPLATE_INVALID_MODE,
                f"Parsing mode must be 'ppl' or 'gen', but got {mode}"
            )


        if isinstance(prompt_template, str):
            return prompt_template

        if self.meta_template:

            prompt = PromptList()
            # Whether to keep generating the prompt
            generate = True

            section_stack = []  # stores tuples: (section_name, start_idx)

            for i, item in enumerate(prompt_template):
                if not generate:
                    break
                if isinstance(item, str):
                    if item.strip():
                        # TODO: logger
                        warnings.warn(
                            "Non-empty string in prompt template "
                            "will be ignored in API models."
                        )
                elif isinstance(item, dict) and "section" in item:
                    if item["pos"] == "end":
                        section_name, start_idx = section_stack.pop(-1)
                        if not section_name == item["section"]:
                            raise AISBenchValueError(
                                MODEL_CODES.UNKNOWN_ERROR,
                                f"section {item['section']} in prompt template must match the last section {section_name}"
                            )
                        if section_name in ["round", "ice"]:
                            dialogue = prompt_template[start_idx:i]
                            round_ranges = self._split_rounds(
                                dialogue, self.meta_template["round"]
                            )
                            # Consider inserting multiple round examples into
                            # template
                            for i in range(len(round_ranges) - 1):
                                start = round_ranges[i]
                                end = round_ranges[i + 1]
                                round_template = dialogue[start:end]
                                role_dict = self._update_role_dict(round_template)
                                api_prompts, generate = self._prompt2api(
                                    self.meta_template["round"],
                                    role_dict,
                                    # Start generating only when the mode is in
                                    # generation and the template reaches the
                                    # last round
                                    for_gen=mode == "gen"
                                    and section_name == "round"
                                    and i == len(round_ranges) - 2,
                                )
                                prompt += api_prompts
                    elif item["pos"] == "begin":
                        if not item["section"] in ["begin", "round", "end", "ice"]:
                            raise AISBenchValueError(
                                MODEL_CODES.UNKNOWN_ERROR,
                                f"section {item['section']} in prompt template is not valid, "
                                "it must be 'begin', 'round', 'end', or 'ice'"
                            )
                        section_stack.append((item["section"], i + 1))
                    else:
                        raise AISBenchValueError(
                            MODEL_CODES.UNKNOWN_ERROR,
                            f"Invalid prompt template item pos {item['pos']}, legal item pos are 'begin' or 'end'."
                        )
                elif section_stack[-1][0] in ["begin", "end"]:
                    role_dict = self._update_role_dict(item)
                    api_prompts, generate = self._prompt2api(
                        item, role_dict, for_gen=mode == "gen"
                    )
                    prompt.append(api_prompts)

            # merge the consecutive prompts assigned to the same role
            new_prompt = PromptList([prompt[0]])
            last_role = prompt[0]["role"]
            for item in prompt[1:]:
                if item["role"] == last_role:
                    new_prompt[-1]["prompt"] += "\n" + item["prompt"]
                else:
                    last_role = item["role"]
                    new_prompt.append(item)
            prompt = new_prompt

            if self.meta_template.get("begin", None):
                prompt.insert(0, self.meta_template["begin"])

        else:
            # in case the model does not have any meta template
            prompt = ""
            last_sep = ""
            prompt_mm = []
            for item in prompt_template:
                if isinstance(item, dict) and set(["section", "pos"]) == set(
                    item.keys()
                ):
                    continue
                if isinstance(item, str):
                    if item:
                        prompt += last_sep + item
                elif item.get("prompt", ""):
                    prompt += last_sep + item.get("prompt", "")
                elif item.get("prompt_mm", ""):
                    prompt_mm += item.get("prompt_mm", [])
                last_sep = "\n"
        return prompt if prompt else prompt_mm

    def _update_role_dict(self, prompts: Union[List, str]) -> Dict[str, Dict]:
        """Update the default role dict with the given prompts."""
        role_dict = deepcopy(self.roles)
        if isinstance(prompts, str):
            return role_dict
        elif isinstance(prompts, dict):
            prompts = [prompts]
        for prompt in prompts:
            if isinstance(prompt, dict):
                role = prompt["role"]
                if role not in self.roles:
                    role = prompt.get("fallback_role", None)
                    if not role:
                        self.logger.warning(
                            f"{prompt} neither has an appropriate role nor "
                            "a fallback role."
                        )
                role_dict[role].update(prompt)
        return role_dict

    def _split_rounds(
        self,
        prompt_template: List[Union[str, Dict]],
        single_round_template: List[Union[str, Dict]],
    ) -> List[int]:
        """Split the prompt template into rounds, based on single round
        template.

        Return the index ranges of each round. Specifically,
        prompt_template[res[i]:res[i+1]] represents the i-th round in the
        template.
        """
        role_idxs = {
            role_cfg["role"]: i
            for i, role_cfg in enumerate(single_round_template)
            if not isinstance(role_cfg, str)
        }
        last_role_idx = -1
        cutoff_idxs = [0]
        for idx, template in enumerate(prompt_template):
            if isinstance(template, str):
                continue
            role_idx = role_idxs.get(template["role"], None)
            if role_idx is None:
                try:
                    role_idx = role_idxs[template["fallback_role"]]
                except KeyError:
                    raise AISBenchKeyError(
                        MODEL_CODES.INVALID_ROLE_IN_PROMPT_TEMPLATE,
                        f"prompt template item {template} neither has an appropriate "
                        "role nor a fallback_role."
                    )
            if role_idx <= last_role_idx:
                cutoff_idxs.append(idx)
            last_role_idx = role_idx
        cutoff_idxs.append(len(prompt_template))
        return cutoff_idxs

    def _prompt2api(
        self,
        prompts: Union[List, str],
        role_dict: Dict[str, Dict],
        for_gen: bool = False,
    ) -> Tuple[List, bool]:
        """Convert the prompts to a API-style prompts, given an updated
        role_dict.

        Args:
            prompts (Union[List, str]): The prompts to be converted.
            role_dict (Dict[str, Dict]): The updated role dict.
            for_gen (bool): If True, the prompts will be converted for
                generation tasks. The conversion stops before the first
                role whose "generate" is set to True.

        Returns:
            Tuple[List, bool]: The converted string, and whether the follow-up
            conversion should be proceeded.
        """
        cont = True
        if isinstance(prompts, str):
            return prompts, cont
        elif isinstance(prompts, dict):
            api_role, cont = self._role2api_role(prompts, role_dict, for_gen)
            return api_role, cont

        res = []
        for prompt in prompts:
            if isinstance(prompt, str):
                raise AISBenchTypeError(
                    MODEL_CODES.MIX_STR_WITHOUT_EXPLICIT_ROLE,
                    "Mixing str without explicit role is not allowed in API models!"
                )
            else:
                api_role, cont = self._role2api_role(prompt, role_dict, for_gen)
                if api_role:
                    res.append(api_role)
                if not cont:
                    break
        return res, cont

    def _role2api_role(
        self, role_prompt: Dict, role_dict: Dict[str, Dict], for_gen: bool = False
    ) -> Tuple[Dict, bool]:
        """Convert a role prompt to a string, given an updated role_dict.

        Args:
            role_prompt (Dict): The role prompt to be converted.
            role_dict (Dict[str, Dict]): The updated role dict.
            for_gen (bool): If True, the prompts will be converted for
                generation tasks. The conversion stops before the first
                role whose "generate" is set to True.

        Returns:
            Tuple[Dict, bool]: The converted string, and whether the follow-up
            conversion should be proceeded.
        """
        merged_prompt = role_dict.get(
            role_prompt["role"], role_dict.get(role_prompt.get("fallback_role"))
        )
        # res_api_prompt = dict(type='', )
        if for_gen and merged_prompt.get("generate", False):
            return None, False
        res = {}
        res["role"] = merged_prompt["api_role"]
        if "prompt" in merged_prompt:
            res["prompt"] = merged_prompt.get("begin", "")
            res["prompt"] += merged_prompt.get("prompt", "")
            res["prompt"] += merged_prompt.get("end", "")
        elif "prompt_mm" in merged_prompt:
            res["prompt"] = merged_prompt.get("prompt_mm", [])
        else:
            raise AISBenchValueError(
                MODEL_CODES.INVALID_PROMPT_CONTENT,
                "Invalid prompt content: without 'prompt' or 'prompt_mm' param!"
            )
        return res, True
