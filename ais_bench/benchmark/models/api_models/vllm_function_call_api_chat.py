import os
import uuid
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from mmengine.config import ConfigDict

from openai import OpenAI

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList

from ais_bench.benchmark.models import BaseAPIModel
from ais_bench.benchmark.datasets.bfcl.bfcl_dependency import *

PromptType = Union[PromptList, str, dict]


class VLLMFunctionBaseAPIChat(BaseAPIModel):
    """
    Base class for VLLM function-calling chat APIs, providing standard interfaces for
    pre-processing queries, injecting holdout functions, handling multi-turn inference,
    and recording execution results.
    """

    def __init__(self, client):
        """
        Initialize the API wrapper.

        Args:
            client: An instance of the VLLM client or wrapper used to perform requests.
        """
        self.client = client

    def pre_query_processing(self, input: dict) -> dict:
        """
        Prepare and sanitize the user input before sending to the model.

        Args:
            input (dict): Raw input payload from the caller.

        Returns:
            dict: Processed payload ready for inference.

        Raises:
            NotImplementedError: Must be overridden in subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement pre_query_processing method."
        )

    def add_holdout_function(
        self, input: dict, inference_data: dict, holdout_function: list[dict]
    ):
        """
        Inject or configure additional functions that the model should consider but not execute
        immediately (holdout functions).

        Args:
            input (dict): Original or pre-processed input payload.
            inference_data (dict): Context or metadata collected during inference.
            holdout_function (list[dict]): List of function definitions to hold out.

        Returns:
            None: Modifies inference_data or input in-place to include holdout definitions.

        Raises:
            NotImplementedError: Must be overridden in subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement add_holdout_function method."
        )

    def inference_multi_turn(
        self, cache_data, generation_kwargs, inference_data, current_turn_response
    ):
        """
        Perform or accumulate results across multiple chat turns.

        Args:
            cache_data: Persistent state between turns (e.g., message log).
            generation_kwargs (dict): Parameters for the model call (e.g. temperature).
            inference_data (dict): Current turn context and metadata.
            current_turn_response: Response object from the last API call.

        Returns:
            Updated cache_data or aggregated output.

        Raises:
            NotImplementedError: Must be overridden in subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement inference_multi_turn method."
        )

    def add_execution_results(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        """
        Record the actual execution results after function calling.

        Args:
            inference_data (dict): Context from the inference stage.
            execution_results (list[str]): Outputs returned by executing functions.
            model_response_data (dict): Raw model API response for reference.

        Returns:
            dict: Enriched inference_data containing execution outputs.

        Raises:
            NotImplementedError: Must be overridden in subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement add_execution_results method."
        )

    def _add_assistant_message(
        self, inference_data: dict, model_responses_message_for_chat_history
    ) -> dict:
        inference_data["message"].append(model_responses_message_for_chat_history)
        return inference_data

    def _get_test_category(self, data_name: str) -> str:
        """Extract test category from data_name."""
        return data_name.rsplit("_", 1)[0]

    def _load_json_field(self, input: dict, key: str):
        """Safely load a JSON field from input dict."""
        return json.loads(input[key]) if key in input else []


class VLLMFunctionAPIChat(VLLMFunctionBaseAPIChat):
    def __init__(self, client):
        super().__init__(client)

    def pre_query_processing(self, input: dict) -> dict:
        """Preprocess inputs and compile tool information."""
        inference_data = {"message": []}
        functions = self._load_json_field(input, "function")
        test_category = self._get_test_category(input.get("data_name", ""))
        inference_data = self._compile_tools(inference_data, functions, test_category)
        input["prompt"] = self._load_json_field(input, "prompt")
        return inference_data

    def add_holdout_function(
        self, input: dict, inference_data: dict, holdout_function: list[dict]
    ):
        functions = self._load_json_field(input, "function")
        test_category = self._get_test_category(input.get("data_name", ""))
        functions.extend(holdout_function)
        inference_data = self._compile_tools(inference_data, functions, test_category)
        current_turn_message = [
            {
                "role": "user",
                "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
            }
        ]
        return inference_data, current_turn_message

    def inference_multi_turn(
        self, cache_data, generation_kwargs, inference_data, current_turn_response
    ):
        generation_kwargs.update({"tools": inference_data.get("tools")})
        response = self.client.request(cache_data, generation_kwargs)
        inference_data["tool_call_ids"] = self.client.tool_call_ids
        current_turn_response.append(response)
        inference_data = self._add_assistant_message(
            inference_data, self.client.model_responses_message_for_chat_history
        )
        try:
            result = json.loads(response)
            result = convert_to_function_call(result)
        except Exception as e:
            return []
        return result

    def _compile_tools(self, inference_data: dict, functions: dict, test_category: str) -> dict:
        """编译函数为工具格式。"""
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OpenAI)
        inference_data["tools"] = tools
        return inference_data

    def add_execution_results(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        """将执行结果添加为 tool 消消息。"""
        for execution_result, tool_call_id in zip(
            execution_results, self.client.tool_call_ids
        ):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)
        return inference_data


class VLLMPromptAPIChat(VLLMFunctionBaseAPIChat):
    def __init__(self, client):
        super().__init__(client)

    def pre_query_processing(self, input: dict) -> dict:
        """预处理输入，处理系统提示。"""
        functions = self._load_json_field(input, "function")
        test_category = self._get_test_category(input["data_name"])
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        prompts = self._load_json_field(input, "prompt")
        prompts[0] = system_prompt_pre_processing_chat_model(
            prompts[0], functions, test_category
        )
        input["prompt"] = prompts
        return {"message": []}

    def add_holdout_function(
        self, input: dict, inference_data: dict, holdout_function: list[dict]
    ):
        current_turn_message = [
            {
                "role": "user",
                "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                    functions=holdout_function
                ),
            }
        ]
        return inference_data, current_turn_message

    def inference_multi_turn(
        self, cache_data, generation_kwargs, inference_data, current_turn_response
    ):
        response = self.client.request(cache_data, generation_kwargs)
        current_turn_response.append(cache_data.output)
        inference_data = self._add_assistant_message(
            inference_data, self.client.model_responses_message_for_chat_history
        )
        try:
            result = default_decode_execute_prompting(cache_data.output)
        except Exception:
            return []
        return result

    def add_execution_results(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response: list[str],
    ) -> dict:
        model_response_data = {"model_responses_decoded": model_response}
        formatted_results_message = format_execution_results_prompting(
            {}, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )
        return inference_data


@MODELS.register_module()
class VLLMFunctionCallAPIChat(BaseAPIModel):
    pass