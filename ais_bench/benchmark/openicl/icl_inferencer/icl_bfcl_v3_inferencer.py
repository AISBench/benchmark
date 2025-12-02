"""BFCL V3 Inferencer."""

import json
import asyncio
from typing import Any, List, Optional, Union
import uuid
from aiohttp import ClientSession
from multiprocessing import BoundedSemaphore

from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.datasets.bfcl.bfcl_dependency import *
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_api_inferencer import BaseApiInferencer
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.bfcl_v3_output_handler import BFCLV3OutputHandler
from ais_bench.benchmark.models.output import FunctionCallOutput
from ais_bench.benchmark.utils.logging.exceptions import AISBenchNotImplementedError
from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES
from ais_bench.benchmark.utils.logging.logger import AISLogger

PromptType = Union[PromptList, str]


MESSAGE_ROLE_MAP = {
    "user": "HUMAN",
    "assistant": "BOT",
    "tool": "TOOL",
    "system": "SYSTEM",
}


class BaseBFCLV3Inferencer:
    """Base class for BFCL V3 Inferencer."""
    logger = AISLogger()

    def pre_query_processing(self, input: dict) -> dict:
        """
        Prepare and sanitize the user input before sending to the model.

        Args:
            input (dict): Raw input payload from the caller.

        Returns:
            dict: Processed payload ready for inference.

        Raises:
            AISBenchNotImplementedError: Must be overridden in subclass.
        """
        raise AISBenchNotImplementedError(
            ICLI_CODES.UNKNOWN_ERROR,
            f"pre_query_processing method must be overridden in {self.__class__.__name__}."
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
            AISBenchNotImplementedError: Must be overridden in subclass.
        """
        raise AISBenchNotImplementedError(
            ICLI_CODES.UNKNOWN_ERROR,
            f"add_holdout_function method must be overridden in {self.__class__.__name__}."
        )

    def extra_multi_turn_response(
        self,
        output: FunctionCallOutput,
        inference_data: dict,
        current_turn_response: list[str],
    ):
        """
        Perform or accumulate results across multiple chat turns.

        Args:
            output: FunctionCallOutput: The output from the model.
            inference_data: dict: Current turn context and metadata.
            current_turn_response: list[str]: The response from the last API call.

        Returns:
            list[str]: The response from the model.
        """
        raise AISBenchNotImplementedError(
            ICLI_CODES.UNKNOWN_ERROR,
            f"extra_multi_turn_response method must be overridden in {self.__class__.__name__}."
        )

    def extrat_single_turn_response(self, output: FunctionCallOutput):
        """
        Extract the single turn response from the model.

        Args:
            output (FunctionCallOutput): The output from the model.

        Returns:
            None: Modifies output in-place to include tool calls.

        Raises:
            AISBenchNotImplementedError: Must be overridden in subclass.
        """
        raise AISBenchNotImplementedError(
            ICLI_CODES.UNKNOWN_ERROR,
            f"extrat_single_turn_response method must be overridden in {self.__class__.__name__}."
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
            AISBenchNotImplementedError: Must be overridden in subclass.
        """
        raise AISBenchNotImplementedError(
            ICLI_CODES.UNKNOWN_ERROR,
            f"add_execution_results method must be overridden in {self.__class__.__name__}."
        )

    def _add_assistant_message(
        self, inference_data: dict, model_responses_message_for_chat_history
    ) -> dict:
        inference_data["message"].append(model_responses_message_for_chat_history)
        return inference_data

    def _get_test_category(self, data_name: str) -> str:
        """Extract test category from data_name."""
        return data_name.rsplit("_", 1)[0]

    def _load_json_field(self, input: dict, key: str, default: Any = []) -> Any:
        """Safely load a JSON field from input dict."""
        return json.loads(input[key]) if key in input else default

    def convert_message_to_prompt_list(self, message: list[dict]) -> PromptList:
        """Convert message to prompt list.

        Args:
            message: List of message

        Returns:
            PromptList: Prompt list
        """
        prompt_list = PromptList()
        for item in message:
            role = item.get("role", "")
            content = item.get("content", "")
            if not role:
                if not content:
                    self.logger.warning("Message without role and content, skip.")
                    continue
                self.logger.warning(f"Message without role: {item}, use [HUMAN] instead.")
                role = "user"
            elif role not in MESSAGE_ROLE_MAP:
                self.logger.warning(f"Unknown message role: {role}")
                continue
            new_item = {
                "role": MESSAGE_ROLE_MAP[role],
                "prompt": content,
            }
            for key, value in item.items():
                if key not in ["role", "content"]:
                    new_item[key] = value
            prompt_list.append(new_item)
        return prompt_list


class BFCLV3FunctionInferencer(BaseBFCLV3Inferencer):

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

    def extra_multi_turn_response(
        self,
        output: FunctionCallOutput,
        inference_data: dict,
        current_turn_response: list[str],
    ):
        message = output.extra_details_data.get("message", {})
        tool_calls = message.get("tool_calls", [])
        if tool_calls is None:
            tool_calls = []
        model_responses = [
            {func_call["function"]["name"].strip(): func_call["function"]["arguments"]}
            for func_call in tool_calls
        ]
        inference_data["tool_call_ids"] = [func_call["id"] for func_call in tool_calls]

        if not model_responses:
            model_responses = message.get("content", "")
            try:
                model_responses = json.loads(model_responses)
            except Exception as e:
                self.logger.debug(f"Failed to load model responses: {model_responses}. Error: {e}. Use empty list instead.")
                model_responses = []
        current_turn_response.append(model_responses)
        inference_data = self._add_assistant_message(inference_data, message)
        try:
            result = convert_to_function_call(model_responses)
        except Exception as e:
            return []
        return result

    def extrat_single_turn_response(self, output: FunctionCallOutput):
        message = output.extra_details_data.get("message", {})
        tool_calls = message.get("tool_calls", [])
        model_responses = [
            {func_call["function"]["name"].strip(): func_call["function"]["arguments"]}
            for func_call in tool_calls
        ]
        if not model_responses:
            model_responses = message.get("content", "")
        output.tool_calls = model_responses if model_responses else []

    def _compile_tools(
        self, inference_data: dict, functions: dict, test_category: str
    ) -> dict:
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
        for execution_result, tool_call_id in zip(
            execution_results, inference_data['tool_call_ids']
        ):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)
        return inference_data


class BFCLV3PromptInferencer(BaseBFCLV3Inferencer):

    def pre_query_processing(self, input: dict) -> dict:
        """Preprocess inputs and compile tool information."""
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

    def extra_multi_turn_response(
        self,
        output: FunctionCallOutput,
        inference_data: dict,
        current_turn_response: list[str],
    ):
        current_turn_response.append(output.content)
        inference_data = self._add_assistant_message(
            inference_data, output.extra_details_data.get("message", {})
        )
        try:
            result = default_decode_execute_prompting(output.content)
        except Exception:
            return []
        return result

    def extrat_single_turn_response(self, output: FunctionCallOutput):
        output.tool_calls = output.content if output.content else ""

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
class BFCLV3FunctionCallInferencer(BaseApiInferencer):
    """BFCLV3 Function Call Inferencer class to evaluate by function calling.

    Attributes:
        model_cfg
        batch_size (:obj:`int`, optional): Batch size for the
            :obj:`DataLoader`.
        output_json_filepath (:obj:`str`, optional): File path for output
            `JSON` file.
        gen_field_replace_token (:obj:`str`, optional): Used to replace the
            generation field token when generating prompts.
        save_every (:obj:`int`, optional): Save intermediate results every
            `save_every` iters. Defaults to 1.
    """

    def __init__(
        self,
        model_cfg,
        stopping_criteria: List[str] = [],
        batch_size: Optional[int] = 1,
        mode: Optional[str] = "infer",
        gen_field_replace_token: Optional[str] = "",
        output_json_filepath: Optional[str] = "./icl_inference_output",
        save_every: Optional[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            batch_size=batch_size,
            mode=mode,
            output_json_filepath=output_json_filepath,
            save_every=save_every,
            **kwargs,
        )
        if self.perf_mode:
            raise AISBenchNotImplementedError(
                ICLI_CODES.IMPLEMENTATION_ERROR_BFCL_V3_NOT_SUPPORT_PERF_MODE,
                "BFCLV3FunctionCallInferencer does not support perf_mode."
            )
        if hasattr(self.model, "stream") and self.model.stream:
            raise AISBenchNotImplementedError(
                ICLI_CODES.IMPLEMENTATION_ERROR_BFCL_V3_NOT_SUPPORT_STREAM,
                "BFCLV3FunctionCallInferencer does not support stream. Please set [stream] to False in the model config."
            )

        self.returns_tool_calls = model_cfg.get("returns_tool_calls", False)
        if self.returns_tool_calls:
            self.logger.info("returns_tool_calls = True, parse tools from model response tool calls. "
            "Make sure tool calls is supported by the model. Otherwise, the result will be incorrect.")
            self.impl = BFCLV3FunctionInferencer()
            self.model_name = "function-call-model-" + str(uuid.uuid4()).split("-")[-1]
        else:
            self.logger.info("returns_tool_calls = False, parse tools from model response content.")
            self.impl = BFCLV3PromptInferencer()
            self.model_name = "prompt-model-" + str(uuid.uuid4()).split("-")[-1]
        self.output_handler = BFCLV3OutputHandler(
            perf_mode=self.perf_mode, save_every=self.save_every
        )

    def get_data_list(
        self,
        retriever: BaseRetriever,
    ) -> List:
        data_abbr = retriever.dataset.abbr
        test_datasets = retriever.dataset_reader.dataset["test"]
        extra_infos = [
            "involved_classes",  # the external interfaces/components/tools that are used.
            "initial_config",  # used to decide the executable operations, parameter values, validation conditions, and subsequent state updates.
            "missed_function",  # the functions/operations/components/tools that are missed.
        ]
        data_list = []
        for index, test_data in enumerate(test_datasets):
            data = {
                "prompt": test_data["question"],  # multi-turn prompt json.dumped string
                "function": test_data["function"],  # function list json.dumped string
                "data_name": test_data[
                    "id"
                ],  # data name "simple_0"  "multi_turn_base_0" etc.
                "data_abbr": data_abbr,
                "index": index,  # data index in the test dataset
            }
            for key in extra_infos:  # extra infos are not required, so we only add them if they exist.
                if key in test_data:
                    data[key] = test_data[key]
                else:
                    continue
            data_list.append(data)
        self.logger.debug(f"Get {len(data_list)} data from {data_abbr}")
        return data_list

    def decode_multi_turn_response(
        self,
        data: dict,
        output: FunctionCallOutput,
        inference_data: dict,
        current_turn_response: list[str],
        current_step_inference_log: list[dict],
    ) -> bool:
        decoded_model_responses = self.impl.extra_multi_turn_response(
            output, inference_data, current_turn_response
        )
        current_step_inference_log.append(
            {
                "model_response_content": output.content,
                "model_response_reasoning_content": output.reasoning_content,
                "model_response_raw_response": current_turn_response[-1],
            }
        )
        if is_empty_execute_response(decoded_model_responses):
            return False

        test_category = data["data_name"].split("_")[0]
        execution_results, involved_instances = execute_multi_turn_func_call(
            decoded_model_responses,
            data.get("initial_config", []),
            data.get("involved_classes", {}),
            self.model_name,
            data["data_name"],
            long_context=(
                "long_context" in test_category or "composite" in test_category
            ),
        )
        current_step_inference_log.append(
            {
                "model_response_decoded": decoded_model_responses,
                "execution_results": execution_results,
                "involved_instances": involved_instances,
            }
        )
        inference_data = self.impl.add_execution_results(
            inference_data, execution_results, decoded_model_responses
        )
        return True

    async def do_request(self, data: dict, token_bucket: BoundedSemaphore, session: ClientSession) -> None:
        finial_output = FunctionCallOutput(self.perf_mode)
        finial_output.uuid = str(uuid.uuid4()).replace("-", "")
        if "multi_turn" in data.get("data_name"):
            await self._inference_multi_turn(data, finial_output, session)
        else:
            await self._inference_single_turn(data, finial_output, session)

    async def _add_next_turn_user_message(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    async def _inference_multi_turn(self, data: dict, finial_output: FunctionCallOutput, session: ClientSession) -> None:
        """Inference data multi turn.
        Args:
            data (dict): Data from the dataset.
            finial_output (FunctionCallOutput): Final output.
            session (ClientSession): Aiohttp session.
        """
        index: str = data.get("index")
        data_abbr: str = data.get("data_abbr")
        data["model_name"] = self.model_name
        data["initial_config"] = json.loads(data.get("initial_config"))
        data["involved_classes"] = json.loads(data.get("involved_classes"))
        holdout_function: dict[int, list] = json.loads(
            data.get("missed_function", "{}")
        )
        force_quit = False
        all_model_response: list[list] = []
        inference_data: dict = await asyncio.to_thread(
            self.impl.pre_query_processing, data
        )
        all_multi_turn_messages = data.get("prompt", [])
        finial_output.success = True
        inference_log = finial_output.inference_log

        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_message: list[dict]
            if str(turn_idx) in holdout_function:
                inference_data, current_turn_message = await asyncio.to_thread(
                    self.impl.add_holdout_function,
                    data,
                    inference_data,
                    holdout_function[str(turn_idx)],
                )
            inference_data = await self._add_next_turn_user_message(inference_data, current_turn_message)

            current_turn_inference_log = {"turn_index": turn_idx, "begin_of_turn_query": current_turn_message}
            inference_log.append(current_turn_inference_log)
            current_turn_response = []
            count = 0
            while True:
                current_step_inference_log: list[dict] = []
                # Add to the current_turn_inference_log at beginning of each step so that we don't need to bother dealing with the break statements
                current_turn_inference_log[f"step_{count}"] = current_step_inference_log

                prompt_list = await asyncio.to_thread(
                    self.impl.convert_message_to_prompt_list, inference_data["message"]
                )
                tools = inference_data.get("tools") if self.returns_tool_calls else None
                output = FunctionCallOutput(self.perf_mode)
                output.uuid = str(uuid.uuid4()).replace("-", "")
                await self.status_counter.post()
                await self.model.generate(
                    prompt_list,
                    self.model.max_out_len,
                    output,
                    session=session,
                    tools=tools,
                )
                if output.success:
                    await self.status_counter.rev()
                else:
                    await self.status_counter.failed()
                    self.logger.warning(f"Model has failed to generate response for turn {turn_idx} step {count}."
                    f" Error: {output.error_info}")
                    finial_output.success = False
                    force_quit = True
                    current_step_inference_log.append(
                        {
                            "multi_turn_finish_reason": "model failed to generate response.",
                            "model_response_error_info": output.error_info,
                            "multi_turn_finish_step": count,
                        }
                    )
                    break
                await self.status_counter.finish()

                ret = await asyncio.to_thread(
                    self.decode_multi_turn_response,
                    data,
                    output,
                    inference_data,
                    current_turn_response,
                    current_step_inference_log,
                )
                if not ret:
                    current_step_inference_log.append(
                        {
                            "multi_turn_finish_reason": "Model has returned empty execution results.",
                            "multi_turn_finish_step": count,
                        }
                    )
                    break
                count += 1
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    current_step_inference_log.append(
                        {
                            "multi_turn_finish_reason": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                            "multi_turn_finish_step": count,
                        }
                    )
                    break
            all_model_response.append(current_turn_response)
            if force_quit:
                break
        await self.status_counter.case_finish()
        if all_model_response:
            finial_output.tool_calls = all_model_response
        await self.output_handler.report_cache_info(
            index, prompt_list, finial_output, data_abbr
        )

    async def _inference_single_turn(self, data: dict, output: FunctionCallOutput, session: ClientSession) -> None:
        """Inference data only single turn.
        Args:
            data (dict): Data from the dataset.
            output (FunctionCallOutput): Final output.
            session (ClientSession): Aiohttp session.
        """
        inference_data = await asyncio.to_thread(self.impl.pre_query_processing, data)
        index = data.get("index")
        prompt = data.get("prompt")
        data_abbr = data.get("data_abbr")
        inference_log = output.inference_log

        inference_data = await self._add_next_turn_user_message(
            inference_data, prompt[0]
        )
        prompt_list = await asyncio.to_thread(
            self.impl.convert_message_to_prompt_list, inference_data["message"]
        )
        tools = inference_data.get("tools") if self.returns_tool_calls else None

        inference_log.append(
            {
                "single_turn_inference_data": inference_data,
            }
        )
        await self.status_counter.post()
        await self.model.generate(
            prompt_list, self.model.max_out_len, output, session=session, tools=tools
        )
        if output.success:
            await self.status_counter.rev()
            inference_log.append(
            {
                "model_response_content": output.content,
                "model_response_reasoning_content": output.reasoning_content,
            }
        )
            await asyncio.to_thread(self.impl.extrat_single_turn_response, output)
            inference_log.append(
                {
                    "model_response_tool_calls": output.tool_calls,
                }
            )
        else:
            await self.status_counter.failed()
        await self.status_counter.finish()
        await self.status_counter.case_finish()

        await self.output_handler.report_cache_info(
            index, prompt_list, output, data_abbr
        )
