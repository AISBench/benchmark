"""Test cases for BFCL V3 Inferencer."""

import unittest
from unittest import mock
import asyncio
import json
from aiohttp import ClientSession

from ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer import (
    BaseBFCLV3Inferencer,
    BFCLV3FunctionInferencer,
    BFCLV3PromptInferencer,
    BFCLV3FunctionCallInferencer,
    MESSAGE_ROLE_MAP,
)
from ais_bench.benchmark.models.output import FunctionCallOutput
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.utils.logging.exceptions import AISBenchNotImplementedError
from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever


class DummyModel:
    """Dummy model for testing."""
    def __init__(self):
        self.max_out_len = 16
        self.is_api = True
        self.stream = False

    async def generate(self, prompt_list, max_out_len, output, session=None, tools=None):
        output.success = True
        output.content = "test response"
        output.reasoning_content = None


class DummyRetriever:
    """Dummy retriever for testing."""
    def __init__(self):
        self.dataset = mock.MagicMock()
        self.dataset.abbr = "test_dataset"
        self.dataset_reader = mock.MagicMock()
        self.dataset_reader.dataset = {
            "test": [
                {
                    "id": "simple_0",
                    "question": json.dumps([{"role": "user", "content": "test"}]),
                    "function": json.dumps([{"name": "test_func"}]),
                    "initial_config": json.dumps([]),
                    "involved_classes": json.dumps({}),
                },
                {
                    "id": "multi_turn_base_0",
                    "question": json.dumps([
                        {"role": "user", "content": "turn 1"},
                        {"role": "user", "content": "turn 2"}
                    ]),
                    "function": json.dumps([{"name": "test_func"}]),
                    "initial_config": json.dumps([]),
                    "involved_classes": json.dumps({}),
                    "missed_function": json.dumps({"0": [{"name": "holdout_func"}]}),
                }
            ]
        }


class TestBaseBFCLV3Inferencer(unittest.TestCase):
    """Test BaseBFCLV3Inferencer class."""

    def test_pre_query_processing_not_implemented(self):
        """Test that pre_query_processing raises NotImplementedError."""
        inferencer = BaseBFCLV3Inferencer()
        with self.assertRaises(AISBenchNotImplementedError):
            inferencer.pre_query_processing({})

    def test_add_holdout_function_not_implemented(self):
        """Test that add_holdout_function raises NotImplementedError."""
        inferencer = BaseBFCLV3Inferencer()
        with self.assertRaises(AISBenchNotImplementedError):
            inferencer.add_holdout_function({}, {}, [])

    def test_extra_multi_turn_response_not_implemented(self):
        """Test that extra_multi_turn_response raises NotImplementedError."""
        inferencer = BaseBFCLV3Inferencer()
        output = FunctionCallOutput()
        with self.assertRaises(AISBenchNotImplementedError):
            inferencer.extra_multi_turn_response(output, {}, [])

    def test_extrat_single_turn_response_not_implemented(self):
        """Test that extrat_single_turn_response raises NotImplementedError."""
        inferencer = BaseBFCLV3Inferencer()
        output = FunctionCallOutput()
        with self.assertRaises(AISBenchNotImplementedError):
            inferencer.extrat_single_turn_response(output)

    def test_add_execution_results_not_implemented(self):
        """Test that add_execution_results raises NotImplementedError."""
        inferencer = BaseBFCLV3Inferencer()
        with self.assertRaises(AISBenchNotImplementedError):
            inferencer.add_execution_results({}, [], {})

    def test_add_assistant_message(self):
        """Test _add_assistant_message method."""
        inferencer = BaseBFCLV3Inferencer()
        inference_data = {"message": []}
        message = {"role": "assistant", "content": "test"}
        result = inferencer._add_assistant_message(inference_data, message)
        self.assertEqual(result["message"], [message])
        self.assertEqual(inference_data["message"], [message])

    def test_get_test_category(self):
        """Test _get_test_category method."""
        inferencer = BaseBFCLV3Inferencer()
        self.assertEqual(inferencer._get_test_category("simple_0"), "simple")
        self.assertEqual(inferencer._get_test_category("multi_turn_base_1"), "multi_turn_base")

    def test_load_json_field(self):
        """Test _load_json_field method."""
        inferencer = BaseBFCLV3Inferencer()
        # Test with existing key
        input_data = {"function": json.dumps([{"name": "test"}])}
        result = inferencer._load_json_field(input_data, "function")
        self.assertEqual(result, [{"name": "test"}])

        # Test with missing key and default
        result = inferencer._load_json_field(input_data, "missing", default=[])
        self.assertEqual(result, [])

        # Test with missing key and no default
        result = inferencer._load_json_field(input_data, "missing")
        self.assertEqual(result, [])

    def test_convert_message_to_prompt_list(self):
        """Test convert_message_to_prompt_list method."""
        inferencer = BaseBFCLV3Inferencer()

        # Test normal message
        message = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = inferencer.convert_message_to_prompt_list(message)
        self.assertIsInstance(result, PromptList)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "HUMAN")
        self.assertEqual(result[0]["prompt"], "Hello")
        self.assertEqual(result[1]["role"], "BOT")
        self.assertEqual(result[1]["prompt"], "Hi")

        # Test message without role
        message = [{"content": "test"}]
        with mock.patch.object(inferencer.logger, 'warning'):
            result = inferencer.convert_message_to_prompt_list(message)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["role"], "HUMAN")

        # Test message without role and content
        message = [{}]
        with mock.patch.object(inferencer.logger, 'warning'):
            result = inferencer.convert_message_to_prompt_list(message)
            self.assertEqual(len(result), 0)

        # Test unknown role
        message = [{"role": "unknown", "content": "test"}]
        with mock.patch.object(inferencer.logger, 'warning'):
            result = inferencer.convert_message_to_prompt_list(message)
            self.assertEqual(len(result), 0)

        # Test message with extra fields
        message = [{"role": "user", "content": "test", "extra": "field"}]
        result = inferencer.convert_message_to_prompt_list(message)
        self.assertEqual(result[0]["extra"], "field")


class TestBFCLV3FunctionInferencer(unittest.TestCase):
    """Test BFCLV3FunctionInferencer class."""

    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.func_doc_language_specific_pre_processing')
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.convert_to_tool')
    def test_pre_query_processing(self, mock_convert, mock_preprocess):
        """Test pre_query_processing method."""
        inferencer = BFCLV3FunctionInferencer()
        mock_preprocess.return_value = [{"name": "processed_func"}]
        mock_convert.return_value = [{"type": "function", "function": {"name": "test_func"}}]

        input_data = {
            "function": json.dumps([{"name": "test_func"}]),
            "data_name": "simple_0",
            "prompt": json.dumps([{"role": "user", "content": "test"}]),
        }
        result = inferencer.pre_query_processing(input_data)

        self.assertIn("message", result)
        self.assertIn("tools", result)
        self.assertEqual(input_data["prompt"], [{"role": "user", "content": "test"}])

    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.func_doc_language_specific_pre_processing')
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.convert_to_tool')
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC')
    def test_add_holdout_function(self, mock_prompt, mock_convert, mock_preprocess):
        """Test add_holdout_function method."""
        inferencer = BFCLV3FunctionInferencer()
        mock_preprocess.return_value = [{"name": "processed_func"}]
        mock_convert.return_value = [{"type": "function"}]
        mock_prompt = "Add function: {functions}"

        input_data = {
            "function": json.dumps([{"name": "test_func"}]),
            "data_name": "simple_0",
        }
        inference_data = {"message": []}
        holdout_function = [{"name": "holdout_func"}]

        result_inference_data, current_turn_message = inferencer.add_holdout_function(
            input_data, inference_data, holdout_function
        )

        self.assertIn("tools", result_inference_data)
        self.assertEqual(len(current_turn_message), 1)
        self.assertEqual(current_turn_message[0]["role"], "user")

    def test_extra_multi_turn_response_with_tool_calls(self):
        """Test extra_multi_turn_response with tool_calls."""
        inferencer = BFCLV3FunctionInferencer()
        output = FunctionCallOutput()
        output.extra_details_data = {
            "message": {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "func1", "arguments": '{"arg": "value"}'}
                    },
                    {
                        "id": "call_2",
                        "function": {"name": "func2", "arguments": '{"arg2": "value2"}'}
                    }
                ]
            }
        }
        inference_data = {"message": []}
        current_turn_response = []

        with mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.convert_to_function_call') as mock_convert:
            mock_convert.return_value = [{"func1": {"arg": "value"}}, {"func2": {"arg2": "value2"}}]
            result = inferencer.extra_multi_turn_response(output, inference_data, current_turn_response)

            self.assertEqual(len(result), 2)
            self.assertIn("tool_call_ids", inference_data)
            self.assertEqual(len(inference_data["message"]), 1)

    def test_extra_multi_turn_response_without_tool_calls(self):
        """Test extra_multi_turn_response without tool_calls."""
        inferencer = BFCLV3FunctionInferencer()
        output = FunctionCallOutput()
        output.extra_details_data = {
            "message": {
                "content": json.dumps([{"func": "test"}]),
                "tool_calls": None
            }
        }
        inference_data = {"message": []}
        current_turn_response = []

        with mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.convert_to_function_call') as mock_convert:
            mock_convert.return_value = [{"func": "test"}]
            result = inferencer.extra_multi_turn_response(output, inference_data, current_turn_response)

            self.assertEqual(len(result), 1)

    def test_extra_multi_turn_response_with_invalid_json(self):
        """Test extra_multi_turn_response with invalid JSON content."""
        inferencer = BFCLV3FunctionInferencer()
        output = FunctionCallOutput()
        output.extra_details_data = {
            "message": {
                "content": "invalid json",
                "tool_calls": None
            }
        }
        inference_data = {"message": []}
        current_turn_response = []

        result = inferencer.extra_multi_turn_response(output, inference_data, current_turn_response)
        self.assertEqual(result, [])

    def test_extra_multi_turn_response_convert_exception(self):
        """Test extra_multi_turn_response when convert_to_function_call raises exception."""
        inferencer = BFCLV3FunctionInferencer()
        output = FunctionCallOutput()
        output.extra_details_data = {
            "message": {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "func1", "arguments": "{}"}}
                ]
            }
        }
        inference_data = {"message": []}
        current_turn_response = []

        with mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.convert_to_function_call', side_effect=Exception("Error")):
            result = inferencer.extra_multi_turn_response(output, inference_data, current_turn_response)
            self.assertEqual(result, [])

    def test_extrat_single_turn_response_with_tool_calls(self):
        """Test extrat_single_turn_response with tool_calls."""
        inferencer = BFCLV3FunctionInferencer()
        output = FunctionCallOutput()
        output.extra_details_data = {
            "message": {
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "func1", "arguments": "{}"}}
                ]
            }
        }

        inferencer.extrat_single_turn_response(output)
        self.assertEqual(len(output.tool_calls), 1)

    def test_extrat_single_turn_response_without_tool_calls(self):
        """Test extrat_single_turn_response without tool_calls."""
        inferencer = BFCLV3FunctionInferencer()
        output = FunctionCallOutput()
        output.extra_details_data = {
            "message": {
                "content": "test content"
            }
        }

        inferencer.extrat_single_turn_response(output)
        self.assertEqual(output.tool_calls, "test content")

    def test_extrat_single_turn_response_empty(self):
        """Test extrat_single_turn_response with empty response."""
        inferencer = BFCLV3FunctionInferencer()
        output = FunctionCallOutput()
        output.extra_details_data = {
            "message": {
                "content": ""
            }
        }

        inferencer.extrat_single_turn_response(output)
        self.assertEqual(output.tool_calls, [])

    def test_add_execution_results(self):
        """Test add_execution_results method."""
        inferencer = BFCLV3FunctionInferencer()
        inference_data = {
            "message": [],
            "tool_call_ids": ["call_1", "call_2"]
        }
        execution_results = ["result1", "result2"]
        model_response_data = {}

        result = inferencer.add_execution_results(inference_data, execution_results, model_response_data)

        self.assertEqual(len(result["message"]), 2)
        self.assertEqual(result["message"][0]["role"], "tool")
        self.assertEqual(result["message"][0]["content"], "result1")
        self.assertEqual(result["message"][0]["tool_call_id"], "call_1")


class TestBFCLV3PromptInferencer(unittest.TestCase):
    """Test BFCLV3PromptInferencer class."""

    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.func_doc_language_specific_pre_processing')
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.system_prompt_pre_processing_chat_model')
    def test_pre_query_processing(self, mock_system_prompt, mock_preprocess):
        """Test pre_query_processing method."""
        inferencer = BFCLV3PromptInferencer()
        mock_preprocess.return_value = [{"name": "processed_func"}]
        mock_system_prompt.return_value = "processed system prompt"

        input_data = {
            "function": json.dumps([{"name": "test_func"}]),
            "data_name": "simple_0",
            "prompt": json.dumps(["system prompt", "user prompt"]),
        }
        result = inferencer.pre_query_processing(input_data)

        self.assertIn("message", result)
        self.assertEqual(input_data["prompt"][0], "processed system prompt")

    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING')
    def test_add_holdout_function(self, mock_prompt_template):
        """Test add_holdout_function method."""
        inferencer = BFCLV3PromptInferencer()
        mock_prompt_template.format.return_value = "Add function: {functions}"

        input_data = {}
        inference_data = {"message": []}
        holdout_function = [{"name": "holdout_func"}]

        result_inference_data, current_turn_message = inferencer.add_holdout_function(
            input_data, inference_data, holdout_function
        )

        self.assertEqual(result_inference_data, inference_data)
        self.assertEqual(len(current_turn_message), 1)
        self.assertEqual(current_turn_message[0]["role"], "user")

    def test_extra_multi_turn_response(self):
        """Test extra_multi_turn_response method."""
        inferencer = BFCLV3PromptInferencer()
        output = FunctionCallOutput()
        output.content = "test response"
        output.extra_details_data = {
            "message": {"role": "assistant", "content": "test"}
        }
        inference_data = {"message": []}
        current_turn_response = []

        with mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.default_decode_execute_prompting') as mock_decode:
            mock_decode.return_value = [{"func": "test"}]
            result = inferencer.extra_multi_turn_response(output, inference_data, current_turn_response)

            self.assertEqual(len(result), 1)
            self.assertEqual(len(current_turn_response), 1)
            self.assertEqual(len(inference_data["message"]), 1)

    def test_extra_multi_turn_response_decode_exception(self):
        """Test extra_multi_turn_response when decode raises exception."""
        inferencer = BFCLV3PromptInferencer()
        output = FunctionCallOutput()
        output.content = "test response"
        output.extra_details_data = {"message": {}}
        inference_data = {"message": []}
        current_turn_response = []

        with mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.default_decode_execute_prompting', side_effect=Exception("Error")):
            result = inferencer.extra_multi_turn_response(output, inference_data, current_turn_response)
            self.assertEqual(result, [])

    def test_extrat_single_turn_response(self):
        """Test extrat_single_turn_response method."""
        inferencer = BFCLV3PromptInferencer()
        output = FunctionCallOutput()
        output.content = "test content"

        inferencer.extrat_single_turn_response(output)
        self.assertEqual(output.tool_calls, "test content")

    def test_extrat_single_turn_response_empty(self):
        """Test extrat_single_turn_response with empty content."""
        inferencer = BFCLV3PromptInferencer()
        output = FunctionCallOutput()
        output.content = ""

        inferencer.extrat_single_turn_response(output)
        self.assertEqual(output.tool_calls, "")

    def test_add_execution_results(self):
        """Test add_execution_results method."""
        inferencer = BFCLV3PromptInferencer()
        inference_data = {"message": []}
        execution_results = ["result1", "result2"]
        model_response = [{"func": "test"}]

        with mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.format_execution_results_prompting') as mock_format:
            mock_format.return_value = "Formatted results"
            result = inferencer.add_execution_results(inference_data, execution_results, model_response)

            self.assertEqual(len(result["message"]), 1)
            self.assertEqual(result["message"][0]["role"], "user")


class TestBFCLV3FunctionCallInferencer(unittest.TestCase):
    """Test BFCLV3FunctionCallInferencer class."""

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_init_with_returns_tool_calls(self, m_abbr, m_build):
        """Test initialization with returns_tool_calls=True."""
        m_build.return_value = DummyModel()
        model_cfg = {"returns_tool_calls": True}

        inferencer = BFCLV3FunctionCallInferencer(model_cfg=model_cfg)

        self.assertTrue(inferencer.returns_tool_calls)
        self.assertIsInstance(inferencer.impl, BFCLV3FunctionInferencer)
        self.assertIn("function-call-model-", inferencer.model_name)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_init_without_returns_tool_calls(self, m_abbr, m_build):
        """Test initialization with returns_tool_calls=False."""
        m_build.return_value = DummyModel()
        model_cfg = {"returns_tool_calls": False}

        inferencer = BFCLV3FunctionCallInferencer(model_cfg=model_cfg)

        self.assertFalse(inferencer.returns_tool_calls)
        self.assertIsInstance(inferencer.impl, BFCLV3PromptInferencer)
        self.assertIn("prompt-model-", inferencer.model_name)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_init_perf_mode_error(self, m_abbr, m_build):
        """Test initialization with perf_mode raises error."""
        m_build.return_value = DummyModel()
        model_cfg = {}

        with self.assertRaises(AISBenchNotImplementedError):
            BFCLV3FunctionCallInferencer(model_cfg=model_cfg, mode="perf")

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_init_stream_error(self, m_abbr, m_build):
        """Test initialization with stream=True raises error."""
        model = DummyModel()
        model.stream = True
        m_build.return_value = model
        model_cfg = {}

        with self.assertRaises(AISBenchNotImplementedError):
            BFCLV3FunctionCallInferencer(model_cfg=model_cfg)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_data_list(self, m_abbr, m_build):
        """Test get_data_list method."""
        m_build.return_value = DummyModel()
        inferencer = BFCLV3FunctionCallInferencer(model_cfg={})
        retriever = DummyRetriever()

        data_list = inferencer.get_data_list(retriever)

        self.assertEqual(len(data_list), 2)
        self.assertEqual(data_list[0]["data_name"], "simple_0")
        self.assertEqual(data_list[0]["data_abbr"], "test_dataset")
        self.assertEqual(data_list[0]["index"], 0)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.is_empty_execute_response')
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.execute_multi_turn_func_call')
    def test_decode_multi_turn_response_success(self, mock_execute, mock_is_empty, m_abbr, m_build):
        """Test decode_multi_turn_response with successful decoding."""
        m_build.return_value = DummyModel()
        inferencer = BFCLV3FunctionCallInferencer(model_cfg={})

        data = {
            "data_name": "simple_0",
            "initial_config": [],
            "involved_classes": {}
        }
        output = FunctionCallOutput()
        output.content = "test"
        output.reasoning_content = None
        output.extra_details_data = {
            "message": {
                "tool_calls": [{"id": "call_1", "function": {"name": "func1", "arguments": "{}"}}]
            }
        }
        inference_data = {"message": []}
        current_turn_response = []
        current_step_inference_log = []

        mock_is_empty.return_value = False
        mock_execute.return_value = (["result1"], {})

        with mock.patch.object(inferencer.impl, 'extra_multi_turn_response', return_value=[{"func1": {}}]) as mock_extra:
            # Make extra_multi_turn_response append to current_turn_response
            def side_effect(output, inference_data, current_turn_response):
                current_turn_response.append({"func1": {}})
                return [{"func1": {}}]
            mock_extra.side_effect = side_effect

            with mock.patch.object(inferencer.impl, 'add_execution_results', return_value=inference_data):
                result = asyncio.run(asyncio.to_thread(
                    inferencer.decode_multi_turn_response,
                    data, output, inference_data, current_turn_response, current_step_inference_log
                ))

                self.assertTrue(result)
                self.assertEqual(len(current_step_inference_log), 2)
                mock_execute.assert_called_once()

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.is_empty_execute_response')
    def test_decode_multi_turn_response_empty(self, mock_is_empty, m_abbr, m_build):
        """Test decode_multi_turn_response with empty response."""
        m_build.return_value = DummyModel()
        inferencer = BFCLV3FunctionCallInferencer(model_cfg={})

        data = {"data_name": "simple_0"}
        output = FunctionCallOutput()
        output.content = "test"
        inference_data = {}
        current_turn_response = []
        current_step_inference_log = []

        mock_is_empty.return_value = True

        with mock.patch.object(inferencer.impl, 'extra_multi_turn_response', return_value=[]) as mock_extra:
            # Make extra_multi_turn_response append to current_turn_response even when empty
            def side_effect(output, inference_data, current_turn_response):
                current_turn_response.append("empty_response")
                return []
            mock_extra.side_effect = side_effect

            result = asyncio.run(asyncio.to_thread(
                inferencer.decode_multi_turn_response,
                data, output, inference_data, current_turn_response, current_step_inference_log
            ))

            self.assertFalse(result)
            self.assertEqual(len(current_step_inference_log), 1)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_add_next_turn_user_message(self, m_abbr, m_build):
        """Test _add_next_turn_user_message method."""
        m_build.return_value = DummyModel()
        inferencer = BFCLV3FunctionCallInferencer(model_cfg={})

        inference_data = {"message": []}
        user_message = [{"role": "user", "content": "test"}]

        async def run_test():
            return await inferencer._add_next_turn_user_message(inference_data, user_message)

        result = asyncio.run(run_test())

        self.assertEqual(result["message"], user_message)
        self.assertEqual(inference_data["message"], user_message)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.is_empty_execute_response')
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.execute_multi_turn_func_call')
    def test_do_request_multi_turn(self, mock_execute, mock_is_empty, m_abbr, m_build):
        """Test do_request with multi_turn data."""
        m_build.return_value = DummyModel()
        inferencer = BFCLV3FunctionCallInferencer(model_cfg={})

        data = {
            "data_name": "multi_turn_base_0",
            "prompt": [
                [{"role": "user", "content": "turn 1"}],
                [{"role": "user", "content": "turn 2"}]
            ],
            "initial_config": json.dumps([]),
            "involved_classes": json.dumps({}),
            "missed_function": json.dumps({}),
        }

        mock_is_empty.return_value = False
        mock_execute.return_value = (["result"], {})

        async def run_test():
            with mock.patch.object(inferencer, '_inference_multi_turn') as mock_multi:
                mock_multi.return_value = None
                session = mock.MagicMock()
                await inferencer.do_request(data, None, session)
                mock_multi.assert_called_once()

        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_do_request_single_turn(self, m_abbr, m_build):
        """Test do_request with single turn data."""
        m_build.return_value = DummyModel()
        inferencer = BFCLV3FunctionCallInferencer(model_cfg={})

        data = {
            "data_name": "simple_0",
            "prompt": [{"role": "user", "content": "test"}],
        }

        async def run_test():
            with mock.patch.object(inferencer, '_inference_single_turn') as mock_single:
                mock_single.return_value = None
                session = mock.MagicMock()
                await inferencer.do_request(data, None, session)
                mock_single.assert_called_once()

        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.is_empty_execute_response')
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.execute_multi_turn_func_call')
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.MAXIMUM_STEP_LIMIT', 2)
    def test_inference_multi_turn(self, mock_execute, mock_is_empty, m_abbr, m_build):
        """Test _inference_multi_turn method."""
        m_build.return_value = DummyModel()
        inferencer = BFCLV3FunctionCallInferencer(model_cfg={})

        data = {
            "index": 0,
            "data_abbr": "test_dataset",
            "data_name": "multi_turn_base_0",
            "prompt": [
                [{"role": "user", "content": "turn 1"}],
                [{"role": "user", "content": "turn 2"}]
            ],
            "function": json.dumps([{"name": "test_func"}]),
            "initial_config": json.dumps([]),
            "involved_classes": json.dumps({}),
            "missed_function": json.dumps({}),
        }
        final_output = FunctionCallOutput()
        session = mock.MagicMock()

        mock_is_empty.return_value = False
        mock_execute.return_value = (["result"], {})

        with mock.patch.object(inferencer.impl, 'pre_query_processing', return_value={"message": []}), \
             mock.patch.object(inferencer.impl, 'convert_message_to_prompt_list', return_value=PromptList()), \
             mock.patch.object(inferencer.impl, 'add_execution_results', return_value={"message": []}), \
             mock.patch.object(inferencer.model, 'generate') as mock_generate, \
             mock.patch.object(inferencer.impl, 'extra_multi_turn_response') as mock_extra, \
             mock.patch.object(inferencer, 'decode_multi_turn_response', return_value=True) as mock_decode, \
             mock.patch.object(inferencer.status_counter, 'post'), \
             mock.patch.object(inferencer.status_counter, 'rev'), \
             mock.patch.object(inferencer.status_counter, 'finish'), \
             mock.patch.object(inferencer.status_counter, 'case_finish'), \
             mock.patch.object(inferencer.output_handler, 'report_cache_info'):

            # Make extra_multi_turn_response append to current_turn_response
            def extra_side_effect(output, inference_data, current_turn_response):
                current_turn_response.append({"func": {}})
                return [{"func": {}}]
            mock_extra.side_effect = extra_side_effect

            async def mock_gen(*args, **kwargs):
                # output is the 3rd positional argument (after prompt_list and max_out_len)
                if len(args) >= 3:
                    output_obj = args[2]
                else:
                    output_obj = kwargs.get('output')
                if output_obj:
                    output_obj.success = True
                    output_obj.content = "test"
                    output_obj.extra_details_data = {
                        "message": {
                            "tool_calls": [{"id": "call_1", "function": {"name": "func1", "arguments": "{}"}}]
                        }
                    }

            mock_generate.side_effect = mock_gen

            async def run_test():
                await inferencer._inference_multi_turn(data, final_output, session)

            asyncio.run(run_test())

            self.assertTrue(final_output.success)
            mock_generate.assert_called()

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.is_empty_execute_response')
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.execute_multi_turn_func_call')
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.MAXIMUM_STEP_LIMIT', 2)
    def test_inference_multi_turn_with_holdout(self, mock_execute, mock_is_empty, m_abbr, m_build):
        """Test _inference_multi_turn with holdout function."""
        m_build.return_value = DummyModel()
        inferencer = BFCLV3FunctionCallInferencer(model_cfg={})

        data = {
            "index": 0,
            "data_abbr": "test_dataset",
            "data_name": "multi_turn_base_0",
            "prompt": [
                [{"role": "user", "content": "turn 1"}],
            ],
            "function": json.dumps([{"name": "test_func"}]),
            "initial_config": json.dumps([]),
            "involved_classes": json.dumps({}),
            "missed_function": json.dumps({"0": [{"name": "holdout_func"}]}),
        }
        final_output = FunctionCallOutput()
        session = mock.MagicMock()

        mock_is_empty.return_value = False
        mock_execute.return_value = (["result"], {})

        with mock.patch.object(inferencer.impl, 'pre_query_processing', return_value={"message": []}), \
             mock.patch.object(inferencer.impl, 'add_holdout_function', return_value=({"message": []}, [{"role": "user", "content": "holdout"}])), \
             mock.patch.object(inferencer.impl, 'convert_message_to_prompt_list', return_value=PromptList()), \
             mock.patch.object(inferencer.impl, 'add_execution_results', return_value={"message": []}), \
             mock.patch.object(inferencer.model, 'generate') as mock_generate, \
             mock.patch.object(inferencer.impl, 'extra_multi_turn_response') as mock_extra, \
             mock.patch.object(inferencer, 'decode_multi_turn_response', return_value=True) as mock_decode, \
             mock.patch.object(inferencer.status_counter, 'post'), \
             mock.patch.object(inferencer.status_counter, 'rev'), \
             mock.patch.object(inferencer.status_counter, 'finish'), \
             mock.patch.object(inferencer.status_counter, 'case_finish'), \
             mock.patch.object(inferencer.output_handler, 'report_cache_info'):

            # Make extra_multi_turn_response append to current_turn_response
            def extra_side_effect(output, inference_data, current_turn_response):
                current_turn_response.append({"func": {}})
                return [{"func": {}}]
            mock_extra.side_effect = extra_side_effect

            async def mock_gen(*args, **kwargs):
                # output is the 3rd positional argument (after prompt_list and max_out_len)
                if len(args) >= 3:
                    output_obj = args[2]
                else:
                    output_obj = kwargs.get('output')
                if output_obj:
                    output_obj.success = True
                    output_obj.content = "test"
                    output_obj.extra_details_data = {
                        "message": {
                            "tool_calls": [{"id": "call_1", "function": {"name": "func1", "arguments": "{}"}}]
                        }
                    }

            mock_generate.side_effect = mock_gen

            async def run_test():
                await inferencer._inference_multi_turn(data, final_output, session)

            asyncio.run(run_test())

            self.assertTrue(final_output.success)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.MAXIMUM_STEP_LIMIT', 2)
    def test_inference_multi_turn_model_failure(self, m_abbr, m_build):
        """Test _inference_multi_turn when model fails."""
        m_build.return_value = DummyModel()
        inferencer = BFCLV3FunctionCallInferencer(model_cfg={})

        data = {
            "index": 0,
            "data_abbr": "test_dataset",
            "data_name": "multi_turn_base_0",
            "prompt": [
                [{"role": "user", "content": "turn 1"}],
            ],
            "function": json.dumps([{"name": "test_func"}]),
            "initial_config": json.dumps([]),
            "involved_classes": json.dumps({}),
            "missed_function": json.dumps({}),
        }
        final_output = FunctionCallOutput()
        session = mock.MagicMock()

        with mock.patch.object(inferencer.impl, 'pre_query_processing', return_value={"message": []}), \
             mock.patch.object(inferencer.impl, 'convert_message_to_prompt_list', return_value=PromptList()), \
             mock.patch.object(inferencer.model, 'generate') as mock_generate, \
             mock.patch.object(inferencer.status_counter, 'post'), \
             mock.patch.object(inferencer.status_counter, 'failed'), \
             mock.patch.object(inferencer.status_counter, 'case_finish'), \
             mock.patch.object(inferencer.output_handler, 'report_cache_info'), \
             mock.patch.object(inferencer.logger, 'warning'):

            async def mock_gen(*args, **kwargs):
                # output is the 3rd positional argument (after prompt_list and max_out_len)
                if len(args) >= 3:
                    output_obj = args[2]
                else:
                    output_obj = kwargs.get('output')
                if output_obj:
                    output_obj.success = False
                    output_obj.error_info = "Test error"

            mock_generate.side_effect = mock_gen

            async def run_test():
                await inferencer._inference_multi_turn(data, final_output, session)

            asyncio.run(run_test())

            self.assertFalse(final_output.success)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.is_empty_execute_response')
    @mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.MAXIMUM_STEP_LIMIT', 2)
    def test_inference_multi_turn_max_steps(self, mock_is_empty, m_abbr, m_build):
        """Test _inference_multi_turn with maximum steps reached."""
        m_build.return_value = DummyModel()
        inferencer = BFCLV3FunctionCallInferencer(model_cfg={})

        data = {
            "index": 0,
            "data_abbr": "test_dataset",
            "data_name": "multi_turn_base_0",
            "prompt": [
                [{"role": "user", "content": "turn 1"}],
            ],
            "function": json.dumps([{"name": "test_func"}]),
            "initial_config": json.dumps([]),
            "involved_classes": json.dumps({}),
            "missed_function": json.dumps({}),
        }
        final_output = FunctionCallOutput()
        session = mock.MagicMock()

        mock_is_empty.return_value = False

        with mock.patch.object(inferencer.impl, 'pre_query_processing', return_value={"message": []}), \
             mock.patch.object(inferencer.impl, 'convert_message_to_prompt_list', return_value=PromptList()), \
             mock.patch.object(inferencer.impl, 'add_execution_results', return_value={"message": []}), \
             mock.patch.object(inferencer.model, 'generate') as mock_generate, \
             mock.patch.object(inferencer.impl, 'extra_multi_turn_response') as mock_extra, \
             mock.patch.object(inferencer, 'decode_multi_turn_response', return_value=True) as mock_decode, \
             mock.patch.object(inferencer.status_counter, 'post'), \
             mock.patch.object(inferencer.status_counter, 'rev'), \
             mock.patch.object(inferencer.status_counter, 'finish'), \
             mock.patch.object(inferencer.status_counter, 'case_finish'), \
             mock.patch.object(inferencer.output_handler, 'report_cache_info'), \
             mock.patch('ais_bench.benchmark.openicl.icl_inferencer.icl_bfcl_v3_inferencer.execute_multi_turn_func_call', return_value=([], {})):

            # Make extra_multi_turn_response append to current_turn_response
            def extra_side_effect(output, inference_data, current_turn_response):
                current_turn_response.append({"func": {}})
                return [{"func": {}}]
            mock_extra.side_effect = extra_side_effect

            async def mock_gen(*args, **kwargs):
                # output is the 3rd positional argument (after prompt_list and max_out_len)
                if len(args) >= 3:
                    output_obj = args[2]
                else:
                    output_obj = kwargs.get('output')
                if output_obj:
                    output_obj.success = True
                    output_obj.content = "test"
                    output_obj.extra_details_data = {
                        "message": {
                            "tool_calls": [{"id": "call_1", "function": {"name": "func1", "arguments": "{}"}}]
                        }
                    }

            mock_generate.side_effect = mock_gen

            async def run_test():
                await inferencer._inference_multi_turn(data, final_output, session)

            asyncio.run(run_test())

            # Should have reached max steps
            self.assertTrue(final_output.success)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_inference_single_turn_success(self, m_abbr, m_build):
        """Test _inference_single_turn with successful generation."""
        m_build.return_value = DummyModel()
        inferencer = BFCLV3FunctionCallInferencer(model_cfg={})

        data = {
            "index": 0,
            "data_abbr": "test_dataset",
            "prompt": [{"role": "user", "content": "test"}],
            "function": json.dumps([{"name": "test_func"}]),
            "data_name": "simple_0",
        }
        output = FunctionCallOutput()
        session = mock.MagicMock()

        with mock.patch.object(inferencer.impl, 'pre_query_processing', return_value={"message": []}), \
             mock.patch.object(inferencer.impl, 'convert_message_to_prompt_list', return_value=PromptList()), \
             mock.patch.object(inferencer.impl, 'extrat_single_turn_response') as mock_extract, \
             mock.patch.object(inferencer.model, 'generate') as mock_generate, \
             mock.patch.object(inferencer.status_counter, 'post'), \
             mock.patch.object(inferencer.status_counter, 'rev'), \
             mock.patch.object(inferencer.status_counter, 'finish'), \
             mock.patch.object(inferencer.status_counter, 'case_finish'), \
             mock.patch.object(inferencer.output_handler, 'report_cache_info'):

            async def mock_gen(*args, **kwargs):
                # output is the 3rd positional argument (after prompt_list and max_out_len)
                if len(args) >= 3:
                    output_obj = args[2]
                else:
                    output_obj = kwargs.get('output')
                if output_obj:
                    output_obj.success = True
                    output_obj.content = "test response"
                    output_obj.reasoning_content = None

            mock_generate.side_effect = mock_gen

            async def run_test():
                await inferencer._inference_single_turn(data, output, session)

            asyncio.run(run_test())

            self.assertTrue(output.success)
            mock_extract.assert_called_once()

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_inference_single_turn_failure(self, m_abbr, m_build):
        """Test _inference_single_turn when model fails."""
        m_build.return_value = DummyModel()
        inferencer = BFCLV3FunctionCallInferencer(model_cfg={})

        data = {
            "index": 0,
            "data_abbr": "test_dataset",
            "prompt": [{"role": "user", "content": "test"}],
            "function": json.dumps([{"name": "test_func"}]),
            "data_name": "simple_0",
        }
        output = FunctionCallOutput()
        session = mock.MagicMock()

        with mock.patch.object(inferencer.impl, 'pre_query_processing', return_value={"message": []}), \
             mock.patch.object(inferencer.impl, 'convert_message_to_prompt_list', return_value=PromptList()), \
             mock.patch.object(inferencer.model, 'generate') as mock_generate, \
             mock.patch.object(inferencer.status_counter, 'post'), \
             mock.patch.object(inferencer.status_counter, 'failed'), \
             mock.patch.object(inferencer.status_counter, 'finish'), \
             mock.patch.object(inferencer.status_counter, 'case_finish'), \
             mock.patch.object(inferencer.output_handler, 'report_cache_info'):

            async def mock_gen(*args, **kwargs):
                # output is the 3rd positional argument (after prompt_list and max_out_len)
                if len(args) >= 3:
                    output_obj = args[2]
                else:
                    output_obj = kwargs.get('output')
                if output_obj:
                    output_obj.success = False
                    output_obj.error_info = "Test error"

            mock_generate.side_effect = mock_gen

            async def run_test():
                await inferencer._inference_single_turn(data, output, session)

            asyncio.run(run_test())

            self.assertFalse(output.success)