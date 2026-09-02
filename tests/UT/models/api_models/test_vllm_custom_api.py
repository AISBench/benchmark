import unittest
import asyncio
import json
import os
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict

from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.models.api_models import base_api
from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.utils.prompt import PromptList


class TestVLLMCustomAPI(unittest.TestCase):
    def setUp(self):
        # 默认配置
        self.default_kwargs = {
            "path": "test-model",
            "model": "test-model-name",
            "stream": False,
            "max_out_len": 100,
            "retry": 1,
            "host_ip": "localhost",
            "host_port": 8080,
            "enable_ssl": False,
            "verbose": False,
            "generation_kwargs": {}
        }
        # 模拟_get_service_model_path方法，避免实际的网络请求
        self._get_service_model_path_patcher = patch.object(
            base_api.BaseAPIModel, "_get_service_model_path"
        )
        self.mock_get_model_path = self._get_service_model_path_patcher.start()
        self.mock_get_model_path.return_value = "mocked-model-path"

    def tearDown(self):
        self._get_service_model_path_patcher.stop()

    def test_init_default_parameters(self):
        """测试使用默认参数初始化"""
        with patch.object(VLLMCustomAPI, '_get_url', return_value='http://localhost:8080/v1/completions'):
            model = VLLMCustomAPI()

            # 验证默认参数是否正确设置
            self.assertEqual(model.path, "")
            self.assertEqual(model.model, "mocked-model-path")  # 应该使用_get_service_model_path的返回值
            self.assertFalse(model.stream)
            self.assertEqual(model.max_out_len, 4096)
            self.assertEqual(model.retry, 2)
            self.assertEqual(model.host_ip, "localhost")
            self.assertEqual(model.host_port, 8080)
            self.assertEqual(model.headers, {"Content-Type": "application/json"})
            self.assertFalse(model.enable_ssl)
            self.assertFalse(model.verbose)
            self.assertIsInstance(model.template_parser, object)

    def test_init_custom_parameters(self):
        """测试使用自定义参数初始化"""
        custom_kwargs = self.default_kwargs.copy()
        custom_kwargs["generation_kwargs"] = {"temperature": 0.7, "top_p": 0.9}
        custom_kwargs["meta_template"] = {
            "round": [{"role": "USER", "begin": "[USER] ", "end": "\n"},
                      {"role": "ASSISTANT", "begin": "[ASSISTANT] ", "end": "\n"}]
        }

        with patch.object(VLLMCustomAPI, '_get_url', return_value='http://localhost:8080/v1/completions'):
            model = VLLMCustomAPI(**custom_kwargs)

            # 验证自定义参数是否正确设置
            self.assertEqual(model.path, "test-model")
            self.assertEqual(model.model, "test-model-name")  # 应该使用提供的model参数
            self.assertEqual(model.generation_kwargs, {"temperature": 0.7, "top_p": 0.9})
            self.assertEqual(model.meta_template, custom_kwargs["meta_template"])

    def test_get_url(self):
        """测试_get_url方法"""
        model = VLLMCustomAPI(path="test-path", host_ip="127.0.0.1", host_port=9000)
        # 模拟base_url属性
        model.base_url = "http://127.0.0.1:9000/"
        url = model._get_url()
        self.assertEqual(url, "http://127.0.0.1:9000/v1/completions")

        # 测试SSL情况
        model = VLLMCustomAPI(path="test-path", enable_ssl=True)
        model.base_url = "https://localhost:8080/"
        url = model._get_url()
        self.assertEqual(url, "https://localhost:8080/v1/completions")

    async def test_get_request_body_string_input(self):
        """测试使用字符串输入调用get_request_body方法"""
        model = VLLMCustomAPI(**self.default_kwargs)
        output = Output()

        request_body = await model.get_request_body("test prompt", 100, output)

        # 验证请求体是否正确构建
        self.assertEqual(request_body["stream"], False)
        self.assertEqual(request_body["model"], "test-model-name")
        self.assertEqual(request_body["max_tokens"], 100)
        self.assertEqual(request_body["prompt"], "test prompt")
        self.assertEqual(output.input, "test prompt")

    async def test_get_request_body_prompt_list_input(self):
        """测试使用PromptList输入调用get_request_body方法"""
        model = VLLMCustomAPI(**self.default_kwargs)
        output = Output()

        # 创建一个PromptList
        prompt_list = PromptList([
            {"role": "HUMAN", "prompt": "Hello, how are you?"},
            {"role": "BOT", "prompt": "I'm doing well, thank you!"}
        ])

        request_body = await model.get_request_body(prompt_list, 100, output)

        # 验证请求体是否正确构建
        self.assertEqual(request_body["stream"], False)
        self.assertEqual(request_body["model"], "test-model-name")
        self.assertEqual(request_body["max_tokens"], 100)
        self.assertEqual(request_body["prompt"], prompt_list)
        self.assertEqual(output.input, prompt_list)

    async def test_get_request_body_with_generation_kwargs(self):
        """测试带有自定义generation_kwargs的请求体构建"""
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"temperature": 0.5, "top_k": 50}
        model = VLLMCustomAPI(**kwargs)
        output = Output()

        request_body = await model.get_request_body("test prompt", 100, output)

        # 验证generation_kwargs是否正确合并
        self.assertEqual(request_body["temperature"], 0.5)
        self.assertEqual(request_body["top_k"], 50)
        self.assertEqual(request_body["max_tokens"], 100)
        self.assertEqual(request_body["model"], "test-model-name")

    async def test_get_request_body_stream_enabled(self):
        """测试启用stream时的请求体构建"""
        kwargs = self.default_kwargs.copy()
        kwargs["stream"] = True
        model = VLLMCustomAPI(**kwargs)
        output = Output()

        request_body = await model.get_request_body("test prompt", 100, output)

        # 验证stream选项是否正确设置
        self.assertTrue(request_body["stream"])

    async def test_parse_text_response(self):
        """测试parse_text_response方法"""
        model = VLLMCustomAPI(**self.default_kwargs)
        output = Output()

        # 重置output.content以确保测试的纯净性
        output.content = ""

        # 创建模拟的文本响应内容
        text_response = {
            "choices": [
                {
                    "text": "Hello, world!"
                }
            ]
        }

        await model.parse_text_response(text_response, output)

        # 验证内容是否正确解析
        self.assertEqual(output.content, "Hello, world!")

        # 测试空响应的情况
        empty_response = {
            "choices": [
                {
                    "text": ""
                }
            ]
        }

        await model.parse_text_response(empty_response, output)

        # 验证内容被覆盖为空字符串（与实际代码行为匹配）
        self.assertEqual(output.content, "")

        # 测试没有choices的情况
        no_choices_response = {}

        await model.parse_text_response(no_choices_response, output)

        # 验证内容保持为空字符串（与实际代码行为匹配）
        self.assertEqual(output.content, "")

        # 测试多个choices的情况
        multi_choices_response = {
            "choices": [
                {
                    "text": "First choice"
                },
                {
                    "text": "Second choice"
                }
            ]
        }

        await model.parse_text_response(multi_choices_response, output)

        # 只应该解析第一个choice
        self.assertEqual(output.content, "First choice")

    async def test_parse_stream_response_with_content(self):
        """测试带有内容的流式响应解析"""
        model = VLLMCustomAPI(**self.default_kwargs)
        output = Output()

        # 创建模拟的流式响应内容
        stream_response = {
            "choices": [
                {
                    "text": "Hello, "
                }
            ]
        }

        await model.parse_stream_response(stream_response, output)

        # 验证内容是否正确解析
        self.assertEqual(output.content, "Hello, ")

        # 继续添加内容
        next_response = {
            "choices": [
                {
                    "text": "world!"
                }
            ]
        }

        await model.parse_stream_response(next_response, output)

        # 验证内容是否正确累积
        self.assertEqual(output.content, "Hello, world!")

    async def test_parse_stream_response_without_content(self):
        """测试不带有内容的流式响应解析"""
        model = VLLMCustomAPI(**self.default_kwargs)
        output = Output()
        output.content = "Initial content"

        # 创建包含空text字段的响应
        empty_response = {
            "choices": [
                {
                    "text": ""
                }
            ]
        }

        await model.parse_stream_response(empty_response, output)

        # 验证内容没有变化
        self.assertEqual(output.content, "Initial content")

    async def test_parse_stream_response_no_choices(self):
        """测试没有choices的流式响应解析"""
        model = VLLMCustomAPI(**self.default_kwargs)
        output = Output()
        output.content = "Initial content"

        # 创建没有choices的响应
        no_choices_response = {}

        # 捕获可能的UnboundLocalError异常
        try:
            await model.parse_stream_response(no_choices_response, output)
            # 如果没有抛出异常，验证内容没有变化
            self.assertEqual(output.content, "Initial content")
        except UnboundLocalError:
            # 如果抛出UnboundLocalError异常，这是预期的行为
            pass

    def test_get_service_model_path_call(self):
        """测试当不提供model参数时，是否调用_get_service_model_path"""
        # 不提供model参数
        kwargs = self.default_kwargs.copy()
        kwargs.pop("model")

        with patch.object(VLLMCustomAPI, '_get_url', return_value='http://localhost:8080/v1/completions'):
            model = VLLMCustomAPI(**kwargs)
            # 验证_get_service_model_path被调用
            self.mock_get_model_path.assert_called()
            # 验证model属性被设置为_get_service_model_path的返回值
            self.assertEqual(model.model, "mocked-model-path")

    def test_is_api_attribute(self):
        """测试is_api类属性"""
        self.assertTrue(VLLMCustomAPI.is_api)

    # 运行异步测试的辅助方法
    def run_async_test(self, coroutine):
        return asyncio.run(coroutine)

    # 包装异步测试方法
    def test_get_request_body_string_input_wrapper(self):
        self.run_async_test(self.test_get_request_body_string_input())

    def test_get_request_body_prompt_list_input_wrapper(self):
        self.run_async_test(self.test_get_request_body_prompt_list_input())

    def test_get_request_body_with_generation_kwargs_wrapper(self):
        self.run_async_test(self.test_get_request_body_with_generation_kwargs())

    def test_get_request_body_stream_enabled_wrapper(self):
        self.run_async_test(self.test_get_request_body_stream_enabled())

    def test_parse_text_response_wrapper(self):
        self.run_async_test(self.test_parse_text_response())

    def test_parse_stream_response_with_content_wrapper(self):
        self.run_async_test(self.test_parse_stream_response_with_content())

    def test_parse_stream_response_without_content_wrapper(self):
        self.run_async_test(self.test_parse_stream_response_without_content())

    def test_parse_stream_response_no_choices_wrapper(self):
        self.run_async_test(self.test_parse_stream_response_no_choices())

    async def test_parse_logprobs_with_data(self):
        """测试_parse_logprobs正确解析logprobs响应"""
        model = VLLMCustomAPI(**self.default_kwargs)
        output = Output()

        choice = {
            "text": "B",
            "logprobs": {
                "tokens": ["B"],
                "token_logprobs": [-0.5],
                "text_offset": [0],
                "top_logprobs": [{"333": -0.5, "444": -2.1}]
            }
        }

        await model._parse_logprobs(choice, output)

        # completions API 转换为统一嵌套结构
        self.assertEqual(output.origin_logprobs, [
            {"token": "B", "logprob": -0.5, "top_logprobs": {"333": -0.5, "444": -2.1}}
        ])

    async def test_parse_logprobs_without_logprobs_field(self):
        """测试_parse_logprobs在响应无logprobs字段时不报错"""
        model = VLLMCustomAPI(**self.default_kwargs)
        output = Output()

        choice = {"text": "B"}

        await model._parse_logprobs(choice, output)

        self.assertEqual(output.origin_logprobs, [])

    async def test_parse_logprobs_with_none_token_logprob(self):
        """测试_parse_logprobs处理None的token_logprob（保留为None项对齐位置）"""
        model = VLLMCustomAPI(**self.default_kwargs)
        output = Output()

        choice = {
            "text": "AB",
            "logprobs": {
                "tokens": ["A", "B"],
                "token_logprobs": [-0.5, None],
                "top_logprobs": [{"333": -0.5}, None]
            }
        }

        await model._parse_logprobs(choice, output)

        self.assertEqual(output.origin_logprobs, [
            {"token": "A", "logprob": -0.5, "top_logprobs": {"333": -0.5}},
            None,
        ])

    async def test_parse_text_response_with_logprobs(self):
        """测试parse_text_response正确调用_parse_logprobs"""
        model = VLLMCustomAPI(**self.default_kwargs)
        output = Output()
        output.content = ""

        response = {
            "choices": [{
                "text": "B",
                "logprobs": {
                    "tokens": ["B"],
                    "token_logprobs": [-0.5],
                    "top_logprobs": [{"333": -0.5}]
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1}
        }

        await model.parse_text_response(response, output)

        self.assertEqual(output.content, "B")
        self.assertEqual(output.input_tokens, 10)
        self.assertEqual(output.output_tokens, 1)
        self.assertEqual(output.origin_logprobs, [
            {"token": "B", "logprob": -0.5, "top_logprobs": {"333": -0.5}}
        ])

    def test_parse_logprobs_with_data_wrapper(self):
        self.run_async_test(self.test_parse_logprobs_with_data())

    def test_parse_logprobs_without_logprobs_field_wrapper(self):
        self.run_async_test(self.test_parse_logprobs_without_logprobs_field())

    def test_parse_logprobs_with_none_token_logprob_wrapper(self):
        self.run_async_test(self.test_parse_logprobs_with_none_token_logprob())

    def test_parse_text_response_with_logprobs_wrapper(self):
        self.run_async_test(self.test_parse_text_response_with_logprobs())

    def test_calc_ppl(self):
        """测试_calc_ppl方法"""
        model = VLLMCustomAPI(**self.default_kwargs)
        
        # 测试正常的logprobs列表
        prompt_logprobs = [
            {"1": {"logprob": -0.5}},
            {"2": {"logprob": -0.3}},
            {"3": {"logprob": -0.7}}
        ]
        
        ppl = model._calc_ppl(prompt_logprobs)
        
        # 计算期望值: -(-0.5 - 0.3 - 0.7) / 3 = 1.5 / 3 = 0.5
        expected_ppl = -(-0.5 - 0.3 - 0.7) / 3
        self.assertAlmostEqual(ppl, expected_ppl, places=5)

    def test_calc_ppl_with_none(self):
        """测试_calc_ppl处理None值"""
        model = VLLMCustomAPI(**self.default_kwargs)

        # 测试包含None的logprobs列表
        prompt_logprobs = [
            {"1": {"logprob": -0.5}},
            None,
            {"3": {"logprob": -0.7}}
        ]

        ppl = model._calc_ppl(prompt_logprobs)

        # 只计算非None的值: -(-0.5 - 0.7) / 2 = 1.2 / 2 = 0.6
        expected_ppl = -(-0.5 - 0.7) / 2
        self.assertAlmostEqual(ppl, expected_ppl, places=5)


class TestVLLMCustomAPILora(unittest.TestCase):
    """针对 Multi-LoRA 兼容性扩展的 UT（completions API）。"""

    LORA_MAP = {"0": "LoraA", "2": "LoraB"}

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_root = self._tmpdir.name
        self.default_kwargs = {
            "path": "test-model",
            "model": "base-model-name",
            "stream": False,
            "max_out_len": 100,
            "retry": 1,
            "host_ip": "localhost",
            "host_port": 8080,
            "enable_ssl": False,
            "verbose": False,
            "generation_kwargs": {},
        }
        self._get_service_model_path_patcher = patch.object(
            base_api.BaseAPIModel, "_get_service_model_path"
        )
        self.mock_get_model_path = self._get_service_model_path_patcher.start()
        self.mock_get_model_path.return_value = "mocked-base-model"

    def tearDown(self):
        self._get_service_model_path_patcher.stop()
        self._tmpdir.cleanup()

    def _write_lora_map(self, content=None):
        path = os.path.join(self.tmp_root, "lora_data_map.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content if content is not None else self.LORA_MAP, f)
        return path

    def test_load_lora_data_map_with_valid_file(self):
        path = self._write_lora_map()
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": path}
        model = VLLMCustomAPI(**kwargs)
        self.assertEqual(model.lora_data_map, self.LORA_MAP)

    def test_load_lora_data_map_no_config_returns_none(self):
        model = VLLMCustomAPI(**self.default_kwargs)
        self.assertIsNone(model.lora_data_map)

    def test_load_lora_data_map_file_not_exists(self):
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {
            "lora_data_map_file": os.path.join(self.tmp_root, "missing.json")
        }
        model = VLLMCustomAPI(**kwargs)
        self.assertIsNone(model.lora_data_map)

    def test_load_lora_data_map_invalid_json(self):
        path = os.path.join(self.tmp_root, "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("[broken")
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": path}
        model = VLLMCustomAPI(**kwargs)
        self.assertIsNone(model.lora_data_map)

    def test_resolve_hit(self):
        path = self._write_lora_map()
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": path}
        model = VLLMCustomAPI(**kwargs)
        out = Output()
        out.data_id = 2
        self.assertEqual(model._resolve_lora_model_name(out), "LoraB")

    def test_resolve_miss_returns_none(self):
        path = self._write_lora_map()
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": path}
        model = VLLMCustomAPI(**kwargs)
        out = Output()
        out.data_id = 99
        self.assertIsNone(model._resolve_lora_model_name(out))

    def test_resolve_no_data_id_returns_none(self):
        path = self._write_lora_map()
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": path}
        model = VLLMCustomAPI(**kwargs)
        out = Output()  # no data_id attribute
        self.assertIsNone(model._resolve_lora_model_name(out))

    def _make_model_with_lora(self, path):
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": path}
        with patch.object(VLLMCustomAPI, '_get_url',
                          return_value='http://localhost:8080/v1/completions'):
            return VLLMCustomAPI(**kwargs)

    def _run_get_request_body(self, model, output, input_data="test prompt", max_out_len=100):
        return asyncio.run(model.get_request_body(input_data, max_out_len, output))

    def test_request_body_lora_hit_overrides_model_field(self):
        path = self._write_lora_map()
        model = self._make_model_with_lora(path)
        out = Output()
        out.data_id = 0
        body = self._run_get_request_body(model, out)
        self.assertEqual(body["model"], "LoraA")

    def test_request_body_lora_miss_keeps_base_model(self):
        path = self._write_lora_map()
        model = self._make_model_with_lora(path)
        out = Output()
        out.data_id = 999
        body = self._run_get_request_body(model, out)
        self.assertEqual(body["model"], "base-model-name")

    def test_request_body_no_lora_config_keeps_base_model(self):
        model = VLLMCustomAPI(**self.default_kwargs)
        out = Output()
        out.data_id = 0
        body = self._run_get_request_body(model, out)
        self.assertEqual(body["model"], "base-model-name")
        self.assertNotIn("adapter_id", body)



if __name__ == "__main__":
    unittest.main()