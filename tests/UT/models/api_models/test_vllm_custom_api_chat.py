import unittest
import asyncio
import json
import os
import tempfile
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List

from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.models.api_models import base_api
from ais_bench.benchmark.models.output import RequestOutput
from ais_bench.benchmark.utils.prompt import PromptList


class TestVLLMCustomAPIChat(unittest.TestCase):
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
        
        # 模拟uuid生成
        self.uuid_patcher = patch('uuid.uuid4')
        self.mock_uuid = self.uuid_patcher.start()
        self.mock_uuid.return_value = uuid.UUID('12345678-1234-5678-1234-567812345678')
    
    def tearDown(self):
        self._get_service_model_path_patcher.stop()
        self.uuid_patcher.stop()
    
    def test_init_default_parameters(self):
        """测试使用默认参数初始化"""
        with patch.object(VLLMCustomAPIChat, '_get_url', return_value='http://localhost:8080/v1/chat/completions'):
            model = VLLMCustomAPIChat()
            
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
    
    def test_init_custom_parameters(self):
        """测试使用自定义参数初始化"""
        custom_kwargs = self.default_kwargs.copy()
        custom_kwargs["generation_kwargs"] = {"temperature": 0.7, "top_p": 0.9}
        custom_kwargs["meta_template"] = {
            "round": [{"role": "USER", "api_role": "user"}, {"role": "ASSISTANT", "api_role": "assistant"}],
            "reserved_roles": [{"role": "SYSTEM", "api_role": "system"}]
        }
        
        with patch.object(VLLMCustomAPIChat, '_get_url', return_value='http://localhost:8080/v1/chat/completions'):
            model = VLLMCustomAPIChat(**custom_kwargs)
            
            # 验证自定义参数是否正确设置
            self.assertEqual(model.path, "test-model")
            self.assertEqual(model.model, "test-model-name")  # 应该使用提供的model参数
            self.assertEqual(model.generation_kwargs, {"temperature": 0.7, "top_p": 0.9})
            self.assertEqual(model.meta_template, custom_kwargs["meta_template"])
    
    def test_get_url(self):
        """测试_get_url方法"""
        model = VLLMCustomAPIChat(path="test-path", host_ip="127.0.0.1", host_port=9000)
        url = model._get_url()
        self.assertEqual(url, "http://127.0.0.1:9000/v1/chat/completions")
        
        # 测试SSL情况
        model = VLLMCustomAPIChat(path="test-path", enable_ssl=True)
        url = model._get_url()
        self.assertTrue(url.startswith("https://"))
    
    async def test_get_request_body_string_input(self):
        """测试使用字符串输入调用get_request_body方法"""
        model = VLLMCustomAPIChat(**self.default_kwargs)
        output = RequestOutput()
        
        request_body = await model.get_request_body("test prompt", 100, output)
        
        # 验证请求体是否正确构建
        self.assertEqual(request_body["stream"], False)
        self.assertEqual(request_body["model"], "test-model-name")
        self.assertEqual(request_body["max_tokens"], 100)
        self.assertEqual(len(request_body["messages"]), 1)
        self.assertEqual(request_body["messages"][0]["role"], "user")
        self.assertEqual(request_body["messages"][0]["content"], "test prompt")
        self.assertEqual(output.input, request_body["messages"])
    
    async def test_get_request_body_prompt_list_input(self):
        """测试使用PromptList输入调用get_request_body方法"""
        model = VLLMCustomAPIChat(**self.default_kwargs)
        output = RequestOutput()
        
        # 创建一个包含HUMAN、BOT和SYSTEM角色的PromptList
        prompt_list = [
            {"role": "SYSTEM", "prompt": "You are a helpful assistant."},
            {"role": "HUMAN", "prompt": "Hello, how are you?"},
            {"role": "BOT", "prompt": "I'm doing well, thank you!"},
            {"role": "HUMAN", "prompt": "What's your name?"}
        ]
        
        request_body = await model.get_request_body(prompt_list, 100, output)
        
        # 验证请求体是否正确构建
        self.assertEqual(len(request_body["messages"]), 4)
        self.assertEqual(request_body["messages"][0]["role"], "system")
        self.assertEqual(request_body["messages"][0]["content"], "You are a helpful assistant.")
        self.assertEqual(request_body["messages"][1]["role"], "user")
        self.assertEqual(request_body["messages"][1]["content"], "Hello, how are you?")
        self.assertEqual(request_body["messages"][2]["role"], "assistant")
        self.assertEqual(request_body["messages"][2]["content"], "I'm doing well, thank you!")
        self.assertEqual(request_body["messages"][3]["role"], "user")
        self.assertEqual(request_body["messages"][3]["content"], "What's your name?")
    
    async def test_get_request_body_max_out_len_zero(self):
        """测试max_out_len <= 0的情况"""
        model = VLLMCustomAPIChat(**self.default_kwargs)
        output = RequestOutput()
        
        # 测试max_out_len = 0
        request_body = await model.get_request_body("test prompt", 0, output)
        self.assertEqual(request_body, "")
        
        # 测试max_out_len = -1
        request_body = await model.get_request_body("test prompt", -1, output)
        self.assertEqual(request_body, "")
    
    async def test_get_request_body_stream_enabled(self):
        """测试启用stream时的请求体构建"""
        kwargs = self.default_kwargs.copy()
        kwargs["stream"] = True
        model = VLLMCustomAPIChat(**kwargs)
        output = RequestOutput()
        
        request_body = await model.get_request_body("test prompt", 100, output)
        
        # 验证stream选项是否正确设置
        self.assertTrue(request_body["stream"])
        self.assertIn("stream_options", request_body)
        self.assertEqual(request_body["stream_options"]["include_usage"], True)

    def test_response_anomaly_requests_vllm_token_ids(self):
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {
            "response_anomaly_enabled": True,
            "return_token_ids": False,
            "return_tokens_as_token_ids": False,
        }
        model = VLLMCustomAPIChat(**kwargs)

        request_body = asyncio.run(
            model.get_request_body("test prompt", 100, RequestOutput())
        )

        self.assertTrue(request_body["return_token_ids"])
        self.assertTrue(request_body["return_tokens_as_token_ids"])
        self.assertNotIn("response_anomaly_enabled", request_body)
    
    async def test_parse_stream_response(self):
        """测试parse_stream_response方法"""
        model = VLLMCustomAPIChat(**self.default_kwargs)
        output = RequestOutput()
        
        # 创建模拟的流式响应内容
        stream_response = {
            "choices": [
                {
                    "delta": {
                        "content": "Hello, ",
                        "reasoning_content": "Let me think about this."
                    }
                }
            ]
        }
        
        await model.parse_stream_response(stream_response, output)
        
        # 验证内容是否正确解析
        self.assertEqual(output.content, "Hello, ")
        self.assertEqual(output.reasoning_content, "Let me think about this.")
        
        # 测试包含usage信息的响应
        usage_response = {
            "choices": [
                {
                    "delta": {
                        "content": "world!"
                    }
                }
            ],
            "usage": {
                "completion_tokens": 5
            }
        }
        
        await model.parse_stream_response(usage_response, output)
        
        # 验证内容累积和token计数
        self.assertEqual(output.content, "Hello, world!")
        self.assertEqual(output.output_tokens, 5)
        
        # 测试没有content或reasoning_content的情况
        empty_response = {
            "choices": [
                {
                    "delta": {}
                }
            ]
        }
        
        await model.parse_stream_response(empty_response, output)
        
        # 验证内容没有变化
        self.assertEqual(output.content, "Hello, world!")
        self.assertEqual(output.reasoning_content, "Let me think about this.")
    
    async def test_parse_text_response(self):
        """测试parse_text_response方法"""
        model = VLLMCustomAPIChat(**self.default_kwargs)
        output = RequestOutput()
        
        # 创建模拟的文本响应内容
        text_response = {
            "choices": [
                {
                    "message": {
                        "content": "Hello, world!",
                        "reasoning_content": "This is my response."
                    }
                }
            ],
            "usage": {
                "completion_tokens": 5
            }
        }
        
        await model.parse_text_response(text_response, output)
        
        # 验证内容是否正确解析
        self.assertEqual(output.content, "Hello, world!")
        self.assertEqual(output.reasoning_content, "This is my response.")
        self.assertEqual(output.output_tokens, 5)
        
        # 测试多个choices的情况
        multi_choice_response = {
            "choices": [
                {
                    "message": {
                        "content": "First response"
                    }
                },
                {
                    "message": {
                        "content": "Second response",
                        "reasoning_content": "Additional reasoning"
                    }
                }
            ]
        }
        
        await model.parse_text_response(multi_choice_response, output)
        
        # 验证内容是否正确累积
        self.assertEqual(output.content, "Hello, world!First responseSecond response")
        self.assertEqual(output.reasoning_content, "This is my response.Additional reasoning")
    
    def test_get_service_model_path_call(self):
        """测试当不提供model_name时，是否调用_get_service_model_path"""
        # 不提供model参数
        kwargs = self.default_kwargs.copy()
        kwargs.pop("model")
        
        with patch.object(VLLMCustomAPIChat, '_get_url', return_value='http://localhost:8080/v1/chat/completions'):
            model = VLLMCustomAPIChat(**kwargs)
            # 验证_get_service_model_path被调用
            self.mock_get_model_path.assert_called()
            # 验证model属性被设置为_get_service_model_path的返回值
            self.assertEqual(model.model, "mocked-model-path")
    
    def test_init_with_empty_meta_template(self):
        """测试使用空meta_template初始化"""
        kwargs = self.default_kwargs.copy()
        kwargs["meta_template"] = None
        
        with patch.object(VLLMCustomAPIChat, '_get_url', return_value='http://localhost:8080/v1/chat/completions'):
            model = VLLMCustomAPIChat(**kwargs)
            # 验证默认meta_template被设置
            self.assertIn("round", model.meta_template)
            self.assertIn("reserved_roles", model.meta_template)
            self.assertEqual(len(model.meta_template["round"]), 2)
            self.assertEqual(len(model.meta_template["reserved_roles"]), 1)

    # 运行异步测试的辅助方法
    def run_async_test(self, coroutine):
        return asyncio.run(coroutine)
    
    # 包装异步测试方法
    def test_get_request_body_string_input_wrapper(self):
        self.run_async_test(self.test_get_request_body_string_input())
    
    def test_get_request_body_prompt_list_input_wrapper(self):
        self.run_async_test(self.test_get_request_body_prompt_list_input())
    
    def test_get_request_body_max_out_len_zero_wrapper(self):
        self.run_async_test(self.test_get_request_body_max_out_len_zero())
    
    def test_get_request_body_stream_enabled_wrapper(self):
        self.run_async_test(self.test_get_request_body_stream_enabled())
    
    def test_parse_stream_response_wrapper(self):
        self.run_async_test(self.test_parse_stream_response())
    
    def test_parse_text_response_wrapper(self):
        self.run_async_test(self.test_parse_text_response())

    async def test_parse_logprobs_with_data(self):
        """测试_parse_logprobs正确解析chat API logprobs响应"""
        model = VLLMCustomAPIChat(**self.default_kwargs)
        output = RequestOutput()

        content_list = [
            {"token": "B", "logprob": -0.5, "bytes": [66], "top_logprobs": [{"token": "B", "logprob": -0.5}, {"token": "A", "logprob": -2.1}]}
        ]
        choice = {
            "message": {"content": "B"},
            "logprobs": {"content": content_list}
        }

        await model._parse_logprobs(choice, output)

        # chat API 直接透传 lp.content，保持 vLLM 原始结构
        self.assertEqual(output.origin_logprobs, content_list)

    async def test_parse_logprobs_without_logprobs_field(self):
        """测试_parse_logprobs在响应无logprobs字段时不报错"""
        model = VLLMCustomAPIChat(**self.default_kwargs)
        output = RequestOutput()

        choice = {"message": {"content": "B"}}

        await model._parse_logprobs(choice, output)

        self.assertEqual(output.origin_logprobs, [])

    async def test_parse_logprobs_enabled_but_missing_warns(self):
        """测试开启logprobs但响应缺失时，写入logprobs_warning到extra_details_data"""
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"logprobs": True}
        model = VLLMCustomAPIChat(**kwargs)
        output = RequestOutput()

        choice = {"message": {"content": "B"}}  # 无 logprobs 字段

        await model._parse_logprobs(choice, output)

        self.assertEqual(output.origin_logprobs, [])
        self.assertIn("logprobs_warning", output.extra_details_data)
        self.assertIn("logprobs is enabled", output.extra_details_data["logprobs_warning"])

    async def test_parse_logprobs_with_none_content(self):
        """测试_parse_logprobs保留None项（predefined token）"""
        model = VLLMCustomAPIChat(**self.default_kwargs)
        output = RequestOutput()

        content_list = [
            None,
            {"token": "B", "logprob": -0.3, "top_logprobs": [{"token": "B", "logprob": -0.3}]}
        ]
        choice = {
            "message": {"content": "AB"},
            "logprobs": {"content": content_list}
        }

        await model._parse_logprobs(choice, output)

        # 直接透传，None 项保留
        self.assertEqual(output.origin_logprobs, content_list)

    async def test_parse_text_response_with_logprobs(self):
        """测试parse_text_response正确调用_parse_logprobs"""
        model = VLLMCustomAPIChat(**self.default_kwargs)
        output = RequestOutput()

        content_list = [
            {"token": "B", "logprob": -0.5, "top_logprobs": [{"token": "B", "logprob": -0.5}]}
        ]
        response = {
            "choices": [{
                "message": {"content": "B"},
                "logprobs": {"content": content_list}
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1}
        }

        await model.parse_text_response(response, output)

        self.assertEqual(output.content, "B")
        self.assertEqual(output.input_tokens, 10)
        self.assertEqual(output.output_tokens, 1)
        self.assertEqual(output.origin_logprobs, content_list)

    def test_parse_logprobs_with_data_wrapper(self):
        self.run_async_test(self.test_parse_logprobs_with_data())

    def test_parse_logprobs_without_logprobs_field_wrapper(self):
        self.run_async_test(self.test_parse_logprobs_without_logprobs_field())

    def test_parse_logprobs_enabled_but_missing_warns_wrapper(self):
        self.run_async_test(self.test_parse_logprobs_enabled_but_missing_warns())

    def test_parse_logprobs_with_none_content_wrapper(self):
        self.run_async_test(self.test_parse_logprobs_with_none_content())

    def test_parse_text_response_with_logprobs_wrapper(self):
        self.run_async_test(self.test_parse_text_response_with_logprobs())

    def test_calc_ppl(self):
        """测试_calc_ppl方法"""
        model = VLLMCustomAPIChat(**self.default_kwargs)
        
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
        model = VLLMCustomAPIChat(**self.default_kwargs)

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


class TestVLLMCustomAPIChatLora(unittest.TestCase):
    """针对 Multi-LoRA 兼容性扩展的 UT（base model 内嵌，无新增子类）。"""

    LORA_MAP = {"0": "LoraA", "1": "LoraB", "6": "LoraA"}

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
        # Avoid actual service discovery on instantiation.
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

    # ---------- _load_lora_data_map ----------

    def test_load_lora_data_map_with_valid_file(self):
        path = self._write_lora_map()
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": path}
        model = VLLMCustomAPIChat(**kwargs)
        self.assertEqual(model.lora_data_map, self.LORA_MAP)

    def test_load_lora_data_map_missing_key(self):
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"temperature": 0.7}
        model = VLLMCustomAPIChat(**kwargs)
        self.assertIsNone(model.lora_data_map)

    def test_load_lora_data_map_empty_generation_kwargs(self):
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = None
        model = VLLMCustomAPIChat(**kwargs)
        self.assertIsNone(model.lora_data_map)

    def test_load_lora_data_map_none_value(self):
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": None}
        model = VLLMCustomAPIChat(**kwargs)
        self.assertIsNone(model.lora_data_map)

    def test_load_lora_data_map_empty_string(self):
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": ""}
        model = VLLMCustomAPIChat(**kwargs)
        self.assertIsNone(model.lora_data_map)

    def test_load_lora_data_map_file_not_exists(self):
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {
            "lora_data_map_file": os.path.join(self.tmp_root, "no_such_file.json")
        }
        model = VLLMCustomAPIChat(**kwargs)
        self.assertIsNone(model.lora_data_map)

    def test_load_lora_data_map_invalid_json(self):
        path = os.path.join(self.tmp_root, "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("not json {{{")
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": path}
        model = VLLMCustomAPIChat(**kwargs)
        self.assertIsNone(model.lora_data_map)

    def test_load_lora_data_map_empty_dict(self):
        path = self._write_lora_map(content={})
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": path}
        model = VLLMCustomAPIChat(**kwargs)
        self.assertEqual(model.lora_data_map, {})

    # ---------- _resolve_lora_model_name ----------

    def test_resolve_hit(self):
        path = self._write_lora_map()
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": path}
        model = VLLMCustomAPIChat(**kwargs)
        out = RequestOutput()
        out.data_id = 0
        self.assertEqual(model._resolve_lora_model_name(out), "LoraA")
        out.data_id = 1
        self.assertEqual(model._resolve_lora_model_name(out), "LoraB")
        out.data_id = 6
        self.assertEqual(model._resolve_lora_model_name(out), "LoraA")

    def test_resolve_miss_returns_none(self):
        path = self._write_lora_map()
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": path}
        model = VLLMCustomAPIChat(**kwargs)
        out = RequestOutput()
        out.data_id = 999
        self.assertIsNone(model._resolve_lora_model_name(out))

    def test_resolve_no_map_returns_none(self):
        model = VLLMCustomAPIChat(**self.default_kwargs)
        out = RequestOutput()
        out.data_id = 0
        self.assertIsNone(model._resolve_lora_model_name(out))

    def test_resolve_output_without_data_id_returns_none(self):
        path = self._write_lora_map()
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": path}
        model = VLLMCustomAPIChat(**kwargs)
        out = RequestOutput()  # no data_id attribute
        self.assertIsNone(model._resolve_lora_model_name(out))

    # ---------- get_request_body LoRA injection ----------

    def _make_model_with_lora(self, path):
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = {"lora_data_map_file": path}
        with patch.object(VLLMCustomAPIChat, '_get_url',
                          return_value='http://localhost:8080/v1/chat/completions'):
            return VLLMCustomAPIChat(**kwargs)

    def _run_get_request_body(self, model, output, input_data="test prompt", max_out_len=100):
        return asyncio.run(model.get_request_body(input_data, max_out_len, output))

    def test_request_body_lora_hit_overrides_model_field(self):
        path = self._write_lora_map()
        model = self._make_model_with_lora(path)
        out = RequestOutput()
        out.data_id = 0
        body = self._run_get_request_body(model, out)
        self.assertEqual(body["model"], "LoraA")

    def test_request_body_lora_miss_keeps_base_model(self):
        path = self._write_lora_map()
        model = self._make_model_with_lora(path)
        out = RequestOutput()
        out.data_id = 999  # not in map
        body = self._run_get_request_body(model, out)
        # Falls back to base model name.
        self.assertEqual(body["model"], "base-model-name")

    def test_request_body_no_lora_config_keeps_base_model(self):
        model = VLLMCustomAPIChat(**self.default_kwargs)
        out = RequestOutput()
        out.data_id = 0
        body = self._run_get_request_body(model, out)
        self.assertEqual(body["model"], "base-model-name")
        # adapter_id should NEVER appear in the vLLM body even with data_id set.
        self.assertNotIn("adapter_id", body)

    def test_request_body_lora_hit_with_promptlist(self):
        path = self._write_lora_map()
        model = self._make_model_with_lora(path)
        out = RequestOutput()
        out.data_id = 1
        prompt_list = [
            {"role": "HUMAN", "prompt": "Hello"},
        ]
        body = self._run_get_request_body(model, out, input_data=prompt_list)
        self.assertEqual(body["model"], "LoraB")
        self.assertEqual(len(body["messages"]), 1)



if __name__ == "__main__":
    unittest.main()
