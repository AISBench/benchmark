import unittest
import asyncio
import json
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



if __name__ == "__main__":
    unittest.main()