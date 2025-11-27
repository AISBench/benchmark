import sys
import json
import asyncio
import unittest
from unittest import mock
from copy import deepcopy

from ais_bench.benchmark.models import TGICustomAPI
from ais_bench.benchmark.models.api_models import tgi_api
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.models.output import Output


class TestTGICustomAPI(unittest.TestCase):

    def setUp(self):
        self.default_kwargs = {
            "path": "test-model",
            "stream": False,
            "max_out_len": 4096,
            "retry": 2,
            "host_ip": "localhost",
            "host_port": 8080,
            "generation_kwargs": {"temperature": 0.7, "top_p": 0.9}
        }

    def test_init_default_parameters(self):
        """测试使用默认参数初始化"""
        model = TGICustomAPI()
        self.assertEqual(model.path, "")
        self.assertFalse(model.stream)
        self.assertEqual(model.max_out_len, 4096)
        self.assertEqual(model.retry, 2)
        self.assertEqual(model.host_ip, "localhost")
        self.assertEqual(model.host_port, 8080)
        self.assertEqual(model.base_url, "http://localhost:8080/")
        self.assertEqual(model.url, "http://localhost:8080/generate")

    def test_init_custom_parameters(self):
        """测试使用自定义参数初始化"""
        kwargs = self.default_kwargs.copy()
        model = TGICustomAPI(**kwargs)
        self.assertEqual(model.path, "test-model")
        self.assertFalse(model.stream)
        self.assertEqual(model.max_out_len, 4096)
        self.assertEqual(model.retry, 2)
        self.assertEqual(model.host_ip, "localhost")
        self.assertEqual(model.host_port, 8080)
        self.assertEqual(model.base_url, "http://localhost:8080/")
        self.assertEqual(model.generation_kwargs, {"temperature": 0.7, "top_p": 0.9})
        self.assertEqual(model.url, "http://localhost:8080/generate")

    def test_init_with_ssl(self):
        """测试启用SSL的初始化"""
        kwargs = self.default_kwargs.copy()
        kwargs["enable_ssl"] = True
        model = TGICustomAPI(**kwargs)
        self.assertEqual(model.base_url, "https://localhost:8080/")
        self.assertEqual(model.url, "https://localhost:8080/generate")

    def test_init_with_url(self):
        """测试直接指定URL的初始化"""
        kwargs = self.default_kwargs.copy()
        kwargs["url"] = "https://custom-api.com/v1/"
        model = TGICustomAPI(**kwargs)
        self.assertEqual(model.base_url, "https://custom-api.com/v1/")
        self.assertEqual(model.url, "https://custom-api.com/v1/generate")

    def test_get_url_stream_true(self):
        """测试stream=True时的_get_url方法"""
        kwargs = self.default_kwargs.copy()
        kwargs["stream"] = True
        model = TGICustomAPI(**kwargs)
        url = model._get_url()
        self.assertEqual(url, "http://localhost:8080/generate_stream")

    def test_get_url_stream_false(self):
        """测试stream=False时的_get_url方法"""
        kwargs = self.default_kwargs.copy()
        kwargs["stream"] = False
        model = TGICustomAPI(**kwargs)
        url = model._get_url()
        self.assertEqual(url, "http://localhost:8080/generate")

    @mock.patch.object(tgi_api, 'LMTemplateParser')
    def test_init_with_meta_template(self, mock_template_parser):
        """测试带有meta_template的初始化"""
        kwargs = self.default_kwargs.copy()
        meta_template = {"round": [{"role": "user", "begin": "user:"}, {"role": "assistant", "begin": "assistant:"}]}
        kwargs["meta_template"] = meta_template

        model = TGICustomAPI(**kwargs)
        mock_template_parser.assert_called_once_with(meta_template)
        self.assertEqual(model.meta_template, meta_template)

    def test_get_request_body_string_input(self):
        """测试使用字符串输入获取请求体"""
        async def run():
            model = TGICustomAPI(**self.default_kwargs)
            output = Output()
            input_data = "test prompt"
            max_out_len = 100

            request_body = await model.get_request_body(input_data, max_out_len, output)

            # 验证请求体格式
            self.assertEqual(request_body["inputs"], "test prompt")
            self.assertEqual(request_body["parameters"]["max_new_tokens"], 100)
            self.assertEqual(request_body["parameters"]["temperature"], 0.7)
            # 验证output.input被正确设置
            self.assertEqual(output.input, "test prompt")
        asyncio.run(run())

    def test_get_request_body_promptlist_input(self):
        """测试使用PromptList输入获取请求体"""
        async def run():
            model = TGICustomAPI(**self.default_kwargs)
            output = Output()
            input_data = PromptList(["test prompt 1", "test prompt 2"])
            max_out_len = 100

            request_body = await model.get_request_body(input_data, max_out_len, output)

            # 验证请求体格式
            self.assertEqual(request_body["inputs"], input_data)
            self.assertEqual(request_body["parameters"]["max_new_tokens"], 100)
            # 验证output.input被正确设置
            self.assertEqual(output.input, input_data)
        asyncio.run(run())

    async def test_get_request_body_with_empty_generation_kwargs(self):
        """测试generation_kwargs为空的情况"""
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = None
        model = TGICustomAPI(**kwargs)
        output = Output()
        request_body = await model.get_request_body("test", 100, output)

        self.assertEqual(request_body["parameters"]["max_new_tokens"], 100)
        # 只有max_new_tokens，没有其他参数
        self.assertEqual(len(request_body["parameters"]), 1)

    def test_parse_text_response_with_generated_text(self):
        """测试解析带有generated_text的文本响应"""
        async def run():
            model = TGICustomAPI(**self.default_kwargs)
            output = Output()
            api_response = {"generated_text": "Hello world!"}

            await model.parse_text_response(api_response, output)

            self.assertEqual(output.content, "Hello world!")
        asyncio.run(run())

    def test_parse_text_response_with_empty_generated_text(self):
        """测试解析generated_text为空的文本响应"""
        async def run():
            model = TGICustomAPI(**self.default_kwargs)
            output = Output()
            api_response = {"generated_text": ""}

            await model.parse_text_response(api_response, output)

            self.assertEqual(output.content, "")
        asyncio.run(run())

    def test_parse_text_response_without_generated_text(self):
        """测试解析没有generated_text字段的文本响应"""
        async def run():
            model = TGICustomAPI(**self.default_kwargs)
            output = Output()
            api_response = {"other_field": "some value"}

            await model.parse_text_response(api_response, output)

            self.assertEqual(output.content, "")  # 应该使用默认空字符串
        asyncio.run(run())

    def test_parse_stream_response_with_generated_text(self):
        """测试解析带有generated_text的流式响应"""
        async def run():
            model = TGICustomAPI(**self.default_kwargs)
            output = Output()
            api_response = {"generated_text": "Hello world!"}

            await model.parse_stream_response(api_response, output)

            self.assertEqual(output.content, "Hello world!")
        asyncio.run(run())

    def test_parse_stream_response_with_empty_generated_text(self):
        """测试解析generated_text为空的流式响应"""
        async def run():
            model = TGICustomAPI(**self.default_kwargs)
            output = Output()
            api_response = {"generated_text": ""}

            await model.parse_stream_response(api_response, output)

            self.assertEqual(output.content, "")
        asyncio.run(run())

    def test_parse_stream_response_without_generated_text(self):
        """测试解析没有generated_text字段的流式响应"""
        async def run():
            model = TGICustomAPI(**self.default_kwargs)
            output = Output()
            api_response = {"other_field": "some value"}

            await model.parse_stream_response(api_response, output)

            self.assertEqual(output.content, "")  # 应该使用默认空字符串
        asyncio.run(run())

    @mock.patch('aiohttp.ClientSession')
    async def test_generate_integration_stream_false(self, mock_session_class):
        # 设置mock
        mock_session = mock.AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session

        model = TGICustomAPI(**self.default_kwargs)
        output = Output()

        # Mock text_infer方法
        with mock.patch.object(model, 'text_infer') as mock_text_infer:
            await model.generate("test prompt", 100, output)
            mock_text_infer.assert_called_once()

    @mock.patch('aiohttp.ClientSession')
    async def test_generate_integration_stream_true(self, mock_session_class):
        """集成测试stream=True时的generate方法"""
        # 设置mock
        mock_session = mock.AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session

        kwargs = self.default_kwargs.copy()
        kwargs["stream"] = True
        model = TGICustomAPI(**kwargs)
        output = Output()

        # Mock stream_infer方法
        with mock.patch.object(model, 'stream_infer') as mock_stream_infer:
            await model.generate("test prompt", 100, output)
            mock_stream_infer.assert_called_once()


if __name__ == '__main__':
    unittest.main()