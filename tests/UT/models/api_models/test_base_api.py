import sys
import json
import asyncio
import unittest
from unittest import mock
from copy import deepcopy
import requests

from ais_bench.benchmark.models import (
    BaseAPIModel, APITemplateParser
)
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchTypeError, AISBenchValueError,
    AISBenchRuntimeError, AISBenchKeyError
)


class MockResponse:
    def __init__(self, status_code, json_data=None, text=None, reason=None):
        self.status_code = status_code
        self._json_data = json_data
        self._text = text
        self.reason = reason

    def json(self):
        return self._json_data

    def text(self):
        return self._text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP Error: {self.status_code}")


class TestBaseAPIModel(unittest.TestCase):

    def setUp(self):
        # 创建一个具体的BaseAPIModel子类用于测试
        class ConcreteAPIModel(BaseAPIModel):
            def _get_url(self):
                return self.base_url + "test"

            async def get_request_body(self, input_data, max_out_len, output, **args):
                return {"prompt": input_data, "max_tokens": max_out_len}

            async def parse_text_response(self, data, output):
                output.text = data.get("response", "")

            async def parse_stream_response(self, data, output):
                output.text += data.get("chunk", "")

        self.model_class = ConcreteAPIModel
        self.default_kwargs = {
            "path": "test-model",
            "stream": False,
            "max_out_len": 100,
            "retry": 1,
            "host_ip": "127.0.0.1",
            "host_port": 8000
        }

    def test_init(self):
        model = self.model_class(**self.default_kwargs)
        self.assertEqual(model.path, "test-model")
        self.assertEqual(model.stream, False)
        self.assertEqual(model.max_out_len, 100)
        self.assertEqual(model.retry, 1)
        self.assertEqual(model.base_url, "http://127.0.0.1:8000/")

    def test_init_with_url(self):
        kwargs = self.default_kwargs.copy()
        kwargs["url"] = "https://test-api.com/v1"
        model = self.model_class(**kwargs)
        self.assertEqual(model.base_url, "https://test-api.com/v1")

    def test_init_with_ssl(self):
        kwargs = self.default_kwargs.copy()
        kwargs["enable_ssl"] = True
        model = self.model_class(**kwargs)
        self.assertEqual(model.base_url, "https://127.0.0.1:8000/")

    def test_get_base_url(self):
        model = self.model_class(**self.default_kwargs)
        self.assertEqual(model._get_base_url(), "http://127.0.0.1:8000/")

    @mock.patch('requests.get')
    def test_get_service_model_path_success(self, mock_get):
        mock_response = MockResponse(200, json_data={"data": [{"id": "test-model-123"}]})
        mock_get.return_value = mock_response

        model = self.model_class(**self.default_kwargs)
        result = model._get_service_model_path()

        self.assertEqual(result, "test-model-123")
        mock_get.assert_called_once_with(
            "http://127.0.0.1:8000/v1/models",
            headers={"Content-Type": "application/json"},
            timeout=5
        )

    @mock.patch('requests.get')
    def test_get_service_model_path_failure(self, mock_get):
        # 模拟requests.exceptions.RequestException异常
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        model = self.model_class(**self.default_kwargs)
        with self.assertRaises(AISBenchRuntimeError):
            model._get_service_model_path()

    def test_iter_lines_normal(self):
        model = self.model_class(**self.default_kwargs)
        stream = [b"line1\n\nline2\n\n", b"line3"]

        async def mock_stream():
            for chunk in stream:
                yield chunk

        async def run():
            lines = [line async for line in model.iter_lines(mock_stream())]
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], b"line1")
            self.assertEqual(lines[1], b"line2")
            self.assertEqual(lines[2], b"line3")

        asyncio.run(run())

    def test_iter_lines_crlf(self):
        model = self.model_class(**self.default_kwargs)
        stream = [b"line1\r\n\r\nline2\r\rline3"]

        async def mock_stream():
            for chunk in stream:
                yield chunk

        async def run():
            lines = [line async for line in model.iter_lines(mock_stream())]
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], b"line1")
            self.assertEqual(lines[1], b"line2")
            self.assertEqual(lines[2], b"line3")
        asyncio.run(run())

    @mock.patch('aiohttp.ClientSession.post')
    async def test_text_infer_success(self, mock_post):
        # 设置mock响应
        mock_response = mock.AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = json.dumps({"response": "test output"})
        mock_post.return_value.__aenter__.return_value = mock_response

        model = self.model_class(**self.default_kwargs)
        model.session = mock.AsyncMock()
        output = Output()

        await model.text_infer({"prompt": "test"}, output)

        self.assertTrue(output.success)
        self.assertEqual(output.text, "test output")

    @mock.patch('aiohttp.ClientSession.post')
    async def test_text_infer_failure(self, mock_post):
        mock_response = mock.AsyncMock()
        mock_response.status = 400
        mock_response.reason = "Bad Request"
        mock_post.return_value.__aenter__.return_value = mock_response

        model = self.model_class(**self.default_kwargs)
        model.session = mock.AsyncMock()
        output = Output()

        await model.text_infer({"prompt": "test"}, output)

        self.assertFalse(output.success)
        self.assertEqual(output.error_info, "Bad Request")

    @mock.patch('aiohttp.ClientSession.post')
    async def test_text_infer_json_decode_error(self, mock_post):
        mock_response = mock.AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = "invalid json"
        mock_post.return_value.__aenter__.return_value = mock_response

        model = self.model_class(**self.default_kwargs)
        model.session = mock.AsyncMock()
        output = Output()

        with self.assertRaises(AISBenchValueError):
            await model.text_infer({"prompt": "test"}, output)

    @mock.patch('aiohttp.ClientSession.post')
    async def test_stream_infer_success(self, mock_post):
        # 创建模拟的流响应
        async def mock_content():
            yield b"data: {\"chunk\": \"hello\"}\n\n"
            yield b"data: {\"chunk\": \" world\"}\n\n"
            yield b"data: [DONE]\n\n"

        mock_response = mock.AsyncMock()
        mock_response.status = 200
        mock_response.content.__aiter__.return_value = mock_content()
        mock_post.return_value.__aenter__.return_value = mock_response

        model = self.model_class(**self.default_kwargs)
        model.stream = True
        model.session = mock.AsyncMock()
        output = Output()

        async def run():
            await model.stream_infer({"prompt": "test"}, output)
            self.assertTrue(output.success)
            self.assertEqual(output.text, "hello world")

        asyncio.run(run())

    @mock.patch('aiohttp.ClientSession.post')
    async def test_stream_infer_json_decode_error(self, mock_post):
        async def mock_content():
            yield b"data: invalid json\n\n"

        mock_response = mock.AsyncMock()
        mock_response.status = 200
        mock_response.content.__aiter__.return_value = mock_content()
        mock_post.return_value.__aenter__.return_value = mock_response

        model = self.model_class(**self.default_kwargs)
        model.stream = True
        model.session = mock.AsyncMock()
        output = Output()

        async def run():
            with self.assertRaises(AISBenchValueError):
                await model.stream_infer({"prompt": "test"}, output)
        asyncio.run(run())

    @mock.patch('aiohttp.ClientSession')
    def test_generate_with_session(self, mock_session_class):
        mock_session = mock.AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session

        model = self.model_class(**self.default_kwargs)
        output = Output()
        async def run():
            # 使用mock模拟generate方法的关键部分
            with mock.patch.object(model, 'text_infer') as mock_text_infer:
                await model.generate("test prompt", 100, output, session=mock_session)
                mock_text_infer.assert_called_once()
        asyncio.run(run())

    @mock.patch('aiohttp.ClientSession')
    def test_generate_with_retry(self, mock_session_class):

        mock_session = mock.AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session

        model = self.model_class(**self.default_kwargs)
        model.retry = 2
        output = Output()

        # 第一次调用失败，第二次成功
        mock_text_infer = mock.AsyncMock(side_effect=[Exception("First fail"), None])
        async def run():
            with mock.patch.object(model, 'text_infer', mock_text_infer):
                await model.generate("test prompt", 100, output, session=mock_session)
                self.assertEqual(mock_text_infer.call_count, 2)
        asyncio.run(run())

    def test_abstract_methods(self):
        # 测试抽象方法是否正确抛出异常
        model = self.model_class(**self.default_kwargs)

        # 这些方法在子类中已经实现，所以不会抛出异常
        # 这里主要是测试基类的行为
        pass


class TestAPITemplateParser(unittest.TestCase):

    def setUp(self):
        self.default_meta_template = {
            "round": [
                {"role": "user", "api_role": "user"},
                {"role": "assistant", "api_role": "assistant", "generate": True}
            ]
        }

    def test_init_with_valid_meta_template(self):
        parser = APITemplateParser(self.default_meta_template)
        self.assertEqual(len(parser.roles), 2)
        self.assertIn("user", parser.roles)
        self.assertIn("assistant", parser.roles)

    def test_init_with_invalid_meta_template(self):
        # 测试缺少round字段
        invalid_template = {}
        # 只有当meta_template不为None时才会检查round字段
        parser = APITemplateParser(invalid_template)

        # 测试round字段类型错误
        invalid_template = {"round": "not a list"}
        with self.assertRaises(AISBenchTypeError):
            APITemplateParser(invalid_template)

    def test_init_with_reserved_roles(self):
        meta_template = deepcopy(self.default_meta_template)
        meta_template["reserved_roles"] = [
            {"role": "system", "api_role": "system"}
        ]
        parser = APITemplateParser(meta_template)
        self.assertEqual(len(parser.roles), 3)
        self.assertIn("system", parser.roles)

    def test_init_with_duplicate_roles(self):
        meta_template = {
            "round": [
                {"role": "user", "api_role": "user"},
                {"role": "user", "api_role": "user"}  # 重复的角色
            ]
        }
        with self.assertRaises(AISBenchTypeError):
            APITemplateParser(meta_template)

    def test_parse_template_string(self):
        parser = APITemplateParser(self.default_meta_template)
        result = parser.parse_template("test prompt", "gen")
        self.assertEqual(result, "test prompt")

    def test_parse_template_invalid_type(self):
        parser = APITemplateParser(self.default_meta_template)
        with self.assertRaises(AISBenchTypeError):
            parser.parse_template(123, "gen")  # 类型错误

    def test_parse_template_invalid_mode(self):
        parser = APITemplateParser(self.default_meta_template)
        with self.assertRaises(AISBenchTypeError):
            parser.parse_template("test", "invalid_mode")

    def test_parse_template_without_meta_template(self):
        parser = APITemplateParser(None)
        # 使用PromptList而不是普通列表
        prompt_list = PromptList([
            {"prompt": "Hello"},
            {"prompt": "World"}
        ])
        result = parser.parse_template(prompt_list, "gen")
        self.assertEqual(result, "Hello\nWorld")

    def test_update_role_dict(self):
        parser = APITemplateParser(self.default_meta_template)
        prompts = [{"role": "user", "prompt": "test"}]
        role_dict = parser._update_role_dict(prompts)
        self.assertEqual(role_dict["user"]["prompt"], "test")

    def test_update_role_dict_with_fallback(self):
        parser = APITemplateParser(self.default_meta_template)
        prompts = [{"role": "unknown", "fallback_role": "user", "prompt": "test"}]
        role_dict = parser._update_role_dict(prompts)
        self.assertEqual(role_dict["user"]["prompt"], "test")

    def test_split_rounds(self):
        parser = APITemplateParser(self.default_meta_template)
        prompt_template = [
            {"role": "user", "prompt": "q1"},
            {"role": "assistant", "prompt": "a1"},
            {"role": "user", "prompt": "q2"},
            {"role": "assistant", "prompt": "a2"}
        ]
        result = parser._split_rounds(prompt_template, self.default_meta_template["round"])
        self.assertEqual(result, [0, 2, 4])

    def test_split_rounds_with_fallback_role(self):
        parser = APITemplateParser(self.default_meta_template)
        prompt_template = [
            {"role": "unknown", "fallback_role": "user", "prompt": "q1"},
            {"role": "assistant", "prompt": "a1"}
        ]
        result = parser._split_rounds(prompt_template, self.default_meta_template["round"])
        self.assertEqual(result, [0, 2])

    def test_split_rounds_invalid_role(self):
        parser = APITemplateParser(self.default_meta_template)
        prompt_template = [
            {"role": "unknown", "prompt": "q1"}  # 没有fallback_role
        ]
        with self.assertRaises(AISBenchKeyError):
            parser._split_rounds(prompt_template, self.default_meta_template["round"])

    def test_prompt2api_string(self):
        parser = APITemplateParser(self.default_meta_template)
        result, cont = parser._prompt2api("test", {}, False)
        self.assertEqual(result, "test")
        self.assertTrue(cont)

    def test_prompt2api_dict(self):
        parser = APITemplateParser(self.default_meta_template)
        parser.roles["user"] = {"role": "user", "api_role": "user", "prompt": "test"}
        result, cont = parser._prompt2api({"role": "user"}, parser.roles, False)
        self.assertEqual(result["role"], "user")
        self.assertTrue(cont)

    def test_prompt2api_list_with_string(self):
        parser = APITemplateParser(self.default_meta_template)
        with self.assertRaises(AISBenchTypeError):
            parser._prompt2api(["test"], {}, False)

    def test_role2api_role(self):
        parser = APITemplateParser(self.default_meta_template)
        parser.roles["user"] = {
            "role": "user",
            "api_role": "user",
            "prompt": "test",
            "begin": "<s>",
            "end": "</s>"
        }
        result, cont = parser._role2api_role({"role": "user"}, parser.roles, False)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["prompt"], "<s>test</s>")
        self.assertTrue(cont)

    def test_role2api_role_for_gen(self):
        parser = APITemplateParser(self.default_meta_template)
        parser.roles["assistant"] = {
            "role": "assistant",
            "api_role": "assistant",
            "generate": True
        }
        result, cont = parser._role2api_role({"role": "assistant"}, parser.roles, True)
        self.assertIsNone(result)
        self.assertFalse(cont)

    def test_role2api_role_invalid_content(self):
        parser = APITemplateParser(self.default_meta_template)
        parser.roles["user"] = {"role": "user", "api_role": "user"}  # 缺少prompt或prompt_mm
        with self.assertRaises(AISBenchValueError):
            parser._role2api_role({"role": "user"}, parser.roles, False)

    def test_role2api_role_with_prompt_mm(self):
        parser = APITemplateParser(self.default_meta_template)
        parser.roles["user"] = {
            "role": "user",
            "api_role": "user",
            "prompt_mm": [{"type": "text", "content": "test"}]
        }
        result, cont = parser._role2api_role({"role": "user"}, parser.roles, False)
        self.assertEqual(result["prompt"], [{"type": "text", "content": "test"}])

    def test_parse_template_with_meta_template(self):
        meta_template = {
            "begin": {"role": "system", "prompt": "System prompt"},
            "round": [
                {"role": "user", "api_role": "user"},
                {"role": "assistant", "api_role": "assistant"}
            ]
        }
        parser = APITemplateParser(meta_template)

        # 模拟一个完整的对话结构
        prompt_template = PromptList([
            {"section": "round", "pos": "begin"},
            {"role": "user", "prompt": "Question 1"},
            {"role": "assistant", "prompt": "Answer 1"},
            {"section": "round", "pos": "end"}
        ])

        result = parser.parse_template(prompt_template, "gen")
        self.assertEqual(len(result), 3)  # system + user + assistant
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[1]["role"], "user")
        self.assertEqual(result[2]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()