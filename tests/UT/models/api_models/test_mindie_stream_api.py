import sys
import json
import asyncio
import os
import tempfile
import unittest
from unittest import mock
from copy import deepcopy

from ais_bench.benchmark.models import MindieStreamApi
from ais_bench.benchmark.models.api_models import mindie_stream_api
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.models.output import Output


class TestMindieStreamApi(unittest.TestCase):

    def setUp(self):
        self.default_kwargs = {
            "path": "test-model",
            "stream": True,
            "max_out_len": 4096,
            "retry": 2,
            "host_ip": "localhost",
            "host_port": 8080,
            "generation_kwargs": {"temperature": 0.7, "top_p": 0.9}
        }

    def test_init_default_parameters(self):
        """测试使用默认参数初始化"""
        model = MindieStreamApi()
        self.assertEqual(model.path, "")
        self.assertTrue(model.stream)
        self.assertEqual(model.max_out_len, 4096)
        self.assertEqual(model.retry, 2)
        self.assertEqual(model.host_ip, "localhost")
        self.assertEqual(model.host_port, 8080)
        self.assertEqual(model.base_url, "http://localhost:8080/")

    def test_init_custom_parameters(self):
        """测试使用自定义参数初始化"""
        kwargs = self.default_kwargs.copy()
        model = MindieStreamApi(**kwargs)
        self.assertEqual(model.path, "test-model")
        self.assertTrue(model.stream)
        self.assertEqual(model.max_out_len, 4096)
        self.assertEqual(model.retry, 2)
        self.assertEqual(model.host_ip, "localhost")
        self.assertEqual(model.host_port, 8080)
        self.assertEqual(model.base_url, "http://localhost:8080/")
        self.assertEqual(model.generation_kwargs, {"temperature": 0.7, "top_p": 0.9})

    def test_init_with_ssl(self):
        """测试启用SSL的初始化"""
        kwargs = self.default_kwargs.copy()
        kwargs["enable_ssl"] = True
        model = MindieStreamApi(**kwargs)
        self.assertEqual(model.base_url, "https://localhost:8080/")

    def test_init_with_url(self):
        """测试直接指定URL的初始化"""
        kwargs = self.default_kwargs.copy()
        kwargs["url"] = "https://custom-api.com/v1/"
        model = MindieStreamApi(**kwargs)
        self.assertEqual(model.base_url, "https://custom-api.com/v1/")

    def test_get_url(self):
        """测试_get_url方法"""
        model = MindieStreamApi(**self.default_kwargs)
        url = model._get_url()
        self.assertEqual(url, "http://localhost:8080/infer")

        # 测试使用自定义URL的情况
        kwargs = self.default_kwargs.copy()
        kwargs["url"] = "https://custom-api.com/v1/"
        model = MindieStreamApi(**kwargs)
        url = model._get_url()
        self.assertEqual(url, "https://custom-api.com/v1/infer")

    @mock.patch.object(mindie_stream_api, 'LMTemplateParser')
    def test_init_with_meta_template(self, mock_template_parser):
        """测试带有meta_template的初始化"""
        kwargs = self.default_kwargs.copy()
        meta_template = {"round": [{"role": "user", "begin": "user:"}, {"role": "assistant", "begin": "assistant:"}]}
        kwargs["meta_template"] = meta_template

        model = MindieStreamApi(**kwargs)
        mock_template_parser.assert_called_once_with(meta_template)
        self.assertEqual(model.meta_template, meta_template)

    def test_get_request_body_string_input(self):
        """测试使用字符串输入获取请求体"""
        async def run():
            model = MindieStreamApi(**self.default_kwargs)
            output = Output()
            input_data = "test prompt"
            max_out_len = 100

            request_body = await model.get_request_body(input_data, max_out_len, output)

            # 验证请求体格式
            self.assertEqual(request_body["inputs"], "test prompt")
            self.assertTrue(request_body["stream"])
            self.assertEqual(request_body["parameters"]["max_new_tokens"], 100)
            self.assertEqual(request_body["parameters"]["temperature"], 0.7)
            # 验证output.input被正确设置
            self.assertEqual(output.input, "test prompt")
        asyncio.run(run())

    def test_get_request_body_promptlist_input(self):
        """测试使用PromptList输入获取请求体"""
        async def run():
            model = MindieStreamApi(**self.default_kwargs)
            output = Output()
            input_data = PromptList(["test prompt 1", "test prompt 2"])
            max_out_len = 100

            request_body = await model.get_request_body(input_data, max_out_len, output)

            # 验证请求体格式
            self.assertEqual(request_body["inputs"], input_data)
            self.assertTrue(request_body["stream"])
            self.assertEqual(request_body["parameters"]["max_new_tokens"], 100)
            # 验证output.input被正确设置
            self.assertEqual(output.input, input_data)
        asyncio.run(run())


    async def test_get_request_body_with_empty_generation_kwargs(self):
        """测试generation_kwargs为空的情况"""
        kwargs = self.default_kwargs.copy()
        kwargs["generation_kwargs"] = None
        model = MindieStreamApi(**kwargs)
        output = Output()
        request_body = await model.get_request_body("test", 100, output)

        self.assertEqual(request_body["parameters"]["max_new_tokens"], 100)
        # 只有max_new_tokens，没有其他参数
        self.assertEqual(len(request_body["parameters"]), 1)

    def test_parse_stream_response_with_generated_text(self):
        """测试解析带有generated_text的流式响应"""
        async def run():
            model = MindieStreamApi(**self.default_kwargs)
            output = Output()
            api_response = {"generated_text": "Hello world!"}

            await model.parse_stream_response(api_response, output)

            self.assertEqual(output.content, "Hello world!")
        asyncio.run(run())

    def test_parse_stream_response_with_empty_generated_text(self):
        """测试解析generated_text为空的流式响应"""
        async def run():
            model = MindieStreamApi(**self.default_kwargs)
            output = Output()
            api_response = {"generated_text": ""}

            await model.parse_stream_response(api_response, output)

            self.assertEqual(output.content, "")
        asyncio.run(run())

    def test_parse_stream_response_without_generated_text(self):
        """测试解析没有generated_text字段的流式响应"""
        async def run():
            model = MindieStreamApi(**self.default_kwargs)
            output = Output()
            api_response = {"other_field": "some value"}

            await model.parse_stream_response(api_response, output)

            self.assertEqual(output.content, "")  # 应该使用默认空字符串
        asyncio.run(run())

    @mock.patch('aiohttp.ClientSession')
    async def test_generate_integration(self, mock_session_class):
        """集成测试generate方法"""

        # 设置mock
        mock_session = mock.AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session

        model = MindieStreamApi(**self.default_kwargs)
        output = Output()

        # Mock stream_infer方法
        with mock.patch.object(model, 'stream_infer') as mock_stream_infer:
            await model.generate("test prompt", 100, output)
            mock_stream_infer.assert_called_once()

    @mock.patch('aiohttp.ClientSession')
    async def test_generate_with_retry(self, mock_session_class):
        """测试generate方法的重试机制"""

        mock_session = mock.AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session

        model = MindieStreamApi(**self.default_kwargs)
        model.retry = 2
        output = Output()

        # 第一次调用失败，第二次成功
        mock_stream_infer = mock.AsyncMock(side_effect=[Exception("First fail"), None])
        with mock.patch.object(model, 'stream_infer', mock_stream_infer):
            await model.generate("test prompt", 100, output)
            self.assertEqual(mock_stream_infer.call_count, 2)

    @mock.patch('aiohttp.ClientSession.post')
    async def test_stream_infer_integration(self, mock_post):
        """集成测试stream_infer方法"""
        # 创建模拟的流响应
        async def mock_content():
            yield b"data: {\"generated_text\": \"Hello\"}\n\n"
            yield b"data: {\"generated_text\": \" world\"}\n\n"
            yield b"data: [DONE]\n\n"

        mock_response = mock.AsyncMock()
        mock_response.status = 200
        mock_response.content.__aiter__.return_value = mock_content()
        mock_post.return_value.__aenter__.return_value = mock_response

        model = MindieStreamApi(**self.default_kwargs)
        model.session = mock.AsyncMock()
        output = Output()

        # 由于我们不能直接调用stream_infer（它依赖于父类的实现），我们通过模拟parse_stream_response来测试
        with mock.patch.object(model, 'parse_stream_response') as mock_parse:
            await model.stream_infer({"inputs": "test", "stream": True, "parameters": {"max_new_tokens": 100}}, output)
            # 应该被调用两次，每次处理一个数据块
            self.assertEqual(mock_parse.call_count, 2)

    def test_is_api_property(self):
        """测试is_api属性"""
        model = MindieStreamApi(**self.default_kwargs)
        self.assertTrue(model.is_api)


class TestMindieStreamApiLora(unittest.TestCase):
    """针对 Multi-LoRA 兼容性扩展的 UT（MindIE 通过 parameters.adapter_id 路由 LoRA）。"""

    LORA_MAP = {"0": "LoraA", "1": "LoraB"}

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_root = self._tmpdir.name

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_lora_map(self, content=None):
        path = os.path.join(self.tmp_root, "lora_data_map.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content if content is not None else self.LORA_MAP, f)
        return path

    def _make_model(self, **overrides):
        kwargs = {
            "path": "test-model",
            "stream": True,
            "max_out_len": 4096,
            "retry": 2,
            "host_ip": "localhost",
            "host_port": 8080,
            "generation_kwargs": {"temperature": 0.7, "top_p": 0.9},
        }
        kwargs.update(overrides)
        return MindieStreamApi(**kwargs)

    # ---------- _load_lora_data_map ----------

    def test_load_lora_data_map_with_valid_file(self):
        path = self._write_lora_map()
        model = self._make_model(generation_kwargs={"lora_data_map_file": path})
        self.assertEqual(model.lora_data_map, self.LORA_MAP)

    def test_load_lora_data_map_no_config_returns_none(self):
        model = self._make_model()
        self.assertIsNone(model.lora_data_map)

    def test_load_lora_data_map_none_value_returns_none(self):
        model = self._make_model(generation_kwargs={"lora_data_map_file": None})
        self.assertIsNone(model.lora_data_map)

    def test_load_lora_data_map_file_not_exists(self):
        model = self._make_model(
            generation_kwargs={"lora_data_map_file": os.path.join(self.tmp_root, "absent.json")}
        )
        self.assertIsNone(model.lora_data_map)

    def test_load_lora_data_map_invalid_json(self):
        path = os.path.join(self.tmp_root, "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("definitely not json")
        model = self._make_model(generation_kwargs={"lora_data_map_file": path})
        self.assertIsNone(model.lora_data_map)

    def test_load_lora_data_map_empty_dict(self):
        path = self._write_lora_map(content={})
        model = self._make_model(generation_kwargs={"lora_data_map_file": path})
        self.assertEqual(model.lora_data_map, {})

    # ---------- _resolve_lora_model_name ----------

    def test_resolve_hit(self):
        path = self._write_lora_map()
        model = self._make_model(generation_kwargs={"lora_data_map_file": path})
        out = Output()
        out.data_id = 0
        self.assertEqual(model._resolve_lora_model_name(out), "LoraA")

    def test_resolve_miss_returns_none(self):
        path = self._write_lora_map()
        model = self._make_model(generation_kwargs={"lora_data_map_file": path})
        out = Output()
        out.data_id = 999
        self.assertIsNone(model._resolve_lora_model_name(out))

    def test_resolve_no_map_returns_none(self):
        model = self._make_model()
        out = Output()
        out.data_id = 0
        self.assertIsNone(model._resolve_lora_model_name(out))

    def test_resolve_output_without_data_id_returns_none(self):
        path = self._write_lora_map()
        model = self._make_model(generation_kwargs={"lora_data_map_file": path})
        out = Output()
        self.assertIsNone(model._resolve_lora_model_name(out))

    # ---------- get_request_body LoRA injection ----------

    def _run_get_request_body(self, model, output, input_data="test prompt", max_out_len=100):
        return asyncio.run(model.get_request_body(input_data, max_out_len, output))

    def test_request_body_lora_hit_writes_adapter_id(self):
        path = self._write_lora_map()
        model = self._make_model(generation_kwargs={"lora_data_map_file": path})
        out = Output()
        out.data_id = 1
        body = self._run_get_request_body(model, out)
        self.assertEqual(body["parameters"]["adapter_id"], "LoraB")
        self.assertTrue(body["stream"])
        self.assertEqual(body["parameters"]["max_new_tokens"], 100)

    def test_request_body_lora_miss_no_adapter_id(self):
        path = self._write_lora_map()
        model = self._make_model(generation_kwargs={"lora_data_map_file": path})
        out = Output()
        out.data_id = 99
        body = self._run_get_request_body(model, out)
        self.assertNotIn("adapter_id", body["parameters"])

    def test_request_body_no_lora_config_preserves_existing_kwargs(self):
        model = self._make_model()
        out = Output()
        out.data_id = 0
        body = self._run_get_request_body(model, out)
        self.assertEqual(body["parameters"]["temperature"], 0.7)
        self.assertEqual(body["parameters"]["top_p"], 0.9)
        self.assertNotIn("adapter_id", body["parameters"])

    def test_request_body_lora_hit_with_promptlist(self):
        path = self._write_lora_map()
        model = self._make_model(generation_kwargs={"lora_data_map_file": path})
        out = Output()
        out.data_id = 0
        prompt_list = PromptList(["p0", "p1"])
        body = self._run_get_request_body(model, out, input_data=prompt_list)
        self.assertEqual(body["parameters"]["adapter_id"], "LoraA")
        self.assertEqual(body["inputs"], prompt_list)


if __name__ == '__main__':
    unittest.main()