import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch


PLUGIN_ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = PLUGIN_ROOT / "ais_bench_prefix_cache"


class _Registry:
    def register_module(self):
        return lambda cls: cls


class _FakeAISLogger:
    def __init__(self):
        self.debug_messages = []
        self.info_messages = []

    def debug(self, *args, **kwargs):
        self.debug_messages.append((args, kwargs))

    def info(self, *args, **kwargs):
        self.info_messages.append((args, kwargs))


class _FakeDataset(list):
    @classmethod
    def from_list(cls, rows):
        return cls(rows)

    @property
    def column_names(self):
        return list(self[0]) if self else []


class _FakeVLLMCustomAPI:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.headers = {"Authorization": "Bearer test"}
        self.model = kwargs.get("model", "model")
        self.stream = kwargs.get("stream", False)
        self.max_out_len = kwargs.get("max_out_len", 16)
        self.retry = kwargs.get("retry", 2)
        self.batch_size = kwargs.get("batch_size", 1)
        self.parsed_text = []
        self.parsed_stream = []
        self.anomaly_payloads = []

    async def get_request_body(self, input_data, max_out_len, output, **kwargs):
        return {
            "prompt": input_data,
            "max_tokens": max_out_len,
            "model": self.model,
            "stream": self.stream,
        }

    async def parse_text_response(self, data, output):
        self.parsed_text.append(data)

    async def parse_stream_response(self, data, output):
        self.parsed_stream.append(data)

    def _record_response_anomaly_payload(self, data, output):
        self.anomaly_payloads.append(("text", data))

    def _accumulate_response_anomaly_payload(self, data, output):
        self.anomaly_payloads.append(("stream", data))

    async def iter_lines(self, content):
        for chunk in content:
            yield chunk


class _FakeVLLMCustomAPIChat:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs

    async def get_request_body(self, input_data, max_out_len, output, **kwargs):
        return {
            "messages": input_data,
            "max_tokens": max_out_len,
        }


class _FakeMMPromptTemplate:
    def __init__(self, **kwargs):
        self.template_kwargs = kwargs

    def generate_item(self, entry, **kwargs):
        return self.template_kwargs["generated_prompt"]


class _FakeOutput:
    def __init__(self):
        self.time_points = 0
        self.success = None
        self.error_info = None

    async def record_time_point(self):
        self.time_points += 1


class _FakeResponse:
    def __init__(self, *, status=200, reason="OK", text="{}", content=()):
        self.status = status
        self.reason = reason
        self._text = text
        self.content = list(content)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def text(self):
        return self._text


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeAISBenchValueError(ValueError):
    pass


def _package(name):
    module = ModuleType(name)
    module.__path__ = []
    return module


def _load_source(relative_path, module_name, stubs):
    module_path = PACKAGE_ROOT / relative_path
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(module_name, module_path)
    )
    with patch.dict(sys.modules, stubs | {module_name: module}):
        assert module is not None
        assert module.__spec__ is not None
        assert module.__spec__.loader is not None
        module.__spec__.loader.exec_module(module)
    return module


def _common_aisbench_stubs():
    packages = {
        name: _package(name)
        for name in (
            "ais_bench",
            "ais_bench.benchmark",
            "ais_bench.benchmark.utils",
            "ais_bench.benchmark.utils.logging",
        )
    }
    registry = ModuleType("ais_bench.benchmark.registry")
    registry.LOAD_DATASET = _Registry()
    registry.MODELS = _Registry()
    registry.ICL_PROMPT_TEMPLATES = _Registry()
    logger = ModuleType("ais_bench.benchmark.utils.logging.logger")
    logger.AISLogger = _FakeAISLogger
    return packages | {registry.__name__: registry, logger.__name__: logger}


def _load_prefix_dataset_module():
    stubs = _common_aisbench_stubs()
    stubs.update(
        {
            "ais_bench.benchmark.datasets": _package(
                "ais_bench.benchmark.datasets"
            ),
            "ais_bench_prefix_cache": _package("ais_bench_prefix_cache"),
            "ais_bench_prefix_cache.datasets": _package(
                "ais_bench_prefix_cache.datasets"
            ),
        }
    )
    datasets_module = ModuleType("datasets")
    datasets_module.Dataset = _FakeDataset
    base_module = ModuleType("ais_bench.benchmark.datasets.base")
    base_module.BaseDataset = object
    artifacts_module = ModuleType("ais_bench_prefix_cache.artifacts")
    artifacts_module.read_jsonl = MagicMock()
    artifacts_module.validate_artifacts = MagicMock()
    errors_module = ModuleType("ais_bench_prefix_cache.errors")
    errors_module.ArtifactValidationError = RuntimeError
    stubs.update(
        {
            datasets_module.__name__: datasets_module,
            base_module.__name__: base_module,
            artifacts_module.__name__: artifacts_module,
            errors_module.__name__: errors_module,
        }
    )
    return _load_source(
        "datasets/prefix_cache_dataset.py",
        "ais_bench_prefix_cache.datasets._prefix_cache_dataset_test",
        stubs,
    )


def _load_multimodal_dataset_module():
    stubs = _common_aisbench_stubs()
    stubs.update(
        {
            "ais_bench.benchmark.datasets": _package(
                "ais_bench.benchmark.datasets"
            ),
            "ais_bench_prefix_cache": _package("ais_bench_prefix_cache"),
            "ais_bench_prefix_cache.datasets": _package(
                "ais_bench_prefix_cache.datasets"
            ),
        }
    )
    datasets_module = ModuleType("datasets")
    datasets_module.Dataset = _FakeDataset
    base_module = ModuleType("ais_bench.benchmark.datasets.base")
    base_module.BaseDataset = object
    prompt_module = ModuleType("ais_bench.benchmark.utils.prompt")
    prompt_module.AIS_CONTENT_TAG = "<content>"
    prompt_module.AIS_IMAGE_START = "<image>"
    prompt_module.AIS_TEXT_START = "<text>"
    errors_module = ModuleType("ais_bench_prefix_cache.errors")
    errors_module.ArtifactValidationError = RuntimeError
    stubs.update(
        {
            datasets_module.__name__: datasets_module,
            base_module.__name__: base_module,
            prompt_module.__name__: prompt_module,
            errors_module.__name__: errors_module,
        }
    )
    return _load_source(
        "datasets/multimodal_prefix_cache_dataset.py",
        "ais_bench_prefix_cache.datasets._multimodal_dataset_test",
        stubs,
    )


def _load_prompt_template_module():
    stubs = _common_aisbench_stubs()
    stubs.update(
        {
            "ais_bench.benchmark.openicl": _package(
                "ais_bench.benchmark.openicl"
            ),
            "ais_bench_prefix_cache": _package("ais_bench_prefix_cache"),
            "ais_bench_prefix_cache.openicl": _package(
                "ais_bench_prefix_cache.openicl"
            ),
        }
    )
    template_module = ModuleType(
        "ais_bench.benchmark.openicl.icl_prompt_template"
    )
    template_module.MMPromptTemplate = _FakeMMPromptTemplate
    multimodal_module = ModuleType("ais_bench_prefix_cache.multimodal")
    multimodal_module.expand_prompt_image_refs = MagicMock(
        return_value="expanded-prompt"
    )
    stubs.update(
        {
            template_module.__name__: template_module,
            multimodal_module.__name__: multimodal_module,
        }
    )
    return _load_source(
        "openicl/multimodal_prompt_template.py",
        "ais_bench_prefix_cache.openicl._multimodal_prompt_template_test",
        stubs,
    )


def _model_stubs():
    stubs = _common_aisbench_stubs()
    models_module = ModuleType("ais_bench.benchmark.models")
    models_module.VLLMCustomAPI = _FakeVLLMCustomAPI
    api_models = _package("ais_bench.benchmark.models.api_models")
    chat_module = ModuleType(
        "ais_bench.benchmark.models.api_models.vllm_custom_api_chat"
    )
    chat_module.VLLMCustomAPIChat = _FakeVLLMCustomAPIChat
    codes_module = ModuleType(
        "ais_bench.benchmark.utils.logging.error_codes"
    )
    codes_module.MODEL_CODES = SimpleNamespace(
        PARSE_TEXT_RSP_INVALID_FORMAT="invalid-format"
    )
    exceptions_module = ModuleType(
        "ais_bench.benchmark.utils.logging.exceptions"
    )
    exceptions_module.AISBenchValueError = _FakeAISBenchValueError
    stubs.update(
        {
            models_module.__name__: models_module,
            api_models.__name__: api_models,
            chat_module.__name__: chat_module,
            codes_module.__name__: codes_module,
            exceptions_module.__name__: exceptions_module,
        }
    )
    return stubs


def _load_completion_model_module():
    return _load_source(
        "models/vllm_prefix_cache_api.py",
        "_vllm_prefix_cache_api_test",
        _model_stubs(),
    )


def _load_chat_model_module():
    return _load_source(
        "models/vllm_prefix_cache_chat_api.py",
        "_vllm_prefix_cache_chat_api_test",
        _model_stubs(),
    )


class DatasetAdapterTest(unittest.TestCase):
    def test_prefix_dataset_merges_validated_request_and_route_metadata(self):
        module = _load_prefix_dataset_module()
        requests = [
            {"question": "first", "answer": "a"},
            {"question": "second", "answer": "b"},
        ]
        audits = [
            {
                "sequence_index": 0,
                "request_id": "req-0",
                "group_id": "group-a",
                "dp_rank": 1,
                "lane_sequence": 3,
                "max_tokens": 12,
            },
            {
                "sequence_index": 1,
                "request_id": "req-1",
                "group_id": "group-b",
                "max_tokens": 24,
            },
        ]
        module.validate_artifacts = MagicMock(return_value={"valid": True})
        module.read_jsonl = MagicMock(side_effect=[requests, audits])
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = Path(folder) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "prefix_cache": {"mode": "cold"},
                    }
                ),
                encoding="utf-8",
            )
            dataset = module.PrefixCacheDataset.load(
                "requests.jsonl",
                "full.jsonl",
                str(manifest_path),
                ignored=True,
            )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(
            dataset[0],
            {
                "question": "first",
                "answer": "a",
                "max_out_len": 12,
                "dp_rank": 1,
                "group_id": "group-a",
                "lane_sequence": 3,
                "cache_mode": "cold",
            },
        )
        self.assertIsNone(dataset[1]["dp_rank"])
        self.assertIsNone(dataset[1]["lane_sequence"])
        module.validate_artifacts.assert_called_once_with(manifest_path.resolve())

    def test_prefix_dataset_rejects_count_and_route_order_mismatches(self):
        module = _load_prefix_dataset_module()
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = Path(folder) / "manifest.json"
            manifest_path.write_text(
                json.dumps({"prefix_cache": {"mode": "warm"}}),
                encoding="utf-8",
            )
            module.validate_artifacts = MagicMock(return_value={})
            module.read_jsonl = MagicMock(side_effect=[[{}], []])
            with self.assertRaisesRegex(RuntimeError, "row count mismatch"):
                module.PrefixCacheDataset.load("requests", "full", manifest_path)

            module.read_jsonl = MagicMock(
                side_effect=[
                    [{"question": "q", "answer": "a"}],
                    [
                        {
                            "sequence_index": 9,
                            "group_id": "g",
                            "max_tokens": 1,
                        }
                    ],
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "order mismatch at row 0"):
                module.PrefixCacheDataset.load("requests", "full", manifest_path)

    def test_multimodal_dataset_builds_image_first_content_and_skips_blanks(self):
        module = _load_multimodal_dataset_module()
        row = {
            "question": "what is shown?",
            "answer": "chart",
            "image_refs": ["image-a", "image-b"],
        }
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "requests.jsonl"
            source.write_text(
                "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            dataset = module.MultimodalPrefixCacheDataset.load(
                str(source), ignored=True
            )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(
            dataset[0]["content"],
            "<image>image-a<content><image>image-b<content>"
            "<text>what is shown?",
        )
        self.assertEqual(dataset[0]["answer"], "chart")

    def test_multimodal_dataset_rejects_missing_refs_and_invalid_json(self):
        module = _load_multimodal_dataset_module()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "requests.jsonl"
            source.write_text(
                json.dumps({"question": "q", "image_refs": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "row 1"):
                module.MultimodalPrefixCacheDataset.load(source)

            source.write_text("{invalid", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid JSONL"):
                module.MultimodalPrefixCacheDataset.load(source)


class PromptTemplateAdapterTest(unittest.TestCase):
    def test_generate_item_expands_refs_with_a_private_media_copy(self):
        module = _load_prompt_template_module()
        media = {"image-a": "data:image/png;base64,YQ=="}
        raw_prompt = [{"role": "HUMAN", "prompt": "image-a"}]
        template = module.Base64RefMMPromptTemplate(
            media=media,
            generated_prompt=raw_prompt,
        )
        media["image-a"] = "changed"

        result = template.generate_item(
            {"question": "q"},
            output_field="answer",
            output_field_replace_token="<out>",
            ice_field_replace_token="<ice>",
        )

        self.assertEqual(result, "expanded-prompt")
        self.assertEqual(
            template.media,
            {"image-a": "data:image/png;base64,YQ=="},
        )
        module.expand_prompt_image_refs.assert_called_once_with(
            raw_prompt,
            template.media,
        )


class CompletionModelAdapterTest(unittest.TestCase):
    def test_url_normalization_safe_logging_and_dp_routing(self):
        module = _load_completion_model_module()
        endpoint = (
            "http://user:password@localhost:8000/v1/completions"
            "?token=secret#fragment"
        )
        model = module.VLLMPrefixCacheAPI(
            endpoint,
            model="demo",
            stream=True,
            max_out_len=32,
        )
        self.assertEqual(model.init_kwargs["url"], "http://user:password@localhost:8000/")
        self.assertEqual(model.url, endpoint)
        self.assertEqual(
            model._safe_url(endpoint),
            "http://localhost:8000/v1/completions",
        )
        self.assertEqual(model._safe_url("http://host:bad/path"), "<invalid-url>")

        body = asyncio.run(model.get_request_body("prompt", 7, object(), dp_rank=2))
        payload, headers = model._payload_and_headers(body)
        self.assertNotIn(module._DP_KEY, payload)
        self.assertEqual(headers["X-data-parallel-rank"], "2")
        self.assertEqual(payload["prompt"], "prompt")

        payload, headers = model._payload_and_headers({"prompt": "q"})
        self.assertEqual(payload, {"prompt": "q"})
        self.assertNotIn("X-data-parallel-rank", headers)

        plain = module.VLLMPrefixCacheAPI("http://localhost:8000/custom")
        self.assertEqual(plain.init_kwargs["url"], "http://localhost:8000/custom")

    def test_text_infer_handles_success_http_error_and_invalid_json(self):
        module = _load_completion_model_module()
        model = module.VLLMPrefixCacheAPI("http://localhost/v1/completions")
        body = {
            "prompt": "q",
            "max_tokens": 3,
            module._DP_KEY: 1,
        }

        model.session = _FakeSession(
            _FakeResponse(text=json.dumps({"choices": [{"text": "ok"}]}))
        )
        output = _FakeOutput()
        asyncio.run(model.text_infer(body, output))
        self.assertTrue(output.success)
        self.assertEqual(output.time_points, 2)
        self.assertEqual(model.parsed_text[0]["choices"][0]["text"], "ok")
        self.assertEqual(model.anomaly_payloads[0][0], "text")
        self.assertEqual(
            model.session.calls[0]["headers"]["X-data-parallel-rank"],
            "1",
        )

        model.session = _FakeSession(_FakeResponse(status=429, reason="busy"))
        output = _FakeOutput()
        asyncio.run(model.text_infer(body, output))
        self.assertFalse(output.success)
        self.assertEqual(output.error_info, "busy")

        model.session = _FakeSession(_FakeResponse(text="not-json"))
        output = _FakeOutput()
        with self.assertRaises(_FakeAISBenchValueError):
            asyncio.run(model.text_infer(body, output))
        self.assertFalse(output.success)
        self.assertIn("Unexpected response format", output.error_info)

    def test_stream_infer_skips_sse_control_lines_and_handles_http_error(self):
        module = _load_completion_model_module()
        model = module.VLLMPrefixCacheAPI("http://localhost/v1/completions")
        body = {"prompt": "q", "max_tokens": 2, module._DP_KEY: 0}
        chunk = {"choices": [{"text": "token"}]}
        model.session = _FakeSession(
            _FakeResponse(
                content=[
                    b"",
                    b": keep-alive",
                    f"data: {json.dumps(chunk)}".encode(),
                    b"data: [DONE]",
                ]
            )
        )
        output = _FakeOutput()
        asyncio.run(model.stream_infer(body, output))
        self.assertTrue(output.success)
        self.assertEqual(output.time_points, 2)
        self.assertEqual(model.parsed_stream, [chunk])
        self.assertEqual(model.anomaly_payloads[-1], ("stream", chunk))

        model.session = _FakeSession(_FakeResponse(status=500, reason="failed"))
        output = _FakeOutput()
        asyncio.run(model.stream_infer(body, output))
        self.assertFalse(output.success)
        self.assertEqual(output.error_info, "failed")

    def test_stream_infer_propagates_invalid_json_chunk(self):
        module = _load_completion_model_module()
        model = module.VLLMPrefixCacheAPI("http://localhost/v1/completions")
        model.session = _FakeSession(_FakeResponse(content=[b"data: invalid-json"]))
        with self.assertRaises(json.JSONDecodeError):
            asyncio.run(
                model.stream_infer(
                    {"prompt": "q", "max_tokens": 1},
                    _FakeOutput(),
                )
            )


class ChatModelAdapterTest(unittest.TestCase):
    def test_chat_model_keeps_exact_endpoint_and_logs_final_messages(self):
        module = _load_chat_model_module()
        endpoint = "http://localhost:8000/v1/chat/completions"
        model = module.VLLMPrefixCacheChatAPI(endpoint, model="vlm")
        self.assertEqual(model.init_kwargs["url"], "http://localhost:8000/")
        self.assertEqual(model.url, endpoint)

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "你好"}],
            }
        ]
        body = asyncio.run(model.get_request_body(messages, 8, object()))
        self.assertEqual(body["messages"], messages)
        self.assertEqual(len(module.logger.info_messages), 1)
        log_args = module.logger.info_messages[0][0]
        self.assertEqual(log_args[0], "[aisbench-model] multimodal request prompt=%s")
        self.assertIn("你好", log_args[1])
        self.assertNotIn("\\u4f60", log_args[1])

        custom = module.VLLMPrefixCacheChatAPI("http://localhost:8000/custom")
        self.assertEqual(custom.init_kwargs["url"], "http://localhost:8000/custom")


if __name__ == "__main__":
    unittest.main()
