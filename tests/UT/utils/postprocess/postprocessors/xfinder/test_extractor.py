import json
from types import SimpleNamespace
from unittest.mock import patch, Mock
import pytest

from ais_bench.benchmark.utils.postprocess.postprocessors.xfinder.extractor import (
    Extractor,
)
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchRuntimeError,
)
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES


class DummyChatResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump_json(self):
        return json.dumps(self._payload)


class DummyCompletions:
    def __init__(self, payload: dict = None, exc: Exception = None):
        self._payload = payload or {
            "choices": [{"message": {"content": "extracted-answer"}}]
        }
        self._exc = exc

    def create(self, **kwargs):
        if self._exc:
            raise self._exc
        return DummyChatResponse(self._payload)


class DummyChat:
    def __init__(self, payload: dict = None, exc: Exception = None):
        self.completions = DummyCompletions(payload=payload, exc=exc)


class DummyModels:
    def list(self):
        # Minimal structure when model_name == ''
        return SimpleNamespace(data=[SimpleNamespace(id="dummy-model-id")])


class DummyOpenAIClient:
    def __init__(self, api_key: str, base_url: str):
        # Store inputs for assertions
        self._api_key = api_key
        self._base_url = base_url
        self.chat = DummyChat()
        self.models = DummyModels()


def test_prepare_input_constructs_expected_string():
    item = {
        "question": "What is 2+2?",
        "llm_output": "The answer is 4.",
        "standard_answer_range": "numeric"
    }
    s = Extractor.prepare_input(item)
    assert "Question: \"\"\"What is 2+2?\"\"\"" in s
    assert "Output sentences: \"\"\"The answer is 4.\"\"\"" in s
    assert "Answer range: numeric" in s
    assert s.endswith("Key extracted answer: ")


def test_init_api_mode_sets_attributes():
    ex = Extractor(model_name="xFinder-qwen1505", url="http://api.example.com")
    assert ex.model_name == "xFinder-qwen1505"
    assert ex.url == "http://api.example.com"
    assert ex.mode == "API"
    assert ex.client is None
    assert ex.retry == 0


def test_init_local_mode_initializes_vllm(monkeypatch):
    llm_holder = {}

    class DummyLLM:
        def __init__(self, model, gpu_memory_utilization):
            llm_holder["model"] = model
            llm_holder["gpu"] = gpu_memory_utilization

    class DummySamplingParams:
        def __init__(self, temperature, max_tokens, stop):
            llm_holder["temp"] = temperature
            llm_holder["max_tokens"] = max_tokens
            llm_holder["stop"] = stop

    mock_vllm = Mock()
    mock_vllm.LLM = DummyLLM
    mock_vllm.SamplingParams = DummySamplingParams
    monkeypatch.setattr("sys.modules", {"vllm": mock_vllm})

    ex = Extractor(model_name="xFinder-llama38it", model_path="/path/to/model")
    assert ex.mode == "Local"
    assert llm_holder["model"] == "/path/to/model"
    assert llm_holder["gpu"] == 0.5
    assert llm_holder["temp"] == 0
    assert llm_holder["max_tokens"] == 3000


def test_gen_output_routes_to_openai_infer_for_api():
    ex = Extractor(model_name="xFinder-qwen1505", url="http://u")
    with patch.object(ex, "openai_infer", return_value="api-result") as p:
        assert ex.gen_output("query") == "api-result"
        p.assert_called_once_with("query")


def test_gen_output_routes_to_offline_infer_for_local(monkeypatch):
    # Mock vLLM init to avoid import
    mock_vllm = Mock()
    mock_vllm.LLM = lambda **kwargs: None
    mock_vllm.SamplingParams = lambda **kwargs: None
    monkeypatch.setattr("sys.modules", {"vllm": mock_vllm})

    ex = Extractor(model_name="xFinder-llama38it", model_path="/path")
    with patch.object(ex, "offline_infer", return_value="local-result") as p:
        assert ex.gen_output("query") == "local-result"
        p.assert_called_once_with("query")


def test_openai_infer_client_init_failure_raises(monkeypatch):
    def boom(**kwargs):
        raise ValueError("invalid key")

    monkeypatch.setattr(
        "ais_bench.benchmark.utils.postprocess.postprocessors.xfinder.extractor.OpenAI",
        boom,
    )

    ex = Extractor(model_name="xFinder-qwen1505", url="http://u")
    with pytest.raises(AISBenchRuntimeError) as ei:
        ex.openai_infer("q")
    assert ei.value.error_code_str == UTILS_CODES.UNKNOWN_ERROR.full_code


def test_openai_infer_success_path(monkeypatch):
    client_holder = {}

    def fake_openai(**kwargs):
        client = DummyOpenAIClient(**kwargs)
        client_holder["client"] = client
        return client

    monkeypatch.setattr(
        "ais_bench.benchmark.utils.postprocess.postprocessors.xfinder.extractor.OpenAI",
        fake_openai,
    )

    ex = Extractor(model_name="xFinder-qwen1505", url="http://chosen")
    out = ex.openai_infer("hello")
    assert out == "extracted-answer"
    assert client_holder["client"]._base_url == "http://chosen"


def test_openai_infer_selects_from_url_list(monkeypatch):
    chosen = {"url": None}

    def fake_choice(lst):
        chosen["url"] = lst[1]
        return lst[1]

    def fake_openai(**kwargs):
        return DummyOpenAIClient(**kwargs)

    monkeypatch.setattr("random.choice", fake_choice)
    monkeypatch.setattr(
        "ais_bench.benchmark.utils.postprocess.postprocessors.xfinder.extractor.OpenAI",
        lambda **kwargs: fake_openai(**kwargs),
    )

    ex = Extractor(model_name="xFinder-qwen1505", url=["http://u1", "http://u2"]) 
    _ = ex.openai_infer("hello")
    assert chosen["url"] == "http://u2"


def test_openai_infer_parse_failure_raises(monkeypatch):
    # Provide a payload missing required keys to trigger KeyError
    payload = {"not_choices": []}

    def fake_openai(**kwargs):
        client = DummyOpenAIClient(**kwargs)
        client.chat = DummyChat(payload=payload)
        return client

    monkeypatch.setattr(
        "ais_bench.benchmark.utils.postprocess.postprocessors.xfinder.extractor.OpenAI",
        lambda **kwargs: fake_openai(**kwargs),
    )

    ex = Extractor(model_name="xFinder-qwen1505", url="http://u")
    with pytest.raises(AISBenchRuntimeError) as ei:
        ex.openai_infer("hello")
    assert ei.value.error_code_str == UTILS_CODES.API_RESPONSE_PARSE_FAILED.full_code


def test_openai_infer_retry_exhaustion_raises(monkeypatch):
    # Force a retryable exception from create
    def fake_openai(**kwargs):
        client = DummyOpenAIClient(**kwargs)
        client.chat = DummyChat(exc=ConnectionError("down"))
        return client

    monkeypatch.setattr(
        "ais_bench.benchmark.utils.postprocess.postprocessors.xfinder.extractor.OpenAI",
        lambda **kwargs: fake_openai(**kwargs),
    )

    # Avoid sleeping in tests
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    # Make perf_counter deterministic
    monkeypatch.setattr("time.perf_counter", lambda: 0.0)

    ex = Extractor(model_name="xFinder-qwen1505", url="http://u")
    with pytest.raises(AISBenchRuntimeError) as ei:
        ex.openai_infer("hello", retry=2)
    assert ei.value.error_code_str == UTILS_CODES.API_RETRY_EXCEEDED.full_code


def test_offline_infer_calls_vllm_generate(monkeypatch):
    class DummyOutput:
        def __init__(self, text):
            self.text = text

    class DummyResult:
        def __init__(self, outputs):
            self.outputs = outputs

    llm_mock = SimpleNamespace(
        generate=lambda prompt, params: [DummyResult([DummyOutput("generated text")])]
    )

    mock_vllm = Mock()
    mock_vllm.LLM = lambda **kwargs: llm_mock
    mock_vllm.SamplingParams = lambda **kwargs: None
    monkeypatch.setattr("sys.modules", {"vllm": mock_vllm})

    ex = Extractor(model_name="xFinder-llama38it", model_path="/path")
    result = ex.offline_infer("query")
    assert result == "generated text"