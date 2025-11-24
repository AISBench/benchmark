import json
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from ais_bench.benchmark.utils.postprocess.postprocessors.naive.extractor import (
    NaiveExtractor,
    format_input_naive,
    Meta_Instruction,
)
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchRuntimeError,
    ParameterValueError,
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


def test_format_input_naive_constructs_expected_templates():
    data = [
        {
            "origin_prompt": [{"prompt": "ignored"}, {"prompt": "Q?"}],
            "prediction": "model output",
            "reference": None,
            "gold": "answer",
        }
    ]
    out = format_input_naive(data)
    assert isinstance(out, list) and len(out) == 1
    item = out[0]
    assert item["question"] == "Q?"
    assert item["llm_output"] == "model output"
    assert item["correct_answer"] == "answer"


def test_prepare_input_includes_required_sections():
    ex = NaiveExtractor(model_name="m", url="http://u")
    item = {"question": "Q?", "llm_output": "The content."}
    s = ex.prepare_input(item)
    assert s.startswith(Meta_Instruction)
    assert 'Question: """Q?"""' in s
    assert 'Output sentences: """The content."""' in s
    assert s.endswith('Key extracted answer: ')


def test_gen_output_delegates_to_openai_infer():
    ex = NaiveExtractor(model_name="m", url="http://u")
    with patch.object(ex, "openai_infer", return_value="ok") as p:
        assert ex.gen_output("query") == "ok"
        p.assert_called_once_with("query")


def test_openai_infer_raises_when_url_missing():
    ex = NaiveExtractor(model_name="m", url=None)
    with pytest.raises(ParameterValueError) as ei:
        ex.openai_infer("q")
    assert ei.value.error_code_str == UTILS_CODES.MISSING_API_URL.full_code


def test_openai_init_failure_raises_custom_error(monkeypatch):
    def boom(**kwargs):
        raise Exception("boom")

    # Patch the symbol imported in the module under test
    monkeypatch.setattr(
        "ais_bench.benchmark.utils.postprocess.postprocessors.naive.extractor.OpenAI",
        lambda **kwargs: boom(**kwargs),
    )

    ex = NaiveExtractor(model_name="m", url="http://u")
    with pytest.raises(AISBenchRuntimeError) as ei:
        ex.openai_infer("q")
    assert ei.value.error_code_str == UTILS_CODES.DEPENDENCY_MODULE_IMPORT_ERROR.full_code


def test_openai_infer_success_path_uses_provided_model(monkeypatch):
    # Ensure it doesn't call models.list by providing a non-empty model_name
    client_holder = {}

    def fake_openai(**kwargs):
        client = DummyOpenAIClient(**kwargs)
        client_holder["client"] = client
        return client

    monkeypatch.setattr(
        "ais_bench.benchmark.utils.postprocess.postprocessors.naive.extractor.OpenAI",
        lambda **kwargs: fake_openai(**kwargs),
    )

    ex = NaiveExtractor(model_name="some-model", url="http://chosen")
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
        "ais_bench.benchmark.utils.postprocess.postprocessors.naive.extractor.OpenAI",
        lambda **kwargs: fake_openai(**kwargs),
    )

    ex = NaiveExtractor(model_name="m", url=["http://u1", "http://u2"]) 
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
        "ais_bench.benchmark.utils.postprocess.postprocessors.naive.extractor.OpenAI",
        lambda **kwargs: fake_openai(**kwargs),
    )

    ex = NaiveExtractor(model_name="m", url="http://u")
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
        "ais_bench.benchmark.utils.postprocess.postprocessors.naive.extractor.OpenAI",
        lambda **kwargs: fake_openai(**kwargs),
    )

    # Avoid sleeping in tests
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    # Make perf_counter deterministic
    monkeypatch.setattr("time.perf_counter", lambda: 0.0)

    ex = NaiveExtractor(model_name="m", url="http://u")
    with pytest.raises(AISBenchRuntimeError) as ei:
        ex.openai_infer("hello", retry=2)
    assert ei.value.error_code_str == UTILS_CODES.API_RETRY_EXCEEDED.full_code
