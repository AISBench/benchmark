"""Unit tests for AgentX's shared CLI/config preflight contract."""

import pytest

from ais_bench.benchmark.tasks.custom_tasks.agentx_spec import (
    extra_arg_value,
    normalize_base_url,
    resolve_cooldown,
    resolve_endpoint,
    resolve_variant,
    validate_agentx_args,
)


@pytest.fixture
def model():
    return {
        "model": "example-model",
        "url": "https://example.test:8443/gateway",
        "path": "owner/tokenizer",
    }


@pytest.mark.parametrize(
    ("value", "expected"), [("full", "full"), (" 256k\t", "256k")]
)
def test_resolve_variant_trims_explicit_variant(value, expected):
    assert resolve_variant(value) == expected


@pytest.mark.parametrize("value", [None, 256, "", " \t"])
def test_resolve_variant_requires_explicit_selection(value):
    with pytest.raises(ValueError, match="explicit dataset variant"):
        resolve_variant(value)


@pytest.mark.parametrize("value", ["FULL", "256K", "example-model"])
def test_resolve_variant_does_not_guess_or_case_fold(value):
    with pytest.raises(ValueError, match="Unknown AgentX dataset variant"):
        resolve_variant(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "/v1/chat/completions"), ("  /gateway/chat/  ", "/gateway/chat/")],
)
def test_resolve_endpoint_preserves_absolute_path(value, expected):
    assert resolve_endpoint(value) == expected


@pytest.mark.parametrize("value", [False, "", "  "])
def test_resolve_endpoint_rejects_empty_or_non_string(value):
    with pytest.raises(ValueError, match="non-empty path"):
        resolve_endpoint(value)


@pytest.mark.parametrize(
    "value", ["chat", "/chat?x=1", "/chat#section", "/my chat", "/my\tchat"]
)
def test_resolve_endpoint_rejects_non_path_components(value):
    with pytest.raises(ValueError, match="absolute path"):
        resolve_endpoint(value)


@pytest.mark.parametrize(
    ("url", "normalized"),
    [
        ("http://example.test/", "http://example.test"),
        ("https://example.test:8443/gateway///", "https://example.test:8443/gateway"),
        ("http://[::1]:8080/base/", "http://[::1]:8080/base"),
    ],
)
def test_normalize_url_retains_gateway_prefix_and_port(url, normalized):
    assert normalize_base_url(url) == normalized


@pytest.mark.parametrize("url", ["example.test", "http:///gateway", "ftp://example.test"])
def test_normalize_url_requires_http_and_host(url):
    with pytest.raises(ValueError, match=r"expected http\(s\)://host"):
        normalize_base_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://user@example.test/base",
        "https://user:placeholder@example.test/base",
        "https://example.test/base?q=1",
        "https://example.test/base#section",
    ],
)
def test_normalize_url_rejects_non_base_components(url):
    with pytest.raises(ValueError, match="userinfo, query or fragment"):
        normalize_base_url(url)


def test_normalize_url_rejects_custom_endpoint_already_in_url():
    with pytest.raises(ValueError, match="configure the path with --endpoint"):
        normalize_base_url("https://example.test/gateway/chat/", "/chat/")


@pytest.mark.parametrize(("value", "seconds"), [(None, 5.0), (0, 0.0), (1.25, 1.25)])
def test_resolve_cooldown_returns_float_seconds(value, seconds):
    result = resolve_cooldown(value)
    assert result == seconds
    assert isinstance(result, float)


@pytest.mark.parametrize("value", [True, "5", -0.01, float("inf"), float("nan")])
def test_resolve_cooldown_rejects_invalid_seconds(value):
    with pytest.raises(ValueError, match="non-negative finite number"):
        resolve_cooldown(value)


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        (None, None),
        ("  {}  ", None),
        ('{ --model "two words" --unrelated hey }', "two words"),
        ('--model="two words"', "two words"),
        ("--model=first --model second", "first"),
        ("--model-path /tmp/example", None),
        ("--model", None),
        ("--model=", ""),
        ('--model "unfinished', None),
    ],
)
def test_extra_arg_value_extracts_exact_option(extra_args, expected):
    assert extra_arg_value(extra_args, "--model") == expected


def test_validate_defaults_to_single_concurrency(model):
    assert validate_agentx_args(model, {"variant": "full"}) == [1]


@pytest.mark.parametrize(
    ("concurrency", "levels"), [(3, [3]), ([8, 1, 4], [8, 1, 4]), ((8, 1), [8, 1])]
)
def test_validate_preserves_requested_sweep_order(model, concurrency, levels):
    assert validate_agentx_args(model, {"variant": "full", "concurrency": concurrency}) == levels


@pytest.mark.parametrize(
    ("concurrency", "message"),
    [
        (True, "must be an integer"),
        ("2", "must be an integer or a list"),
        ([], "requires at least one level"),
        ([2, True], "must be a positive integer"),
        ([2, 0], "must be a positive integer"),
        ([2, 1.0], "must be a positive integer"),
        ([2, 3, 2], "level 2 is duplicated"),
    ],
)
def test_validate_rejects_invalid_sweep_before_model_lookup(concurrency, message):
    with pytest.raises(ValueError, match=message):
        validate_agentx_args({}, {"concurrency": concurrency})


def test_validate_allows_model_and_url_from_task_args(model):
    args = {"model": model.pop("model"), "url": model.pop("url"), "variant": "256k"}
    assert validate_agentx_args(model, args) == [1]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("model", " ", "non-empty model name"),
        ("url", 123, "service base URL"),
        ("path", None, "model/tokenizer"),
        ("path", " ", "model/tokenizer"),
        ("endpoint", "/chat?bad=1", "absolute path"),
    ],
)
def test_validate_rejects_invalid_model_fields(model, key, value, message):
    model[key] = value
    with pytest.raises(ValueError, match=message):
        validate_agentx_args(model, {"variant": "full"})


def test_validate_requires_explicit_local_tokenizer_directory(model, tmp_path):
    tokenizer_path = tmp_path / "local tokenizer"
    model["path"] = str(tokenizer_path)
    with pytest.raises(FileNotFoundError, match="local model/tokenizer directory"):
        validate_agentx_args(model, {"variant": "full"})
    tokenizer_path.mkdir()
    assert validate_agentx_args(model, {"variant": "full"}) == [1]


def test_validate_raw_mode_skips_model_metadata_but_checks_run_mode():
    assert validate_agentx_args({}, {"raw_mode": True, "concurrency": 2}) == [2]
    with pytest.raises(ValueError, match="run_mode"):
        validate_agentx_args({}, {"raw_mode": True, "run_mode": "unsupported"})


@pytest.mark.parametrize("path", ["", "  ", 123])
def test_validate_rejects_invalid_dataset_path_type_or_content(model, path):
    with pytest.raises(ValueError, match="dataset_path must be a non-empty string"):
        validate_agentx_args(model, {"variant": "full", "dataset_path": path})


def test_validate_requires_dataset_directory_and_trace_file(model, tmp_path):
    dataset_path = tmp_path / "dataset"
    args = {"variant": "full", "dataset_path": str(dataset_path)}
    with pytest.raises(FileNotFoundError, match="dataset_path is not a directory"):
        validate_agentx_args(model, args)

    dataset_path.mkdir()
    (dataset_path / "traces.jsonl").mkdir()
    with pytest.raises(FileNotFoundError, match="must contain traces.jsonl"):
        validate_agentx_args(model, args)


def test_validate_accepts_local_trace_dataset(model, tmp_path):
    (tmp_path / "traces.jsonl").write_text("{}\n", encoding="utf-8")
    assert validate_agentx_args(model, {"variant": "full", "dataset_path": str(tmp_path)}) == [1]


@pytest.mark.parametrize("value", [True, "900", 0, -1, float("inf"), float("nan")])
def test_validate_requires_positive_finite_duration(model, value):
    with pytest.raises(ValueError, match="duration must be a positive finite number"):
        validate_agentx_args(model, {"variant": "full", "duration": value})


def test_validate_benchmark_duration_threshold_is_inclusive(model):
    with pytest.raises(ValueError, match="benchmark duration must be >= 900 seconds"):
        validate_agentx_args(model, {"variant": "full", "duration": 899.99})
    assert validate_agentx_args(model, {"variant": "full", "duration": 900}) == [1]


def test_validate_trace_limit_is_only_allowed_in_smoke_mode(model):
    with pytest.raises(ValueError, match="trace_limit is only valid in smoke mode"):
        validate_agentx_args(model, {"variant": "full", "trace_limit": 1})
    args = {"variant": "full", "duration": 0.5, "trace_limit": 1, "run_mode": "smoke"}
    assert validate_agentx_args(model, args) == [1]


@pytest.mark.parametrize("value", [True, -1, float("inf"), "1"])
def test_validate_rejects_non_positive_finite_trace_limit(model, value):
    with pytest.raises(ValueError, match="trace_limit must be a positive finite number"):
        validate_agentx_args(model, {"variant": "full", "run_mode": "smoke", "trace_limit": value})


def test_validate_rejects_fractional_trace_limit(model):
    with pytest.raises(ValueError, match="trace_limit must be a positive integer"):
        validate_agentx_args(model, {"variant": "full", "run_mode": "smoke", "trace_limit": 1.5})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"api_key": False}, "api_key must be a non-empty string"),
        ({"api_key": " "}, "api_key must be a non-empty string"),
        ({"env": ["FOO=bar"]}, "env must be a mapping"),
        ({"env": {1: "value"}}, "Invalid AgentX env variable name"),
        ({"env": {"KEY-NAME": "value"}}, "Invalid AgentX env variable name"),
        ({"env": {"NAME": 1}}, "env value for NAME must be a string"),
        ({"env": {"NAME": "invalid\x00value"}}, "without NUL characters"),
        ({"extra_args": 1}, "extra_args must be a non-empty string"),
        ({"extra_args": "  "}, "extra_args must be a non-empty string"),
    ],
)
def test_validate_rejects_invalid_optional_parameters(model, overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_agentx_args(model, {"variant": "full", **overrides})


@pytest.mark.parametrize("key", ["AIPERF_CONFIG", "AIPERF_SCENARIO"])
def test_validate_disallows_workload_override_only_in_benchmark_mode(model, key):
    args = {"variant": "full", "env": {key: "local-override"}}
    with pytest.raises(ValueError, match=f"must not override the workload via {key}"):
        validate_agentx_args(model, args)
    assert validate_agentx_args(model, {**args, "run_mode": "smoke"}) == [1]


@pytest.mark.parametrize("key", ["agentx_python", "agentx_bin"])
def test_validate_executable_argument_can_be_a_command_or_existing_file(model, tmp_path, key):
    path = tmp_path / "custom executable"
    args = {"variant": "full", key: str(path)}
    with pytest.raises(FileNotFoundError, match=f"{key} does not exist"):
        validate_agentx_args(model, args)
    path.touch()
    assert validate_agentx_args(model, args) == [1]
    assert validate_agentx_args(model, {**args, key: "custom-command"}) == [1]
    with pytest.raises(ValueError, match=f"{key} must be a non-empty string"):
        validate_agentx_args(model, {**args, key: " "})
