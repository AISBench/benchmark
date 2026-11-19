"""AgentX 参数规格与启动前校验。"""

import math
import os.path as osp
import re
import shlex
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

from ais_bench.benchmark.cli.utils import SWEEP_COOLDOWN_SECONDS

DEFAULT_ENDPOINT = "/v1/chat/completions"
# agentx 场景要求 profiling 时长不低于 900 秒才能进入稳态并触发 KV offload。
MIN_BENCHMARK_DURATION = 900.0

DATASETS: Dict[str, Dict[str, str]] = {
    "full": {
        "alias": "semianalysis_cc_traces_weka_062126",
        "hf_repo": "semianalysisai/cc-traces-weka-062126",
        "hf_revision": "23f152f6f0f9399a85901b89a6458def0ef16729",
        "sha256": "29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276",
    },
    "256k": {
        "alias": "semianalysis_cc_traces_weka_062126_256k",
        "hf_repo": "semianalysisai/cc-traces-weka-062126-256k",
        "hf_revision": "8fecd2fc56694469f758f0afbbb6335ad3043740",
        "sha256": "e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef",
    },
}

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def resolve_variant(variant: Optional[str]) -> str:
    """Require an explicit corpus variant; never infer it from the model name."""
    if not isinstance(variant, str) or not variant.strip():
        raise ValueError(
            "AgentX requires an explicit dataset variant: 'full' or '256k' "
            "(--agentx-dataset-variant or datasets[0].args.variant)."
        )
    variant = variant.strip()
    if variant not in DATASETS:
        raise ValueError(f"Unknown AgentX dataset variant {variant!r}; expected {sorted(DATASETS)}.")
    return variant


def resolve_endpoint(value: Any) -> str:
    if value is None:
        return DEFAULT_ENDPOINT
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"AgentX endpoint must be a non-empty path such as {DEFAULT_ENDPOINT}."
        )
    endpoint = value.strip()
    if (
        not endpoint.startswith("/")
        or any(char in endpoint for char in ("?", "#"))
        or any(char.isspace() for char in endpoint)
    ):
        raise ValueError(
            f"AgentX endpoint must be an absolute path without query, fragment or spaces: {endpoint!r}."
        )
    return endpoint


def normalize_base_url(url: str, endpoint: str = DEFAULT_ENDPOINT) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"Invalid model url {url!r}; expected http(s)://host:port[/path].")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("AgentX model url must not contain userinfo, query or fragment.")
    path = parts.path.rstrip("/")
    if path.endswith(endpoint.rstrip("/")):
        raise ValueError(
            f"AgentX model url must be a base URL without {endpoint}; "
            "configure the path with --endpoint instead."
        )
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def resolve_cooldown(value: Any) -> float:
    """Seconds to wait between consecutive performance runs; default 5."""
    if value is None:
        return float(SWEEP_COOLDOWN_SECONDS)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(
            f"AgentX cooldown must be a non-negative finite number of seconds, got {value!r}."
        )
    return float(value)


def _concurrency_levels(value: Any) -> List[int]:
    """Normalize ``--batch-size`` into ordered, unique positive levels."""
    if isinstance(value, bool):
        raise ValueError(f"AgentX concurrency must be an integer, got {value!r}.")
    if isinstance(value, int):
        levels = [value]
    elif isinstance(value, (list, tuple)):
        levels = list(value)
    else:
        raise ValueError(f"AgentX concurrency must be an integer or a list of integers, got {value!r}.")
    if not levels:
        raise ValueError("AgentX concurrency requires at least one level.")
    seen = set()
    for level in levels:
        if isinstance(level, bool) or not isinstance(level, int) or level < 1:
            raise ValueError(f"AgentX concurrency must be a positive integer, got {level!r}.")
        if level in seen:
            raise ValueError(f"AgentX concurrency level {level} is duplicated.")
        seen.add(level)
    return levels


def _looks_like_path(value: str) -> bool:
    return osp.dirname(value) != ""


def _explicit_local_path(value: str) -> bool:
    return value.startswith((".", "~", "/", "\\")) or osp.isabs(value) or bool(re.match(r"^[A-Za-z]:[\\/]", value))


def extra_arg_value(extra_args: Optional[str], option: str) -> Optional[str]:
    """从 --agentx-extra-arg 原文取某个选项的取值，仅供 raw 模式补默认值。"""
    if not extra_args:
        return None
    text = extra_args.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    try:
        tokens = shlex.split(text)
    except ValueError:
        return None
    for index, token in enumerate(tokens):
        if token == option:
            return tokens[index + 1] if index + 1 < len(tokens) else None
        if token.startswith(option + "="):
            return token.split("=", 1)[1]
    return None


def validate_agentx_args(model_cfg: Dict[str, Any], args: Dict[str, Any]) -> List[int]:
    """CLI/配置共用校验；在下载和 Harness 启动前拒绝非法配置。"""
    levels = _concurrency_levels(args.get("concurrency", 1))
    resolve_cooldown(args.get("cooldown"))
    if not args.get("raw_mode"):
        model = model_cfg.get("model") or args.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("AgentX requires a non-empty model name (models[0].model).")
        url = model_cfg.get("url") or args.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("AgentX requires models[0].url with the service base URL.")
        endpoint = resolve_endpoint(model_cfg.get("endpoint"))
        normalize_base_url(url.strip(), endpoint)

        model_path = model_cfg.get("path")
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("AgentX requires models[0].path for the model/tokenizer.")
        model_path = model_path.strip()
        if _explicit_local_path(model_path) and not osp.isdir(osp.expanduser(model_path)):
            raise FileNotFoundError(f"AgentX local model/tokenizer directory does not exist: {model_path}")
        resolve_variant(args.get("variant"))
    run_mode = args.get("run_mode") or "benchmark"
    if run_mode not in ("benchmark", "smoke"):
        raise ValueError("AgentX run_mode must be 'benchmark' or 'smoke'.")

    dataset_path = args.get("dataset_path")
    if dataset_path is not None:
        if not isinstance(dataset_path, str) or not dataset_path.strip():
            raise ValueError("AgentX dataset_path must be a non-empty string.")
        if not osp.isdir(dataset_path):
            raise FileNotFoundError(f"AgentX dataset_path is not a directory: {dataset_path}")
        if not osp.isfile(osp.join(dataset_path, "traces.jsonl")):
            raise FileNotFoundError(f"dataset_path must contain traces.jsonl: {dataset_path}")

    for key in ("duration", "trace_limit"):
        value = args.get(key)
        if value is None:
            continue
        if key == "trace_limit" and run_mode != "smoke":
            raise ValueError("AgentX trace_limit is only valid in smoke mode.")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"AgentX {key} must be a positive finite number.")
        if key == "trace_limit" and not isinstance(value, int):
            raise ValueError("AgentX trace_limit must be a positive integer.")
        if key == "duration" and run_mode == "benchmark" and value < MIN_BENCHMARK_DURATION:
            raise ValueError(
                f"AgentX benchmark duration must be >= {MIN_BENCHMARK_DURATION:g} seconds; "
                f"got {value}. Shorter runs are only allowed in smoke mode."
            )

    api_key = args.get("api_key")
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
        raise ValueError("AgentX api_key must be a non-empty string.")

    env = args.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("AgentX env must be a mapping of KEY=VALUE.")
    for key, value in env.items():
        if not isinstance(key, str) or not _ENV_NAME_RE.fullmatch(key):
            raise ValueError(f"Invalid AgentX env variable name: {key!r}.")
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError(f"AgentX env value for {key} must be a string without NUL characters.")
        if run_mode == "benchmark" and key in {"AIPERF_CONFIG", "AIPERF_SCENARIO"}:
            raise ValueError(f"AgentX benchmark env must not override the workload via {key}.")

    # extra_args 是通用参数入口：用户直接写出 agentx 原始参数，原样追加到命令末尾。
    extra_args = args.get("extra_args")
    if extra_args is not None and (not isinstance(extra_args, str) or not extra_args.strip()):
        raise ValueError("AgentX extra_args must be a non-empty string.")

    for name in ("agentx_python", "agentx_bin"):
        value = args.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"AgentX {name} must be a non-empty string.")
        if _looks_like_path(value) and not osp.isfile(osp.expanduser(value)):
            raise FileNotFoundError(f"AgentX {name} does not exist: {value}")
    return levels
