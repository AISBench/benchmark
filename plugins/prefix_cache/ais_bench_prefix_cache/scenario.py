from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ScenarioValidationError


_ALLOWED = {
    "": {"schema_version", "run", "tokenizer", "corpus", "requests", "prefix_cache", "service", "validation", "aisbench"},
    "run": {"run_id", "random_seed", "output_dir", "overwrite"},
    "tokenizer": {"path", "block_size", "revision", "trust_remote_code"},
    "corpus": {"path", "field", "selection"},
    "corpus.selection": {"mode", "values", "indices", "question_sha256"},
    "requests": {"count", "input_length", "output_length"},
    "requests.input_length": {"mode", "value", "values", "ranges", "min", "max", "mean", "std", "path"},
    "requests.output_length": {"mode", "value", "min", "max", "mean", "std", "path"},
    "prefix_cache": {"mode", "target_hit_rate", "seed_blocks", "minimum_non_shared_length", "groups", "order"},
    "prefix_cache.groups": {"count", "assignment", "overrides"},
    "prefix_cache.groups.assignment": {"mode", "exponent", "weights"},
    "prefix_cache.order": {"strategy"},
    "service": {"inference_url", "metrics_url", "reset_url", "model", "dp_size", "assume_empty_cache", "engine_label_map", "timeout_seconds", "api_key"},
    "validation": {"target_warning_pp", "actual_warning_pp"},
    "aisbench": {"config", "work_dir", "extra_args"},
}

_MODES = {
    "input": {"fixed", "explicit", "range", "truncated_normal", "csv"},
    "output": {"fixed", "uniform", "truncated_normal", "csv"},
    "selection": {"random", "indices", "question_sha256", "mixed"},
    "assignment": {"uniform", "zipf", "weights"},
    "order": {"sequential", "within_group_shuffle", "interleave", "global_shuffle", "input_len_asc"},
    "cache": {"cold", "warmup"},
}


def _require_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScenarioValidationError(f"{path or 'scenario'} must be an object")
    return value


def _strict_keys(value: dict[str, Any], path: str) -> None:
    allowed = _ALLOWED.get(path)
    if allowed is not None:
        unknown = sorted(set(value) - allowed)
        if unknown:
            prefix = f"{path}." if path else ""
            raise ScenarioValidationError(f"unknown field: {prefix}{unknown[0]}")
    for key, child in value.items():
        child_path = f"{path}.{key}" if path else key
        if child_path in _ALLOWED:
            _strict_keys(_require_dict(child, child_path), child_path)


def _positive(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ScenarioValidationError(f"{path} must be a positive integer")
    return value


def _mode(section: dict[str, Any], allowed: set[str], path: str) -> str:
    value = section.get("mode")
    if value not in allowed:
        raise ScenarioValidationError(f"{path}.mode must be one of {sorted(allowed)}")
    return value


def _validate_input_config(config: dict[str, Any], path: str, base: Path, expected_count: int | None) -> None:
    mode = _mode(config, _MODES["input"], path)
    unknown = set(config) - {"mode", "value", "values", "ranges", "min", "max", "mean", "std", "path"}
    if unknown:
        raise ScenarioValidationError(f"unknown field: {path}.{sorted(unknown)[0]}")
    if mode == "fixed":
        _positive(config.get("value"), f"{path}.value")
    elif mode == "explicit":
        values = config.get("values")
        if not isinstance(values, list) or not values:
            raise ScenarioValidationError(f"{path}.values must be a non-empty list")
        for index, value in enumerate(values):
            _positive(value, f"{path}.values[{index}]")
        if expected_count is not None and len(values) != expected_count:
            raise ScenarioValidationError(f"{path}.values length must equal expected request count")
    elif mode == "range":
        ranges = config.get("ranges")
        if not isinstance(ranges, list) or not ranges:
            raise ScenarioValidationError(f"{path}.ranges must be a non-empty list")
        total = 0
        for index, item in enumerate(ranges):
            if not isinstance(item, dict) or set(item) - {"min", "max", "count"}:
                raise ScenarioValidationError(f"{path}.ranges[{index}] has invalid fields")
            low = _positive(item.get("min"), f"{path}.ranges[{index}].min")
            high = _positive(item.get("max"), f"{path}.ranges[{index}].max")
            if high < low:
                raise ScenarioValidationError(f"{path}.ranges[{index}].max must be >= min")
            total += _positive(item.get("count"), f"{path}.ranges[{index}].count")
        if expected_count is not None and total != expected_count:
            raise ScenarioValidationError(f"{path} range counts must equal expected request count")
    elif mode == "truncated_normal":
        low = _positive(config.get("min"), f"{path}.min")
        high = _positive(config.get("max"), f"{path}.max")
        if high < low:
            raise ScenarioValidationError(f"{path}.max must be >= min")
        if "std" in config and float(config["std"]) <= 0:
            raise ScenarioValidationError(f"{path}.std must be positive")
    else:
        if not isinstance(config.get("path"), str) or not config["path"]:
            raise ScenarioValidationError(f"{path}.path must be a non-empty string")
        config["path"] = _resolve_path(base, config["path"])


def _validate_output_config(config: dict[str, Any], path: str, base: Path) -> None:
    mode = _mode(config, _MODES["output"], path)
    unknown = set(config) - {"mode", "value", "min", "max", "mean", "std", "path"}
    if unknown:
        raise ScenarioValidationError(f"unknown field: {path}.{sorted(unknown)[0]}")
    if mode == "fixed":
        _positive(config.get("value"), f"{path}.value")
    elif mode in {"uniform", "truncated_normal"}:
        low = _positive(config.get("min"), f"{path}.min")
        high = _positive(config.get("max"), f"{path}.max")
        if high < low:
            raise ScenarioValidationError(f"{path}.max must be >= min")
        if mode == "truncated_normal" and "std" in config and float(config["std"]) <= 0:
            raise ScenarioValidationError(f"{path}.std must be positive")
    else:
        if not isinstance(config.get("path"), str) or not config["path"]:
            raise ScenarioValidationError(f"{path}.path must be a non-empty string")
        config["path"] = _resolve_path(base, config["path"])


def _minimum_input_tokens(config: dict[str, Any], path: str) -> int:
    mode = config["mode"]
    if mode == "fixed":
        return int(config["value"])
    if mode == "explicit":
        return min(int(value) for value in config["values"])
    if mode == "range":
        return min(int(item["min"]) for item in config["ranges"])
    if mode == "truncated_normal":
        return int(config["min"])
    try:
        with Path(config["path"]).open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
    except OSError as exc:
        raise ScenarioValidationError(f"{path} CSV cannot be read: {exc}") from exc
    aliases = ("input_prompt_tokens", "content_tokens", "input_tokens")
    if not rows:
        raise ScenarioValidationError(f"{path} CSV must contain at least one data row")
    column = next((name for name in aliases if name in rows[0]), None)
    if column is None:
        raise ScenarioValidationError(f"{path} CSV requires one of columns {list(aliases)}")
    try:
        return min(int(row[column]) for row in rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise ScenarioValidationError(f"{path} CSV contains an invalid input length: {exc}") from exc


@dataclass(frozen=True)
class Scenario:
    source_path: Path
    data: dict[str, Any]

    @property
    def run_id(self) -> str:
        return self.data["run"]["run_id"]

    @property
    def random_seed(self) -> int:
        return self.data["run"]["random_seed"]

    @property
    def output_dir(self) -> Path:
        return Path(self.data["run"]["output_dir"])

    @property
    def block_size(self) -> int:
        return self.data["tokenizer"]["block_size"]

    @property
    def cache_mode(self) -> str:
        return self.data["prefix_cache"]["mode"]

    @property
    def dp_size(self) -> int:
        return self.data["service"]["dp_size"]

    def section(self, name: str) -> dict[str, Any]:
        return self.data[name]

    def to_effective_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def _resolve_path(base: Path, value: str) -> str:
    path = Path(value)
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _validate(raw: dict[str, Any], source: Path) -> dict[str, Any]:
    _strict_keys(raw, "")
    required = {"schema_version", "run", "tokenizer", "corpus", "requests", "prefix_cache", "service"}
    missing = sorted(required - set(raw))
    if missing:
        raise ScenarioValidationError(f"missing field: {missing[0]}")
    data = copy.deepcopy(raw)
    data.setdefault("validation", {})
    data.setdefault("aisbench", {})
    run = _require_dict(data["run"], "run")
    tokenizer = _require_dict(data["tokenizer"], "tokenizer")
    corpus = _require_dict(data["corpus"], "corpus")
    requests = _require_dict(data["requests"], "requests")
    pc = _require_dict(data["prefix_cache"], "prefix_cache")
    service = _require_dict(data["service"], "service")
    if data["schema_version"] != "1.0":
        raise ScenarioValidationError("schema_version must be '1.0'")
    if not isinstance(run.get("run_id"), str) or not run["run_id"].strip():
        raise ScenarioValidationError("run.run_id must be a non-empty string")
    if isinstance(run.get("random_seed"), bool) or not isinstance(run.get("random_seed"), int):
        raise ScenarioValidationError("run.random_seed must be an integer")
    run.setdefault("overwrite", False)
    run["output_dir"] = _resolve_path(source.parent, run["output_dir"])
    tokenizer["block_size"] = _positive(tokenizer.get("block_size"), "tokenizer.block_size")
    tokenizer.setdefault("revision", None)
    tokenizer.setdefault("trust_remote_code", False)
    corpus.setdefault("field", "question")
    corpus["path"] = _resolve_path(source.parent, corpus["path"])
    selection = corpus.setdefault("selection", {"mode": "random"})
    _mode(selection, _MODES["selection"], "corpus.selection")
    count = _positive(requests.get("count"), "requests.count")
    input_cfg = _require_dict(requests.get("input_length"), "requests.input_length")
    output_cfg = _require_dict(requests.get("output_length"), "requests.output_length")
    _validate_input_config(input_cfg, "requests.input_length", source.parent, count)
    _validate_output_config(output_cfg, "requests.output_length", source.parent)
    cache_mode = _mode(pc, _MODES["cache"], "prefix_cache")
    target = pc.get("target_hit_rate")
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not 0 <= target <= 1:
        raise ScenarioValidationError("prefix_cache.target_hit_rate must be in [0, 1]")
    pc["seed_blocks"] = _positive(pc.get("seed_blocks", 1), "prefix_cache.seed_blocks")
    seed_tokens = tokenizer["block_size"] * pc["seed_blocks"]
    pc["minimum_non_shared_length"] = _positive(
        pc.get("minimum_non_shared_length", seed_tokens),
        "prefix_cache.minimum_non_shared_length",
    )
    if pc["minimum_non_shared_length"] < seed_tokens:
        raise ScenarioValidationError(
            f"prefix_cache.minimum_non_shared_length must be at least seed length {seed_tokens}"
        )
    reserved_tokens = pc["minimum_non_shared_length"]
    if _minimum_input_tokens(input_cfg, "requests.input_length") < reserved_tokens:
        raise ScenarioValidationError(
            f"requests.input_length must be at least {reserved_tokens} tokens to contain the configured non-shared region"
        )
    groups = _require_dict(pc.get("groups"), "prefix_cache.groups")
    groups["count"] = _positive(groups.get("count"), "prefix_cache.groups.count")
    assignment = groups.setdefault("assignment", {"mode": "uniform"})
    _mode(assignment, _MODES["assignment"], "prefix_cache.groups.assignment")
    overrides = groups.setdefault("overrides", {})
    if not isinstance(overrides, dict):
        raise ScenarioValidationError("prefix_cache.groups.overrides must be an object")
    for group_id, override in overrides.items():
        expected_group_id = group_id.startswith("group-") and group_id[6:].isdigit() and int(group_id[6:]) < groups["count"]
        if not expected_group_id:
            raise ScenarioValidationError(f"invalid Prefix Group override id: {group_id}")
        if not isinstance(override, dict):
            raise ScenarioValidationError(f"prefix_cache.groups.overrides.{group_id} must be an object")
        unknown = set(override) - {"input_length", "output_length", "corpus_selection"}
        if unknown:
            raise ScenarioValidationError(f"unknown field: prefix_cache.groups.overrides.{group_id}.{sorted(unknown)[0]}")
        if "input_length" in override:
            _validate_input_config(override["input_length"], f"prefix_cache.groups.overrides.{group_id}.input_length", source.parent, None)
            if _minimum_input_tokens(override["input_length"], f"prefix_cache.groups.overrides.{group_id}.input_length") < reserved_tokens:
                raise ScenarioValidationError(
                    f"prefix_cache.groups.overrides.{group_id}.input_length must be at least {reserved_tokens} tokens to contain the configured non-shared region"
                )
        if "output_length" in override:
            _validate_output_config(override["output_length"], f"prefix_cache.groups.overrides.{group_id}.output_length", source.parent)
        if "corpus_selection" in override:
            _mode(override["corpus_selection"], _MODES["selection"], f"prefix_cache.groups.overrides.{group_id}.corpus_selection")
    order = pc.setdefault("order", {"strategy": "interleave"})
    if order.get("strategy") not in _MODES["order"]:
        raise ScenarioValidationError(f"prefix_cache.order.strategy must be one of {sorted(_MODES['order'])}")
    service["dp_size"] = _positive(service.get("dp_size", 1), "service.dp_size")
    service.setdefault("reset_url", None)
    service.setdefault("assume_empty_cache", False)
    service.setdefault("engine_label_map", {})
    service.setdefault("timeout_seconds", 30)
    service.setdefault("api_key", "")
    for field in ("inference_url", "metrics_url", "model"):
        if not isinstance(service.get(field), str) or not service[field]:
            raise ScenarioValidationError(f"service.{field} must be a non-empty string")
    validation = data["validation"]
    validation.setdefault("target_warning_pp", 1.0)
    validation.setdefault("actual_warning_pp", 5.0)
    if cache_mode == "cold" and service["dp_size"] > 1 and not service["inference_url"]:
        raise ScenarioValidationError("cold multi-DP requires inference_url")
    return data


def load_scenario(path: Path | str) -> Scenario:
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioValidationError(f"cannot read scenario {source}: {exc}") from exc
    return Scenario(source, _validate(_require_dict(raw, "scenario"), source))
