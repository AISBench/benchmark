from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from datetime import datetime
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
    """校验 value 必须是 dict（JSON 对象），否则抛校验错误，原样返回。"""
    if not isinstance(value, dict):
        raise ScenarioValidationError(f"{path or 'scenario'} must be an object")
    return value


def _strict_keys(value: dict[str, Any], path: str) -> None:
    """递归校验 dict 的键是否都在白名单 _ALLOWED 内，拒绝未知字段。"""
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
    """校验 value 是正整数（排除 bool），返回原值。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ScenarioValidationError(f"{path} must be a positive integer")
    return value


def _mode(section: dict[str, Any], allowed: set[str], path: str) -> str:
    """校验 section 的 mode 取值必须在 allowed 集合内，返回 mode。"""
    value = section.get("mode")
    if value not in allowed:
        raise ScenarioValidationError(f"{path}.mode must be one of {sorted(allowed)}")
    return value


def _validate_input_config(config: dict[str, Any], path: str, base: Path, expected_count: int | None) -> None:
    """按 mode 校验输入长度配置（fixed/explicit/range/truncated_normal/csv），并解析 csv 路径。"""
    mode = _mode(config, _MODES["input"], path)
    unknown = set(config) - {"mode", "value", "values", "ranges", "min", "max", "mean", "std", "path"}
    if unknown:
        raise ScenarioValidationError(f"unknown field: {path}.{sorted(unknown)[0]}")
    if mode == "fixed":
        _positive(config.get("value"), f"{path}.value")
    elif mode == "explicit":
        # 显式列表：逐项校验为正整数，且数量须等于请求总数。
        values = config.get("values")
        if not isinstance(values, list) or not values:
            raise ScenarioValidationError(f"{path}.values must be a non-empty list")
        for index, value in enumerate(values):
            _positive(value, f"{path}.values[{index}]")
        if expected_count is not None and len(values) != expected_count:
            raise ScenarioValidationError(f"{path}.values length must equal expected request count")
    elif mode == "range":
        # 区间抽样：每段校验 min/max/count，且总数须等于请求总数。
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
        # 截断正态：校验 min/max 区间与 std>0。
        low = _positive(config.get("min"), f"{path}.min")
        high = _positive(config.get("max"), f"{path}.max")
        if high < low:
            raise ScenarioValidationError(f"{path}.max must be >= min")
        if "std" in config and float(config["std"]) <= 0:
            raise ScenarioValidationError(f"{path}.std must be positive")
    else:
        # csv 模式：要求 path 非空并解析为绝对路径。
        if not isinstance(config.get("path"), str) or not config["path"]:
            raise ScenarioValidationError(f"{path}.path must be a non-empty string")
        config["path"] = _resolve_path(base, config["path"])


def _validate_output_config(config: dict[str, Any], path: str, base: Path) -> None:
    """按 mode 校验输出长度配置（fixed/uniform/truncated_normal/csv），并解析 csv 路径。"""
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
    """计算输入长度配置能产生的最小 token 数（用于校验非共享区约束）。"""
    mode = config["mode"]
    if mode == "fixed":
        return int(config["value"])
    if mode == "explicit":
        return min(int(value) for value in config["values"])
    if mode == "range":
        return min(int(item["min"]) for item in config["ranges"])
    if mode == "truncated_normal":
        return int(config["min"])
    # csv 模式：读取文件并取 input 长度列的最小值。
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
    """校验通过后的场景对象：保存源文件路径与规范化后的有效配置。"""

    source_path: Path   # 场景 JSON 文件绝对路径
    data: dict[str, Any]  # 规范化后的完整配置（含默认值）

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
        """按段名取配置，如 scenario.section("service")。"""
        return self.data[name]

    def to_effective_dict(self) -> dict[str, Any]:
        """返回一份深拷贝的有效配置，调用方可安全修改而不影响内部数据。"""
        return copy.deepcopy(self.data)


def new_execution_timestamp() -> str:
    """Return a filename-safe local timestamp with second resolution."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def with_execution_timestamp(scenario: Scenario, timestamp: str) -> Scenario:
    """Append one execution timestamp to both run_id and output_dir."""
    if len(timestamp) != 15 or timestamp[8] != "_" or not (timestamp[:8] + timestamp[9:]).isdigit():
        raise ScenarioValidationError("execution timestamp must use YYYYMMDD_HHMMSS")
    data = scenario.to_effective_dict()
    data["run"]["run_id"] = f"{scenario.run_id}_{timestamp}"
    output_dir = scenario.output_dir
    if not output_dir.name:
        raise ScenarioValidationError("run.output_dir must have a final directory name")
    data["run"]["output_dir"] = str(output_dir.parent / f"{output_dir.name}_{timestamp}")
    return Scenario(scenario.source_path, data)


def _resolve_path(base: Path, value: str) -> str:
    """把配置里的路径解析为绝对路径：相对路径以 base 为基准。"""
    path = Path(value)
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def _validate(raw: dict[str, Any], source: Path) -> dict[str, Any]:
    """对原始场景 dict 做完整语义校验与默认值填充，返回规范化副本。

    校验缺失/未知字段、类型与取值约束、路径解析、prefix cache 相关的一致性
    （如非共享区下限、分组覆盖 id、cold 多 DP 的地址要求），并原地补默认值。
    """
    _strict_keys(raw, "")
    # 深拷贝后再修改，避免污染调用方的原始数据。
    data = copy.deepcopy(raw)
    # Scenario 允许省略配置值；默认值与 config_examples/scenario.example.json
    # 保持一致。多态配置（range/csv 等）只在整个配置缺失或 fixed 模式下
    # 填默认，避免给其他 mode 注入不合法的 fixed 字段。
    data.setdefault("schema_version", "1.0")
    data.setdefault("run", {})
    data.setdefault("tokenizer", {})
    data.setdefault("corpus", {})
    data.setdefault("requests", {})
    data.setdefault("prefix_cache", {})
    data.setdefault("service", {})
    data.setdefault("validation", {})
    data.setdefault("aisbench", {})
    run = _require_dict(data["run"], "run")
    tokenizer = _require_dict(data["tokenizer"], "tokenizer")
    corpus = _require_dict(data["corpus"], "corpus")
    requests = _require_dict(data["requests"], "requests")
    pc = _require_dict(data["prefix_cache"], "prefix_cache")
    service = _require_dict(data["service"], "service")
    run.setdefault("run_id", "gsm8k-prefix-cache-60")
    run.setdefault("random_seed", 42)
    run.setdefault("output_dir", "./outputs/gsm8k-prefix-cache-60")
    tokenizer.setdefault("path", "/home/weights/Qwen3.6-27B")
    tokenizer.setdefault("block_size", 16)
    corpus.setdefault("path", "./GSM8K.jsonl")
    requests.setdefault("count", 100)
    input_cfg = requests.setdefault("input_length", {"mode": "fixed", "value": 1024})
    if isinstance(input_cfg, dict):
        input_cfg.setdefault("mode", "fixed")
        if input_cfg["mode"] == "fixed":
            input_cfg.setdefault("value", 1024)
    output_cfg = requests.setdefault("output_length", {"mode": "fixed", "value": 32})
    if isinstance(output_cfg, dict):
        output_cfg.setdefault("mode", "fixed")
        if output_cfg["mode"] == "fixed":
            output_cfg.setdefault("value", 32)
    pc.setdefault("mode", "warmup")
    pc.setdefault("target_hit_rate", 0.6)
    pc.setdefault("seed_blocks", 1)
    groups_cfg = pc.setdefault("groups", {"count": 1, "assignment": {"mode": "uniform"}})
    if isinstance(groups_cfg, dict):
        groups_cfg.setdefault("count", 1)
        assignment_cfg = groups_cfg.setdefault("assignment", {"mode": "uniform"})
        if isinstance(assignment_cfg, dict):
            assignment_cfg.setdefault("mode", "uniform")
    order_cfg = pc.setdefault("order", {"strategy": "interleave"})
    if isinstance(order_cfg, dict):
        order_cfg.setdefault("strategy", "interleave")
    service.setdefault("inference_url", "http://127.0.0.1:8000/v1/completions")
    service.setdefault("metrics_url", "http://127.0.0.1:8000/metrics")
    service.setdefault("reset_url", "http://127.0.0.1:8000/reset_prefix_cache")
    service.setdefault("model", "model-name")
    service.setdefault("dp_size", 2)
    service.setdefault("assume_empty_cache", False)
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
    if isinstance(selection, dict):
        selection.setdefault("mode", "random")
    _mode(selection, _MODES["selection"], "corpus.selection")
    count = _positive(requests.get("count"), "requests.count")
    input_cfg = _require_dict(input_cfg, "requests.input_length")
    output_cfg = _require_dict(output_cfg, "requests.output_length")
    _validate_input_config(input_cfg, "requests.input_length", source.parent, count)
    _validate_output_config(output_cfg, "requests.output_length", source.parent)
    cache_mode = _mode(pc, _MODES["cache"], "prefix_cache")
    target = pc.get("target_hit_rate")
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not 0 <= target <= 1:
        raise ScenarioValidationError("prefix_cache.target_hit_rate must be in [0, 1]")
    pc["seed_blocks"] = _positive(pc.get("seed_blocks", 1), "prefix_cache.seed_blocks")
    # 非共享区下限 = seed 长度，且须保证输入长度能容纳该非共享区。
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
        # 校验 override 的 id 必须是合法 group-N 且未越界，再校验其字段。
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
    validation = _require_dict(data["validation"], "validation")
    validation.setdefault("target_warning_pp", 1.0)
    validation.setdefault("actual_warning_pp", 5.0)
    # cold 多 DP 必须显式提供推理地址，否则无法路由。
    if cache_mode == "cold" and service["dp_size"] > 1 and not service["inference_url"]:
        raise ScenarioValidationError("cold multi-DP requires inference_url")
    return data


def load_scenario(path: Path | str) -> Scenario:
    """读取并解析场景 JSON 文件，校验后返回 Scenario 对象。

    任何读取/解析/校验失败都会以 ScenarioValidationError 形式抛出。
    """
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioValidationError(f"cannot read scenario {source}: {exc}") from exc
    return Scenario(source, _validate(_require_dict(raw, "scenario"), source))
