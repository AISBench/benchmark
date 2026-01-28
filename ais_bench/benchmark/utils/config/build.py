import os
import copy
import ipaddress

from mmengine.config import ConfigDict

from ais_bench.benchmark.registry import (
    LOAD_DATASET,
    MODELS,
    PERF_METRIC_CALCULATORS,
)

from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES

logger = AISLogger()

def _validate_model_cfg(model_cfg: ConfigDict) -> dict:
    errors = {}

    def check(condition, key, message):
        if not condition:
            errors[key] = message

    validators = {
        "attr": lambda v: (
            v in ("local", "service"),
            "attr must be 'local' or 'service'",
        ),
        "abbr": lambda v: (isinstance(v, str), "abbr must be a string"),
        "use_timestamp": lambda v:(isinstance(v, bool), "use_timestamp must be a boolean"),
        "path": lambda v: (
            not v or (isinstance(v, str) and os.path.exists(v)),
            f"path is not accessible or does not exist: {v}",
        ),
        "model": lambda v: (
            not v or isinstance(v, str),
            "model can be omitted or must be a string",
        ),
        "request_rate": lambda v: (
            isinstance(v, (int, float)) and 0 <= v <= 64000,
            "request_rate must be a number in the range [0, 64000]",
        ),
        "retry": lambda v: (
            isinstance(v, int) and 0 <= v <= 1000,
            "retry must be an integer in the range [0, 1000]",
        ),
        "host_port": lambda v: (
            isinstance(v, int) and (0 < v < 65536),
            "host_port must be a valid port number in the range (0, 65536)",
        ),
        "max_out_len": lambda v: (
            isinstance(v, int) and 0 < v <= 131072,
            "max_out_len must be an integer in the range (0, 131072]",
        ),
        "batch_size": lambda v: (
            isinstance(v, int) and 0 < v <= 100000,
            "batch_size must be an integer in the range (0, 100000]",
        ),
        "generation_kwargs": lambda v: (
            isinstance(v, dict),
            "generation_kwargs must be a dictionary",
        ),
    }

    for key, value in model_cfg.items():
        if key == "type":
            continue  # Not configurable; skip validation

        if key == "host_ip":
            if value == "localhost":
                continue
            try:
                ipaddress.ip_address(value)
            except ValueError:
                errors[key] = "host_ip must be a valid IPv4 or IPv6 address"
            continue

        validator = validators.get(key)
        if validator:
            valid, msg = validator(value)
            check(valid, key, msg)

    traffic_field_validators = {
        "burstiness": (
            lambda v: v is None or v == "" or (isinstance(v, (int, float)) and v >= 0),
            "must be None, empty string '' or non-negative number",
        ),
        "ramp_up_strategy": (
            lambda v: v in [None, "", "linear", "exponential"],
            "must be None, empty string '', 'linear', or 'exponential'",
        ),
        "ramp_up_start_rps": (
            lambda v: v is None or v == "" or (isinstance(v, (int, float)) and v >= 0),
            "must be None, empty string or non-negative number",
        ),
        "ramp_up_end_rps": (
            lambda v: v is None or v == "" or (isinstance(v, (int, float)) and v >= 0),
            "must be None, empty string '' or non-negative number",
        ),
    }

    if "traffic_cfg" in model_cfg:
        traffic_cfg = model_cfg["traffic_cfg"]
        if not isinstance(traffic_cfg, dict):
            errors["traffic_cfg"] = "must be a dictionary"
        else:
            for field, (validator, error_msg) in traffic_field_validators.items():
                if field in traffic_cfg:
                    if not validator(traffic_cfg[field]):
                        errors[f"traffic_cfg.{field}"] = error_msg
    return errors


def build_dataset_from_cfg(dataset_cfg: ConfigDict):
    logger.debug(f"Building dataset from config: type={dataset_cfg.get('type')} abbr={dataset_cfg.get('abbr')}")
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_cfg.pop("infer_cfg", None)
    dataset_cfg.pop("eval_cfg", None)
    return LOAD_DATASET.build(dataset_cfg)


def build_model_from_cfg(model_cfg: ConfigDict):
    logger.debug(f"Building model from config: type={model_cfg.get('type')} abbr={model_cfg.get('abbr')}")
    model_cfg = copy.deepcopy(model_cfg)
    model_name = model_cfg.get("type", "").split(".")[-1]
    errors = _validate_model_cfg(model_cfg)
    if errors:
        logger.warning(f"Model config validation failed for {model_name}: {errors}")
        raise AISBenchConfigError(
            UTILS_CODES.MODEL_CONFIG_VALIDATE_FAILED,
            f"{model_name} build failed with the following errors: {errors}"
        )
    model_cfg.pop("use_timestamp", None)
    model_cfg.pop("run_cfg", None)
    model_cfg.pop("request_rate", None)
    batch_size = model_cfg.pop("batch_size", None)
    abbr = model_cfg.pop("abbr", None)
    attr = model_cfg.pop("attr", None)
    model_cfg.pop("summarizer_abbr", None)
    model_cfg.pop("pred_postprocessor", None)
    model_cfg.pop("min_out_len", None)
    model_cfg.pop("returns_tool_calls", None)
    model_cfg.pop("traffic_cfg", None)
    if attr == "local" and abbr == "mindformer-model" :
        return MODELS.build(model_cfg, batch_size = batch_size)
    else :
        return MODELS.build(model_cfg)

def build_perf_metric_calculator_from_cfg(metric_cfg: ConfigDict):
    logger.debug(f"Building perf metric calculator config: type={metric_cfg.get('type')}")
    metric_cfg = copy.deepcopy(metric_cfg)
    return PERF_METRIC_CALCULATORS.build(metric_cfg)
