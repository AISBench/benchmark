import os
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union

from ais_bench.benchmark import __version__
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import TMAN_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchConfigError
from ais_bench.benchmark.cli.utils import (
    DATASETS_NEED_MODELS,
    MAX_BATCH_SIZE,
    MAX_REQUEST_COUNT,
    MAX_REQUEST_SIZE,
    get_config_type,
)

logger = AISLogger()

# 多组合 sweep 的内部调度字段（非 CLI、不进 manifest）
SWEEP_COOLDOWN_SECONDS = 5
MANIFEST_FILENAME = "combos_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


@dataclass
class CombinationSpec:
    """一个并发档组合的生效参数快照。"""
    index: int
    batch_size: int
    num_prompts: Optional[int]
    request_count: Optional[int]
    request_size: Optional[int]
    prefix_len: Optional[int]
    model_abbr: str
    dataset_abbr: str
    model_name: Optional[str] = None
    base_model_abbr: Optional[str] = None
    dataset_type: Optional[str] = None
    stream: bool = False
    ttft_valid: bool = False
    use_timestamp: bool = False
    max_out_len: Optional[int] = None


@dataclass
class SweepResult:
    """组合展开结果。"""
    models: List[dict] = field(default_factory=list)
    datasets: List[dict] = field(default_factory=list)
    combinations: List[dict] = field(default_factory=list)
    manifest: Optional[dict] = None
    sweep_cooldown_s: int = 0


class PairingStrategy(ABC):
    """配对策略接口。"""

    @abstractmethod
    def pair(self, left, right) -> List[Tuple[Any, Any]]:
        pass


class PairedStrategy(PairingStrategy):
    """双列表等长：按下标一一配对。"""

    def pair(self, left, right) -> List[Tuple[Any, Any]]:
        return list(zip(left, right))


class BroadcastStrategy(PairingStrategy):
    """一列表一标量：标量广播到列表每一项。"""

    def pair(self, left, right) -> List[Tuple[Any, Any]]:
        if isinstance(left, list) and not isinstance(right, list):
            return [(item, right) for item in left]
        if isinstance(right, list) and not isinstance(left, list):
            return [(left, item) for item in right]
        return [(left, right)]


def _normalize_sweep_arg(value) -> Union[int, List[int], None]:
    """长度 1 的列表解包为 int；多元素列表原样返回；空列表报错。"""
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            raise AISBenchConfigError(
                TMAN_CODES.INVALID_ARG_VALUE_IN_CMD,
                "`--batch-size`/`--num-prompts` must not be an empty list.",
            )
        if len(value) == 1:
            return int(value[0])
        return [int(item) for item in value]
    return int(value)


def validate_pairing(left: Union[int, List[int], None],
                     right: Union[int, List[int], None],
                     request_count: Optional[int] = None,
                     test_range_set: bool = False
                     ) -> Tuple[List[Tuple[Optional[int], Optional[int]]], List[str]]:
    """`--batch-size` ↔ `--num-prompts` 配对校验。

    - 双列表等长 → PairedStrategy 按下标配对
    - 一列表一标量 → BroadcastStrategy 标量广播
    - 双列表不等长 → 抛 AISBenchConfigError（不静默截断）
    - 双标量（含单值列表归一后的单档）同样走并发窗口检查，不做特例跳过
    - `batch_size` 是客户端最大并发请求数（并发窗口），请求完成后立即补位、
      不存在固定波次，因此这里不做整除性检查；仅在并发窗口大于生效请求数
      （窗口跑不满、实际并发达不到设定值）时 WARNING，不阻断
    - 生效请求数：指定 `num_prompts` 时为 min(num_prompts, request_count)；
      未指定时默认对全量数据集评测，取 request_count；两者均未知或配置已显式
      设置 test_range 时跳过该检查
    """
    warnings: List[str] = []
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise AISBenchConfigError(
                TMAN_CODES.INVALID_ARG_VALUE_IN_CMD,
                f"`--batch-size` and `--num-prompts` must have the same length when "
                f"both are lists (got {len(left)} vs {len(right)}); make them the same "
                f"length or pass a single value on one side to broadcast.",
            )
        strategy: PairingStrategy = PairedStrategy()
    else:
        strategy = BroadcastStrategy()

    pairs = strategy.pair(left, right)
    if test_range_set:
        return pairs, warnings

    for batch_size, num_prompts in pairs:
        if not isinstance(batch_size, int):
            continue
        if isinstance(num_prompts, int):
            effective = (
                num_prompts if request_count is None
                else min(num_prompts, request_count)
            )
            source = (
                f"min(`--num-prompts`={num_prompts}, `--request-count`={request_count}) "
                f"= {effective}"
                if effective != num_prompts
                else f"`--num-prompts`={num_prompts}"
            )
        else:
            # 未指定 --num-prompts 时默认对全量数据集评测
            effective = request_count
            source = f"all {effective} dataset samples (no `--num-prompts`)"
        if isinstance(effective, int) and batch_size > effective:
            warnings.append(
                f"`--batch-size`={batch_size} is greater than the effective request count "
                f"({source}): the concurrency window cannot be filled and the actual "
                f"concurrency never exceeds {effective}, so the concurrency gain measured "
                f"at this level is underestimated. Keep `--num-prompts` (or "
                f"`--request-count`) not smaller than `--batch-size`."
            )
    return pairs, warnings


def _config_type_name(cfg: Optional[dict]) -> str:
    if not isinstance(cfg, dict):
        return ""
    return get_config_type(cfg.get("type")) or ""


def _is_synthetic_dataset(dataset_cfg: dict) -> bool:
    return _config_type_name(dataset_cfg).endswith("SyntheticDataset")


def _is_dataset_need_models(dataset_cfg: dict) -> bool:
    return _config_type_name(dataset_cfg) in DATASETS_NEED_MODELS


def _synthetic_config(dataset_cfg: dict) -> Optional[dict]:
    synthetic_cfg = dataset_cfg.get("config")
    if isinstance(synthetic_cfg, dict):
        return synthetic_cfg
    return None


def _dataset_request_count(datasets: List[dict], cli_args: dict) -> Optional[int]:
    cli_value = cli_args.get("request_count")
    if cli_value is not None:
        return cli_value
    for dataset_cfg in datasets:
        synthetic_cfg = _synthetic_config(dataset_cfg)
        if synthetic_cfg is not None and synthetic_cfg.get("RequestCount") is not None:
            return synthetic_cfg.get("RequestCount")
    return None


def _apply_model_sweep_overrides(model_cfg: dict, cli_args: dict) -> None:
    """新增模型侧字段覆盖：不要求配置已存在该键。"""
    if cli_args.get("model_abbr") is not None:
        model_cfg["abbr"] = cli_args["model_abbr"]
    if cli_args.get("stream") is not None:
        model_cfg["stream"] = cli_args["stream"]
    if cli_args.get("use_timestamp") is not None:
        model_cfg["use_timestamp"] = cli_args["use_timestamp"]


def _apply_dataset_sweep_overrides(dataset_cfg: dict, cli_args: dict) -> None:
    """新增数据集侧字段覆盖（均为单值，1:1 映射配置字段）。"""
    if cli_args.get("dataset_abbr") is not None:
        dataset_cfg["abbr"] = cli_args["dataset_abbr"]

    synthetic_cfg = _synthetic_config(dataset_cfg)
    if synthetic_cfg is None:
        return

    if cli_args.get("request_count") is not None:
        synthetic_cfg["RequestCount"] = cli_args["request_count"]

    config_type = str(synthetic_cfg.get("Type", "")).lower()
    if config_type == "tokenid":
        tokenid_cfg = synthetic_cfg.get("TokenIdConfig")
        if not isinstance(tokenid_cfg, dict):
            tokenid_cfg = {}
            synthetic_cfg["TokenIdConfig"] = tokenid_cfg
        if cli_args.get("tokenid_config_request_size") is not None:
            tokenid_cfg["RequestSize"] = cli_args["tokenid_config_request_size"]
        if cli_args.get("tokenid_config_prefix_len") is not None:
            tokenid_cfg["PrefixLen"] = cli_args["tokenid_config_prefix_len"]
    elif config_type == "string":
        string_cfg = synthetic_cfg.get("StringConfig")
        if not isinstance(string_cfg, dict):
            string_cfg = {}
            synthetic_cfg["StringConfig"] = string_cfg
        if cli_args.get("string_config_input") is not None:
            string_cfg["Input"] = cli_args["string_config_input"]
        if cli_args.get("string_config_output") is not None:
            string_cfg["Output"] = cli_args["string_config_output"]


def _check_abbr_unique(items: List[dict], label: str, errors: List[str]) -> None:
    seen = {}
    for index, item in enumerate(items):
        abbr = item.get("abbr") if isinstance(item, dict) else None
        if abbr is None:
            continue
        if abbr in seen:
            errors.append(
                f"Duplicate {label} abbr '{abbr}': entry #{seen[abbr] + 1} conflicts "
                f"with entry #{index + 1}"
            )
        else:
            seen[abbr] = index


def validate_config_fields(models: List[dict], datasets: List[dict], cli_args: dict,
                           pairing_warnings: Optional[List[str]] = None) -> List[str]:
    """启动前校验 V01~V09（V01 的配对结果由调用方传入）。

    业务错误一次性汇总后抛 AISBenchConfigError；WARNING 项返回给调用方输出后继续。
    """
    errors: List[str] = []
    warnings: List[str] = list(pairing_warnings or [])
    mode = cli_args.get("mode")

    # V02：batch_size / num_prompts 逐元素范围
    for arg_name, value in (("--batch-size", cli_args.get("batch_size")),
                            ("--num-prompts", cli_args.get("num_prompts"))):
        if isinstance(value, list):
            for index, item in enumerate(value, start=1):
                if not isinstance(item, int) or isinstance(item, bool):
                    errors.append(f"`{arg_name}` value #{index} {item!r} is not an integer")
                elif item < 1:
                    errors.append(f"`{arg_name}` value #{index} {item} is invalid (must be >= 1)")
                elif arg_name == "--batch-size" and item > MAX_BATCH_SIZE:
                    errors.append(
                        f"`{arg_name}` value #{index} {item} is invalid "
                        f"(must be <= {MAX_BATCH_SIZE})"
                    )
        elif value is not None:
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"`{arg_name}` value {value!r} is not an integer")
            elif value < 1:
                errors.append(f"`{arg_name}` value {value} is invalid (must be >= 1)")
            elif arg_name == "--batch-size" and value > MAX_BATCH_SIZE:
                errors.append(
                    f"`{arg_name}` value {value} is invalid (must be <= {MAX_BATCH_SIZE})"
                )

    request_count = cli_args.get("request_count")
    if request_count is not None and not (1 <= request_count <= MAX_REQUEST_COUNT):
        errors.append(
            f"`--request-count` value {request_count} is invalid "
            f"(must satisfy 1 <= v <= {MAX_REQUEST_COUNT})"
        )
    request_size_arg = cli_args.get("tokenid_config_request_size")
    if request_size_arg is not None and not (1 <= request_size_arg <= MAX_REQUEST_SIZE):
        errors.append(
            f"`--tokenid-config-request-size` value {request_size_arg} is invalid "
            f"(must satisfy 1 <= v <= {MAX_REQUEST_SIZE})"
        )
    prefix_len_arg = cli_args.get("tokenid_config_prefix_len")
    # V08：必须是整数 token 数
    if prefix_len_arg is not None:
        if not isinstance(prefix_len_arg, int) or isinstance(prefix_len_arg, bool):
            errors.append(
                f"`--tokenid-config-prefix-len` must be an integer token count, "
                f"got {prefix_len_arg!r}; fractions/ratios are not supported, use an "
                f"absolute token count (e.g. --tokenid-config-prefix-len 6554)"
            )
        elif prefix_len_arg < 0:
            errors.append(
                f"`--tokenid-config-prefix-len` value {prefix_len_arg} is invalid (must be >= 0)"
            )

    # V03：abbr 全局唯一（模型、数据集分别校验）
    _check_abbr_unique(models, "model", errors)
    _check_abbr_unique(datasets, "dataset", errors)

    synthetic_datasets = [d for d in datasets if _is_synthetic_dataset(d)]
    need_model_path = any(_is_dataset_need_models(d) for d in datasets)

    # V04：path 非空且目录存在
    if need_model_path:
        for model_cfg in models:
            path = model_cfg.get("path")
            if not path:
                errors.append(
                    "The dataset requires a model path, `--path` must not be empty "
                    "(replaces runtime UTILS-CFG-001)"
                )
                break
            if not os.path.isdir(os.path.expanduser(str(path))):
                errors.append(
                    f"`--path` directory does not exist: {path} "
                    "(replaces runtime 'ModelPath does not exist')"
                )
                break

    # V07：一次运行只支持一个随机合成数据集
    if len(synthetic_datasets) > 1:
        errors.append(
            f"Only one random synthetic dataset is supported per run, but "
            f"{len(synthetic_datasets)} were detected (multiple ones are merged into a "
            f"single task by _merge_datasets and the results become indistinguishable; "
            f"use an outer loop for multiple specs)"
        )

    tokenid_cli_used = any(
        cli_args.get(key) is not None
        for key in ("tokenid_config_request_size", "tokenid_config_prefix_len")
    )
    string_cli_used = any(
        cli_args.get(key) is not None
        for key in ("string_config_input", "string_config_output")
    )
    for dataset_cfg in synthetic_datasets:
        synthetic_cfg = _synthetic_config(dataset_cfg) or {}
        config_type = str(synthetic_cfg.get("Type", "")).lower()
        # V09：参数与 config.Type 匹配（tokenid 组 / string 组不交叉）
        if tokenid_cli_used and config_type != "tokenid":
            errors.append(
                "`--tokenid-config-request-size`/`--tokenid-config-prefix-len` only apply "
                f"to datasets with config.Type=tokenid, current Type={synthetic_cfg.get('Type')!r}"
            )
        if string_cli_used and config_type != "string":
            errors.append(
                "`--string-config-input`/`--string-config-output` only apply to datasets "
                f"with config.Type=string, current Type={synthetic_cfg.get('Type')!r}"
            )
        if config_type == "tokenid":
            tokenid_cfg = synthetic_cfg.get("TokenIdConfig")
            if isinstance(tokenid_cfg, dict):
                effective_request_size = tokenid_cfg.get("RequestSize")
                effective_prefix_len = tokenid_cfg.get("PrefixLen", 0)
                # V05：PrefixLen ≤ RequestSize
                if (isinstance(effective_request_size, int)
                        and isinstance(effective_prefix_len, int)
                        and effective_prefix_len > effective_request_size):
                    errors.append(
                        f"Dataset PrefixLen={effective_prefix_len} is greater than "
                        f"RequestSize={effective_request_size}; valid range is "
                        f"0 <= PrefixLen <= RequestSize"
                    )
        elif config_type == "string":
            # V09：复用 SyntheticDataset 的 string 子配置校验
            try:
                from ais_bench.benchmark.datasets.synthetic import SyntheticDataset
                string_cfg = synthetic_cfg.get("StringConfig") or {}
                SyntheticDataset.check_synthetic_string_config(
                    {
                        "Input": string_cfg.get("Input"),
                        "Output": string_cfg.get("Output"),
                    }
                )
            except Exception as exc:
                errors.append(
                    f"Invalid StringConfig: {exc}; expected "
                    '{"Input": {"Method": ..., "Params": {...}}, '
                    '"Output": {"Method": ..., "Params": {...}}}'
                )
        # V09：随机合成数据集无 timestamp 字段
        if cli_args.get("use_timestamp") is True:
            errors.append(
                "`--use-timestamp` cannot be used with a random synthetic dataset: it has "
                "no `timestamp` field and would raise NO_TIMESTAMPS_ERROR at runtime; "
                "use `--request-rate` instead"
            )
            break

    # V02：num_prompts > request_count（样本池不足）→ WARNING
    effective_request_count = _dataset_request_count(datasets, cli_args)
    for num_prompts in _as_list(cli_args.get("num_prompts")):
        if (isinstance(num_prompts, int) and isinstance(effective_request_count, int)
                and num_prompts > effective_request_count):
            warnings.append(
                f"`--num-prompts`={num_prompts} is greater than `--request-count`="
                f"{effective_request_count}; the actual request count falls back to "
                f"{effective_request_count} and the concurrency window may not be filled "
                "(V01 warns about it as well). Increase `--request-count` accordingly."
            )

    # V06：stream=False + perf → WARNING 并标注 ttft_valid=false
    if mode == "perf":
        for model_cfg in models:
            if not model_cfg.get("stream", False):
                warnings.append(
                    f"Model [{model_cfg.get('abbr')}] is non-streaming (stream=false), "
                    "TTFT/TPOT cannot be measured; this combo is marked ttft_valid=false"
                )
                break

    # 附加：--use-timestamp 与多并发档语义冲突
    batch_size_arg = cli_args.get("batch_size")
    if cli_args.get("use_timestamp") is True and isinstance(batch_size_arg, list) \
            and len(batch_size_arg) > 1:
        warnings.append(
            "`--use-timestamp` is enabled but `--batch-size` provides multiple "
            "concurrency levels: with timestamp scheduling batch_size no longer limits "
            "in-flight requests, so the swept levels are not comparable"
        )

    if errors:
        raise AISBenchConfigError(
            TMAN_CODES.INVALID_ARG_VALUE_IN_CMD,
            "Sweep argument validation failed:\n" + "\n".join(f"  - {error}" for error in errors),
        )
    return warnings


def _as_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _append_batch_suffix(abbr: str, batch_size: int) -> str:
    """幂等追加 `-bs{N}` 后缀。"""
    suffix = f"-bs{batch_size}"
    if abbr.endswith(suffix):
        return abbr
    return abbr + suffix


def _manifest_cli_args(cli_args: dict) -> dict:
    """manifest 中记录的用户原始参数（脱敏，不含 api_key/url/path）。"""
    mapping = {
        "max_out_len": "max_out_len",
        "num_warmups": "num_warmups",
        "generation_kwargs": "generation_kwargs",
        "batch_size": "batch_size",
        "num_prompts": "num_prompts",
        "request_count": "request_count",
        "request_size": "tokenid_config_request_size",
        "prefix_len": "tokenid_config_prefix_len",
        "string_config_input": "string_config_input",
        "string_config_output": "string_config_output",
        "model_abbr": "model_abbr",
        "dataset_abbr": "dataset_abbr",
        "stream": "stream",
        "use_timestamp": "use_timestamp",
    }
    result = {}
    for manifest_key, cli_key in mapping.items():
        value = cli_args.get(cli_key)
        if value is not None:
            result[manifest_key] = value
    return result


def _build_manifest(specs: List[CombinationSpec], cli_args: dict) -> dict:
    """组合参数快照（一次写入不再修改，不含敏感字段）。"""
    combos = []
    for spec in specs:
        combos.append(
            {
                "index": spec.index,
                "model_abbr": spec.model_abbr,
                "dataset_abbr": spec.dataset_abbr,
                "batch_size": spec.batch_size,
                "num_prompts": spec.num_prompts,
                "request_count": spec.request_count,
                "model_config": {
                    "abbr": spec.base_model_abbr,
                    "model_name": spec.model_name,
                    "stream": spec.stream,
                    "ttft_valid": spec.ttft_valid,
                    "use_timestamp": spec.use_timestamp,
                    "max_out_len": spec.max_out_len,
                },
                "dataset_config": {
                    "Type": spec.dataset_type,
                    "RequestSize": spec.request_size,
                    "PrefixLen": spec.prefix_len,
                },
            }
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "ais_bench_version": __version__,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cli_args": _manifest_cli_args(cli_args),
        "combos": combos,
    }


def expand_sweep_combinations(models: List[dict], datasets: List[dict],
                              cli_args: dict) -> SweepResult:
    """组合展开（详见设计 4.4）。

    - 长度 1 列表先归一为 int，单值路径只做新增字段覆盖，不产生副本/manifest/冷却字段
    - 多档时按组合深拷贝模型与数据集，模型 abbr 追加 `-bs{N}` 保证逐组合唯一
    - 校验 V01~V09，业务错误一次性汇总后抛错，WARNING 汇总输出后继续
    """
    if not models or not datasets:
        return SweepResult(models=list(models), datasets=list(datasets))

    batch_arg = _normalize_sweep_arg(cli_args.get("batch_size"))
    num_arg = _normalize_sweep_arg(cli_args.get("num_prompts"))

    test_range_set = any(
        isinstance(dataset_cfg.get("reader_cfg"), dict)
        and "test_range" in dataset_cfg["reader_cfg"]
        for dataset_cfg in datasets
    )
    request_count = _dataset_request_count(datasets, cli_args)
    pairs, pairing_warnings = validate_pairing(
        batch_arg, num_arg,
        request_count=request_count,
        test_range_set=test_range_set,
    )
    if test_range_set and num_arg is not None:
        pairing_warnings.append(
            "`test_range` has been set, `--num-prompts` will be ignored "
            "(the config test_range is kept and manifest num_prompts is recorded as null)"
        )

    base_models = [copy.deepcopy(model_cfg) for model_cfg in models]
    base_datasets = [copy.deepcopy(dataset_cfg) for dataset_cfg in datasets]
    for model_cfg in base_models:
        _apply_model_sweep_overrides(model_cfg, cli_args)
    for dataset_cfg in base_datasets:
        _apply_dataset_sweep_overrides(dataset_cfg, cli_args)

    warnings = validate_config_fields(
        base_models, base_datasets, cli_args,
        pairing_warnings=pairing_warnings,
    )

    is_multi = len(pairs) > 1

    specs: List[CombinationSpec] = []
    combo_models: List[dict] = []
    combo_datasets: List[dict] = []
    for batch_size, num_prompts in pairs:
        for model_cfg in base_models:
            for dataset_cfg in base_datasets:
                model_copy = copy.deepcopy(model_cfg)
                if batch_size is not None:
                    model_copy["batch_size"] = batch_size
                base_model_abbr = model_copy.get("abbr")
                if is_multi and base_model_abbr:
                    model_copy["abbr"] = _append_batch_suffix(base_model_abbr, batch_size)

                dataset_copy = copy.deepcopy(dataset_cfg)
                if is_multi and num_prompts is not None and not test_range_set:
                    reader_cfg = dataset_copy.get("reader_cfg")
                    if not isinstance(reader_cfg, dict):
                        reader_cfg = {}
                        dataset_copy["reader_cfg"] = reader_cfg
                    reader_cfg["test_range"] = f"[:{num_prompts}]"
                if _is_dataset_need_models(dataset_copy):
                    dataset_copy["model_path"] = model_copy.get("path")
                    dataset_copy["trust_remote_code"] = model_copy.get(
                        "trust_remote_code", False
                    )

                synthetic_cfg = _synthetic_config(dataset_copy) or {}
                tokenid_cfg = synthetic_cfg.get("TokenIdConfig") or {}
                effective_batch_size = model_copy.get("batch_size", 1)
                # manifest / 报告记录生效请求数：未指定 --num-prompts 时默认对全量
                # 数据集评测，取数据集全部样本；指定时受样本池上限约束
                if test_range_set:
                    effective_num_prompts = None
                elif isinstance(num_prompts, int):
                    effective_num_prompts = (
                        num_prompts if request_count is None
                        else min(num_prompts, request_count)
                    )
                else:
                    effective_num_prompts = request_count

                stream = bool(model_copy.get("stream", False))
                specs.append(
                    CombinationSpec(
                        index=len(specs),
                        batch_size=effective_batch_size,
                        num_prompts=effective_num_prompts,
                        request_count=request_count,
                        request_size=tokenid_cfg.get("RequestSize"),
                        prefix_len=tokenid_cfg.get("PrefixLen"),
                        model_abbr=model_copy.get("abbr"),
                        dataset_abbr=dataset_copy.get("abbr"),
                        model_name=model_copy.get("model")
                        or model_copy.get("model_name"),
                        base_model_abbr=base_model_abbr,
                        dataset_type=synthetic_cfg.get("Type"),
                        stream=stream,
                        ttft_valid=stream,
                        use_timestamp=bool(model_copy.get("use_timestamp", False)),
                        max_out_len=model_copy.get("max_out_len"),
                    )
                )
                combo_models.append(model_copy)
                combo_datasets.append(dataset_copy)

    # 展开后仍需保证每个组合的输出路径唯一：同一 (模型 abbr, 数据集 abbr) 只允许
    # 出现一次，否则 performances/<model>/<dataset>.json 会互相覆盖。
    # 多模型/多数据集笛卡尔积下同一 abbr 会在不同组合中重复出现，属正常情况。
    post_errors: List[str] = []
    seen_outputs = {}
    for combo_index, (model_copy, dataset_copy) in enumerate(
            zip(combo_models, combo_datasets)):
        output_key = (model_copy.get("abbr"), dataset_copy.get("abbr"))
        if output_key in seen_outputs:
            post_errors.append(
                f"Duplicate combo output path: model abbr '{output_key[0]}' and dataset "
                f"abbr '{output_key[1]}' appear in both combo "
                f"#{seen_outputs[output_key] + 1} and combo #{combo_index + 1}"
            )
        else:
            seen_outputs[output_key] = combo_index
    if post_errors:
        raise AISBenchConfigError(
            TMAN_CODES.INVALID_ARG_VALUE_IN_CMD,
            "Sweep combo abbr conflict after expansion:\n"
            + "\n".join(f"  - {error}" for error in post_errors),
        )

    for warning in warnings:
        logger.warning(warning)

    if is_multi and cli_args.get("max_num_workers", 1) > 1 and cli_args.get("mode") == "perf":
        logger.warning(
            "Detected %d sweep combos with --max-num-workers=%s > 1: multiple tasks will "
            "hit the same service concurrently and the perf data may pollute each other; "
            "keep it serial (--max-num-workers 1)",
            len(specs),
            cli_args.get("max_num_workers"),
        )

    if not is_multi:
        # 单值等价路径：不产生 model_dataset_combinations / manifest / 冷却字段
        return SweepResult(
            models=combo_models,
            datasets=combo_datasets,
            combinations=[],
            manifest=None,
            sweep_cooldown_s=0,
        )

    combinations = [
        {"models": [model_copy], "datasets": [dataset_copy]}
        for model_copy, dataset_copy in zip(combo_models, combo_datasets)
    ]
    return SweepResult(
        models=combo_models,
        datasets=combo_datasets,
        combinations=combinations,
        manifest=_build_manifest(specs, cli_args),
        sweep_cooldown_s=SWEEP_COOLDOWN_SECONDS,
    )
