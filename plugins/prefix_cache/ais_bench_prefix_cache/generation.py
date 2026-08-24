from __future__ import annotations

import csv
import hashlib
import itertools
import json
import logging
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from .errors import ArtifactValidationError, PromptRoundTripError, ScenarioValidationError

logger = logging.getLogger(__name__)


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str: ...


@dataclass(frozen=True)
class GSMRecord:
    """语料（GSM8K）中的一条记录。"""
    line_index: int        # 源文件行号
    question: str          # 规范化后的题干文本
    question_sha256: str   # 题干的 SHA-256，用于去重/溯源


@dataclass(frozen=True)
class CanonicalPrefix:
    """某个 Prefix Group 的规范化共享前缀（由语料重复拼接并截断到所需长度）。"""
    group_id: str
    text: str                      # 前缀文本
    token_ids: tuple[int, ...]     # 前缀 token 序列
    sha256: str                    # 前缀指纹
    gsm_indices: tuple[int, ...]   # 来源语料行号
    gsm_hashes: tuple[str, ...]    # 来源语料 hash


@dataclass(frozen=True)
class RequestPlan:
    """一条请求的完整生成计划（构造 prompt 与模拟命中率的载体）。"""
    request_id: str
    sequence_index: int            # 全局序号
    group_id: str                  # 所属 Prefix Group
    occurrence_index_within_group: int  # 组内出现次序
    dp_rank: int | None            # cold 模式下路由到的 DP 卡
    lane_sequence: int | None      # 该 lane 内的序号（用于串行放行）
    target_input_tokens: int
    actual_input_tokens: int       # 实际构造出的输入 token 数
    max_tokens: int
    shared_prefix_tokens: int      # 共享前缀长度（求解器的核心决策量）
    seed_tokens: int               # 唯一 seed 块长度
    natural_suffix_tokens: int     # 自然后缀长度
    question: str = ""
    answer: str = "none"
    gsm_indices: tuple[int, ...] = ()
    gsm_hashes: tuple[str, ...] = ()
    canonical_prefix_sha256: str = ""
    seed_sha256: str = ""
    request_random_seed: int = 0
    watermark_before: int = 0      # 模拟：请求到达前缓存水位
    theoretical_hit_tokens: int = 0  # 模拟：理论命中 token 数
    watermark_after: int = 0       # 模拟：请求后缓存水位

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TheorySummary:
    """缓存水位模拟的汇总结果。"""
    rows: tuple[RequestPlan, ...]
    total_input_tokens: int
    total_hit_tokens: int
    global_hit_rate: float
    group_stats: dict[str, dict[str, float | int]]
    dp_stats: dict[int, dict[str, float | int]]


@dataclass(frozen=True)
class SolveResult:
    """共享前缀长度求解的结果与可达性/偏差诊断。"""
    shared_prefix_tokens: tuple[int, ...]   # 每条请求的共享前缀长度（核心产物）
    requested_hit_tokens: int               # 目标命中 token 数
    effective_hit_tokens: int               # 实际达到的命中 token 数
    effective_hit_rate: float
    min_reachable_rate: float               # 理论最低可达命中率（全不共享）
    max_reachable_rate: float               # 理论最高可达命中率（全共享）
    target_reachable: bool                  # 目标命中率是否落在可达区间
    group_reachability: dict[str, dict[str, float]]  # 各组的可达区间
    adjusted: bool                          # 是否因约束无法精确命中目标
    reason: str | None                      # 无法精确命中的原因


def normalize_question(value: str) -> str:
    """规范化题干：合并空白、去首尾空格。"""
    return " ".join(value.strip().split())


def load_gsm8k(path: Path, field: str = "question") -> list[GSMRecord]:
    """逐行读取 GSM8K JSONL 语料并构造 GSMRecord 列表，校验空内容。"""
    logger.info("[gen] load_gsm8k path=%s field=%s", path, field)
    records: list[GSMRecord] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ScenarioValidationError(f"cannot read GSM8K {path}: {exc}") from exc
    logger.info("[gen] load_gsm8k lines=%d", len(lines))
    for line_index, line in enumerate(lines):
        try:
            raw = json.loads(line)
            question = normalize_question(raw[field])
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            raise ScenarioValidationError(f"GSM8K line {line_index} is invalid: {exc}") from exc
        if not question:
            raise ScenarioValidationError(f"GSM8K line {line_index} has empty {field}")
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
        records.append(GSMRecord(line_index, question, digest))
    if not records:
        raise ScenarioValidationError("GSM8K corpus is empty")
    logger.info("[gen] load_gsm8k records=%d first_line_index=%d first_sha256=%s", len(records), records[0].line_index, records[0].question_sha256)
    return records


def select_gsm8k(records: Sequence[GSMRecord], config: dict[str, Any], count: int, seed: int) -> list[GSMRecord]:
    """按 selection 配置（random/indices/question_sha256/mixed）选出 count 条语料。"""
    logger.info("[gen] select_gsm8k mode=%s config=%s count=%d seed=%d", config["mode"], config, count, seed)
    mode = config["mode"]
    by_index = {item.line_index: item for item in records}
    by_hash: dict[str, list[GSMRecord]] = {}
    for item in records:
        by_hash.setdefault(item.question_sha256, []).append(item)
    if mode == "random":
        rng = random.Random(seed)
        selected: list[GSMRecord] = []
        while len(selected) < count:
            cycle = list(records)
            rng.shuffle(cycle)
            selected.extend(cycle)
        result = selected[:count]
        logger.info("[gen] select_gsm8k mode=random selected=%d line_indices=%s", len(result), [item.line_index for item in result])
        return result
    values = config.get("values")
    if values is None:
        values = config.get("indices" if mode == "indices" else "question_sha256", [])
    logger.info("[gen] select_gsm8k values=%s", values)
    if mode == "indices":
        try:
            selected = [by_index[int(value)] for value in values]
        except (KeyError, ValueError, TypeError) as exc:
            raise ScenarioValidationError(f"specified GSM8K index does not exist: {exc}") from exc
    elif mode == "question_sha256":
        selected = []
        for value in values:
            matches = by_hash.get(str(value), [])
            if len(matches) != 1:
                raise ScenarioValidationError(f"GSM8K hash must resolve uniquely: {value}")
            selected.append(matches[0])
    else:
        # mixed 模式：合并 indices 与 question_sha256 两部分的选择结果。
        selected = []
        index_values = config.get("indices", [])
        hash_values = config.get("question_sha256", [])
        if index_values:
            selected.extend(select_gsm8k(records, {"mode": "indices", "values": index_values}, len(index_values), seed))
        if hash_values:
            selected.extend(select_gsm8k(records, {"mode": "question_sha256", "values": hash_values}, len(hash_values), seed))
    if not selected:
        raise ScenarioValidationError("specified GSM8K selection is empty")
    # 不足 count 时循环复用已选中的记录。
    result = [selected[i % len(selected)] for i in range(count)]
    logger.info("[gen] select_gsm8k selected=%d line_indices=%s", len(result), [item.line_index for item in result])
    return result


def _csv_values(path: str, aliases: Sequence[str]) -> list[int]:
    """从 CSV 读取指定别名列的整数值（用于显式长度配置）。"""
    logger.info("[gen] _csv_values path=%s aliases=%s", path, list(aliases))
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    fieldnames = rows[0].keys() if rows else []
    column = next((name for name in aliases if name in fieldnames), None)
    if column is None:
        raise ScenarioValidationError(f"CSV requires one of columns {list(aliases)}")
    values = [int(row[column]) for row in rows]
    if not values or any(value < 1 for value in values):
        raise ScenarioValidationError(f"CSV column {column} must contain positive integers")
    logger.info("[gen] _csv_values rows=%d column=%s values=%s", len(rows), column, values)
    return values


def _log_lengths(label: str, values: list[int]) -> list[int]:
    """带日志返回长度序列（供各生成函数复用）。"""
    logger.info(
        "[gen] %s count=%d min=%d max=%d mean=%.2f values=%s",
        label, len(values), min(values), max(values), sum(values) / len(values), values,
    )
    return values


def build_input_lengths(config: dict[str, Any], count: int, seed: int) -> list[int]:
    """按输入长度配置生成 count 条请求的输入长度序列。"""
    mode = config["mode"]
    logger.info("[gen] build_input_lengths mode=%s config=%s count=%d seed=%d", mode, config, count, seed)
    if mode == "fixed":
        return _log_lengths("build_input_lengths", [int(config["value"])] * count)
    if mode == "explicit":
        values = [int(value) for value in config["values"]]
        if len(values) != count:
            raise ScenarioValidationError("explicit input length count must equal requests.count")
        return _log_lengths("build_input_lengths", values)
    if mode == "csv":
        values = _csv_values(config["path"], ("input_prompt_tokens", "content_tokens", "input_tokens"))
        if len(values) != count:
            raise ScenarioValidationError("input CSV row count must equal requests.count")
        return _log_lengths("build_input_lengths", values)
    if mode == "range":
        rng = random.Random(seed)
        return _log_lengths("build_input_lengths", [rng.randint(int(item["min"]), int(item["max"])) for item in config["ranges"] for _ in range(int(item["count"]))])
    return _truncated_normal_values(config, count, seed, "build_input_lengths")


def _truncated_normal_values(config: dict[str, Any], count: int, seed: int, label: str = "_truncated_normal_values") -> list[int]:
    """从截断正态分布采样 count 个整数值（min<=x<=max），失败时抛校验错误。"""
    logger.info("[gen] %s config=%s count=%d seed=%d", label, config, count, seed)
    low, high = int(config["min"]), int(config["max"])
    if low == high:
        return _log_lengths(label, [low] * count)
    mean = float(config.get("mean", (low + high) / 2))
    std = float(config.get("std", max(1.0, (high - low) / 4)))
    logger.info("[gen] %s low=%d high=%d mean=%.2f std=%.2f", label, low, high, mean, std)
    rng = random.Random(seed)
    values: list[int] = []
    attempts = 0
    while len(values) < count and attempts < max(1000, count * 100):
        value = int(round(rng.gauss(mean, std)))
        if low <= value <= high:
            values.append(value)
        attempts += 1
    if len(values) != count:
        raise ScenarioValidationError("truncated_normal could not produce enough values")
    logger.info("[gen] %s attempts=%d produced=%d", label, attempts, len(values))
    return _log_lengths(label, values)


def build_output_lengths(config: dict[str, Any], count: int, seed: int) -> list[int]:
    """按输出长度配置生成 count 条请求的输出（max_tokens）长度序列。"""
    mode = config["mode"]
    logger.info("[gen] build_output_lengths mode=%s config=%s count=%d seed=%d", mode, config, count, seed)
    if mode == "fixed":
        return _log_lengths("build_output_lengths", [int(config["value"])] * count)
    if mode == "csv":
        values = _csv_values(config["path"], ("output_tokens",))
        if len(values) != count:
            raise ScenarioValidationError("output CSV row count must equal requests.count")
        return _log_lengths("build_output_lengths", values)
    low, high = int(config["min"]), int(config["max"])
    rng = random.Random(seed)
    if mode == "uniform":
        return _log_lengths("build_output_lengths", [rng.randint(low, high) for _ in range(count)])
    return _truncated_normal_values(config, count, seed, "build_output_lengths")


def assign_groups(count: int, config: dict[str, Any], seed: int) -> list[str]:
    """按权重把 count 条请求分配到各 Prefix Group，返回每条请求的 group_id。

    支持 uniform / zipf / weights 三种分配模式；用 Largest Remainder 法
    把配额小数部分均摊到各分组，保证总数恰为 count。
    """
    logger.info("[gen] assign_groups count=%d config=%s seed=%d", count, config, seed)
    group_count = int(config["count"])
    assignment = config["assignment"]
    mode = assignment["mode"]
    if mode == "uniform":
        weights = [1.0] * group_count
    elif mode == "zipf":
        exponent = float(assignment.get("exponent", 1.0))
        if exponent <= 0:
            raise ScenarioValidationError("zipf exponent must be positive")
        weights = [1 / ((index + 1) ** exponent) for index in range(group_count)]
    else:
        weights = [float(value) for value in assignment.get("weights", [])]
        if len(weights) != group_count or any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ScenarioValidationError("explicit group weights must match group count and sum positive")
    logger.info("[gen] assign_groups group_count=%d mode=%s weights=%s", group_count, mode, weights)
    total = sum(weights)
    quotas = [count * value / total for value in weights]
    allocations = [math.floor(value) for value in quotas]
    remaining = count - sum(allocations)
    logger.info("[gen] assign_groups quotas=%s allocations=%s remaining=%d", quotas, allocations, remaining)
    # 按小数余量从大到小把剩余配额依次补齐到各分组。
    order = sorted(range(group_count), key=lambda index: (-(quotas[index] - allocations[index]), index))
    for index in order[:remaining]:
        allocations[index] += 1
    groups = [f"group-{index}" for index, amount in enumerate(allocations) for _ in range(amount)]
    if mode == "zipf":
        # zipf 模式下打乱顺序，避免长尾集中在头部。
        random.Random(seed).shuffle(groups)
    logger.info("[gen] assign_groups groups=%s distribution=%s", groups, {group: groups.count(group) for group in sorted(set(groups))})
    return groups


def order_indices(group_ids: Sequence[str], strategy: str, seed: int, input_lengths: Sequence[int] | None = None) -> list[int]:
    """生成请求的发送顺序排列，控制 Prefix Cache 命中特性。

    策略：sequential（原序）/ global_shuffle（全局乱序）/ within_group_shuffle
    （组内乱序）/ interleave（组间交错）/ input_len_asc（组内按输入长度升序）。
    """
    logger.info("[gen] order_indices count=%d strategy=%s seed=%d input_lengths_provided=%s", len(group_ids), strategy, seed, input_lengths is not None)
    indices = list(range(len(group_ids)))
    rng = random.Random(seed)
    if strategy == "sequential":
        logger.info("[gen] order_indices strategy=sequential result=%s", indices)
        return indices
    if strategy == "global_shuffle":
        rng.shuffle(indices)
        logger.info("[gen] order_indices strategy=global_shuffle result=%s", indices)
        return indices
    # 按分组把请求装进桶，再按策略重排桶内顺序。
    buckets: dict[str, list[int]] = {}
    for index, group in enumerate(group_ids):
        buckets.setdefault(group, []).append(index)
    logger.info("[gen] order_indices buckets=%s", {group: len(members) for group, members in buckets.items()})
    if strategy == "input_len_asc":
        if input_lengths is None or len(input_lengths) != len(group_ids):
            raise ScenarioValidationError("input_len_asc requires one input length per request")
        for group in buckets:
            buckets[group].sort(key=lambda index: (int(input_lengths[index]), index))
    if strategy == "within_group_shuffle":
        result: list[int] = []
        for group in sorted(buckets):
            rng.shuffle(buckets[group])
            result.extend(buckets[group])
        logger.info("[gen] order_indices strategy=within_group_shuffle result=%s", result)
        return result
    # interleave：用 zip_longest 把各组的请求轮流取出，实现交错。
    result = []
    for row in itertools.zip_longest(*(buckets[group] for group in sorted(buckets))):
        result.extend(index for index in row if index is not None)
    logger.info("[gen] order_indices strategy=interleave result=%s", result)
    return result


def assign_cold_routes(group_ids: Sequence[str], dp_size: int, explicit: Sequence[int] | None = None) -> tuple[list[int], list[int]]:
    """cold 模式：为每条请求分派 DP rank 与 lane 序号。

    默认按组内出现次序轮转到各 DP 卡（group_round_robin）；显式 routes 可覆盖。
    lane 序列用于让同一 (group, rank) 上的请求按序发送。
    """
    logger.info("[gen] assign_cold_routes count=%d dp_size=%d explicit=%s", len(group_ids), dp_size, explicit)
    if explicit is not None:
        if len(explicit) != len(group_ids) or any(rank < 0 or rank >= dp_size for rank in explicit):
            raise ScenarioValidationError("explicit DP routes are invalid")
        ranks = list(explicit)
    else:
        # 每个组内第 k 条请求路由到 rank = k % dp_size。
        seen: dict[str, int] = {}
        ranks = []
        for group in group_ids:
            occurrence = seen.get(group, 0)
            ranks.append(occurrence % dp_size)
            seen[group] = occurrence + 1
    lane_seen: dict[tuple[str, int], int] = {}
    lane_sequences = []
    for group, rank in zip(group_ids, ranks):
        lane = (group, rank)
        lane_sequences.append(lane_seen.get(lane, 0))
        lane_seen[lane] = lane_sequences[-1] + 1
    logger.info("[gen] assign_cold_routes ranks=%s lane_sequences=%s", ranks, lane_sequences)
    return ranks, lane_sequences


def simulate_theory(plans: Sequence[RequestPlan], mode: str, warmup_watermarks: dict[str, int] | None = None, verbose: bool = True) -> TheorySummary:
    """按缓存水位模拟理论命中率。

    对每条请求维护其缓存键（组 或 组×rank）的水位；命中 = min(共享前缀, 既有水位)，
    随后水位提升到该请求的共享前缀长度。最终汇总全局/分组/DP 的命中统计。
    """
    if verbose:
        logger.info("[gen] simulate_theory plans=%d mode=%s warmup_watermarks=%s", len(plans), mode, warmup_watermarks)
    watermarks: dict[object, int] = {}
    if mode == "warmup":
        watermarks.update(warmup_watermarks or {})
    rows: list[RequestPlan] = []
    group_totals: dict[str, list[int]] = {}
    dp_totals: dict[int, list[int]] = {}
    for plan in plans:
        key: object = plan.group_id if mode == "warmup" else (plan.group_id, plan.dp_rank or 0)
        before = watermarks.get(key, 0)
        hit = min(plan.shared_prefix_tokens, before)
        after = max(before, plan.shared_prefix_tokens)
        watermarks[key] = after
        row = replace(plan, watermark_before=before, theoretical_hit_tokens=hit, watermark_after=after)
        rows.append(row)
        if verbose:
            logger.info("[gen] simulate_theory request_id=%s key=%s watermark_before=%d hit=%d watermark_after=%d", plan.request_id, key, before, hit, after)
        group_totals.setdefault(plan.group_id, [0, 0])
        group_totals[plan.group_id][0] += plan.actual_input_tokens
        group_totals[plan.group_id][1] += hit
        if plan.dp_rank is not None:
            dp_totals.setdefault(plan.dp_rank, [0, 0])
            dp_totals[plan.dp_rank][0] += plan.actual_input_tokens
            dp_totals[plan.dp_rank][1] += hit
    total_input = sum(row.actual_input_tokens for row in rows)
    total_hit = sum(row.theoretical_hit_tokens for row in rows)
    global_rate = total_hit / total_input if total_input else 0.0
    stats = lambda values: {"input_tokens": values[0], "hit_tokens": values[1], "hit_rate": values[1] / values[0] if values[0] else 0.0}
    group_stats = {key: stats(value) for key, value in group_totals.items()}
    dp_stats = {key: stats(value) for key, value in dp_totals.items()}
    if verbose:
        logger.info("[gen] simulate_theory total_input_tokens=%d total_hit_tokens=%d global_hit_rate=%.4f", total_input, total_hit, global_rate)
        logger.info("[gen] simulate_theory group_stats=%s dp_stats=%s", group_stats, dp_stats)
    return TheorySummary(tuple(rows), total_input, total_hit, global_rate, group_stats, dp_stats)


def _plans_for_prefixes(input_lengths: Sequence[int], output_lengths: Sequence[int], group_ids: Sequence[str], ranks: Sequence[int | None], lane_sequences: Sequence[int | None], prefixes: Sequence[int]) -> list[RequestPlan]:
    """按给定的一组共享前缀长度构造临时 RequestPlan 列表（供求解器打分用）。"""
    occurrences: dict[str, int] = {}
    plans = []
    for index, (length, out_len, group, rank, lane_seq, prefix) in enumerate(zip(input_lengths, output_lengths, group_ids, ranks, lane_sequences, prefixes)):
        occurrence = occurrences.get(group, 0)
        occurrences[group] = occurrence + 1
        plans.append(RequestPlan(f"request-{index:08d}", index, group, occurrence, rank, lane_seq, length, length, out_len, prefix, 0, length - prefix))
    return plans


def solve_prefix_lengths(input_lengths: Sequence[int], output_lengths: Sequence[int], group_ids: Sequence[str], ranks: Sequence[int | None], lane_sequences: Sequence[int | None], block_size: int, minimum_non_shared_tokens: int, mode: str, target_hit_rate: float) -> SolveResult:
    """求解每条请求的共享前缀长度（shared_prefix_tokens）。

    目标：让整体理论命中率尽量逼近配置的 target_hit_rate。由于共享前缀必须按
    block_size 对齐、且必须为每条请求保留 minimum_non_shared_tokens 的非共享区
    （seed + 自然后缀），再叠加 KV cache 水位约束，目标值不一定能精确达到。
    求解器先计算可达性区间（min/max），再将目标钳制到最近的 Block 对齐命中量：
    warmup 按请求容量分配，cold 按 (Prefix Group, DP rank) lane 线性构造精确解。
    """
    logger.info("[gen] solve_prefix_lengths requests=%d block_size=%d minimum_non_shared_tokens=%d mode=%s target_hit_rate=%.4f", len(input_lengths), block_size, minimum_non_shared_tokens, mode, target_hit_rate)
    logger.info("[gen] solve_prefix_lengths input_lengths=%s output_lengths=%s group_ids=%s ranks=%s lane_sequences=%s", list(input_lengths), list(output_lengths), list(group_ids), list(ranks), list(lane_sequences))
    # 每条请求的共享前缀长度候选集：必须是 block_size 的整数倍（前缀要按 block 对齐
    # 才能命中缓存），且最大只能取到 (length - minimum_non_shared_tokens) 向下对齐的值，
    # 从而保证每条请求都留足非共享区（seed + 自然后缀）。
    candidates = [list(range(0, max(0, ((length - minimum_non_shared_tokens) // block_size) * block_size) + 1, block_size)) for length in input_lengths]
    logger.info("[gen] solve_prefix_lengths candidates=%s", candidates)
    total_input = sum(input_lengths)
    # 把目标命中率换算成目标命中 token 总数（四舍五入），作为搜索的靶心。
    target_tokens = int(total_input * target_hit_rate + 0.5)
    logger.info("[gen] solve_prefix_lengths total_input=%d target_tokens=%d", total_input, target_tokens)

    def score(prefixes: Sequence[int]) -> tuple[int, int]:
        """给出一组共享前缀长度，模拟缓存水位后返回 (命中误差, 命中token数)。"""
        # 按候选前缀构造临时请求计划，并模拟真实缓存水位下的理论命中 token 数。
        plans = _plans_for_prefixes(input_lengths, output_lengths, group_ids, ranks, lane_sequences, prefixes)
        # warmup 模式下用各组前缀最大值作为预热水位，模拟缓存已被写满。
        warm = {group: max((prefix for prefix, current in zip(prefixes, group_ids) if current == group), default=0) for group in set(group_ids)} if mode == "warmup" else None
        hit = simulate_theory(plans, mode, warm, verbose=False).total_hit_tokens
        return abs(hit - target_tokens), hit

    # 先评估可达性上下界。旧实现先做局部搜索、最后才计算边界，导致目标高于
    # reachable_max 时仍可能停在次优局部解。边界现在是求解输入而非事后报告。
    zero_prefixes = [0] * len(candidates)
    zero_plans = _plans_for_prefixes(input_lengths, output_lengths, group_ids, ranks, lane_sequences, zero_prefixes)
    zero_warm = {group: 0 for group in set(group_ids)} if mode == "warmup" else None
    zero_theory = simulate_theory(zero_plans, mode, zero_warm)
    zero_hit = zero_theory.total_hit_tokens
    max_prefixes = [values[-1] for values in candidates]
    max_plans = _plans_for_prefixes(input_lengths, output_lengths, group_ids, ranks, lane_sequences, max_prefixes)
    max_warm = {
        group: max((prefix for prefix, current in zip(max_prefixes, group_ids) if current == group), default=0)
        for group in set(group_ids)
    } if mode == "warmup" else None
    max_theory = simulate_theory(max_plans, mode, max_warm)
    max_hit = max_theory.total_hit_tokens

    # 所有共享前缀和理论命中量都是 block_size 的整数倍。按 Block 单位选取距
    # target_tokens 最近、且落在可达边界内的目标命中量。
    max_hit_units = max_hit // block_size
    lower_units = max(0, min(max_hit_units, target_tokens // block_size))
    upper_units = max(0, min(max_hit_units, lower_units + 1))
    desired_hit_units = min(
        {lower_units, upper_units, 0, max_hit_units},
        key=lambda units: (abs(units * block_size - target_tokens), units),
    )
    desired_hit_tokens = desired_hit_units * block_size

    prefixes = [0] * len(candidates)
    caps = [values[-1] // block_size for values in candidates]
    remaining_units = desired_hit_units
    if mode == "warmup":
        # warmup 后每个前缀 token 都命中；因此把目标 Block 数按请求容量依次分配即可。
        for index, cap in enumerate(caps):
            assigned = min(cap, remaining_units)
            prefixes[index] = assigned * block_size
            remaining_units -= assigned
    else:
        # cold 模式按 (group, DP rank) 独立维护水位。对任一 lane：
        #   lane_hit = sum(prefix_i) - max(prefix_i)
        # 选容量最大的请求作为 anchor；把所需 hit Block 分配给其余请求，再让
        # anchor 等于这些请求中的最大前缀。于是 anchor 的前缀恰好抵消 max 项，
        # lane_hit 精确等于已分配 Block 数。每个 0..lane_max 区间都可构造，无需爬山。
        lanes: dict[tuple[str, int], list[int]] = {}
        for index, (group, rank) in enumerate(zip(group_ids, ranks)):
            lanes.setdefault((group, int(rank or 0)), []).append(index)
        for lane_indices in lanes.values():
            anchor = max(lane_indices, key=lambda index: (caps[index], -index))
            lane_capacity = sum(caps[index] for index in lane_indices if index != anchor)
            lane_units = min(lane_capacity, remaining_units)
            lane_remaining = lane_units
            for index in lane_indices:
                if index == anchor:
                    continue
                assigned = min(caps[index], lane_remaining)
                prefixes[index] = assigned * block_size
                lane_remaining -= assigned
            if lane_remaining:
                raise ArtifactValidationError("cold Prefix Cache lane construction did not consume its target")
            prefixes[anchor] = max((prefixes[index] for index in lane_indices if index != anchor), default=0)
            remaining_units -= lane_units
    if remaining_units:
        raise ArtifactValidationError("Prefix Cache solver could not construct the selected reachable target")

    best_error, best_hit = score(prefixes)
    if best_hit != desired_hit_tokens:
        raise ArtifactValidationError(
            f"Prefix Cache solver constructed {best_hit} hit tokens; expected {desired_hit_tokens}"
        )
    logger.info(
        "[gen] solve_prefix_lengths strategy=exact_lane_construction desired_hit=%d chosen_prefixes=%s best_error=%d best_hit=%d",
        desired_hit_tokens, prefixes, best_error, best_hit,
    )
    # 实际/最低/最高可达的命中率。
    effective_rate = best_hit / total_input if total_input else 0.0
    min_rate = zero_hit / total_input if total_input else 0.0
    max_rate = max_hit / total_input if total_input else 0.0
    logger.info("[gen] solve_prefix_lengths zero_hit=%d zero_rate=%.4f max_hit=%d max_rate=%.4f effective_hit=%d effective_rate=%.4f", zero_hit, min_rate, max_hit, max_rate, best_hit, effective_rate)
    # 按组给出可达命中率区间，便于定位是哪个组导致目标不可达。
    group_reachability = {
        group: {
            "min_reachable_rate": float(zero_theory.group_stats[group]["hit_rate"]),
            "max_reachable_rate": float(max_theory.group_stats[group]["hit_rate"]),
        }
        for group in sorted(set(group_ids))
    }
    # 目标命中率落在 [min_rate, max_rate] 区间内即为可达。
    target_reachable = min_rate <= target_hit_rate <= max_rate
    # 无法精确命中目标时标记 adjusted，并区分目标越界与单纯 Block 对齐残差。
    adjusted = best_hit != target_tokens
    if target_tokens > max_hit:
        reason = "target exceeds maximum reachable hit rate"
    elif target_tokens < zero_hit:
        reason = "target is below minimum reachable hit rate"
    elif adjusted:
        reason = "block alignment prevents an exact target hit count"
    else:
        reason = None
    logger.info("[gen] solve_prefix_lengths group_reachability=%s target_reachable=%s adjusted=%s reason=%s", group_reachability, target_reachable, adjusted, reason)
    return SolveResult(
        tuple(prefixes), target_tokens, best_hit, effective_rate, min_rate, max_rate,
        target_reachable, group_reachability, adjusted,
        reason,
    )


def _safe_token_text(tokenizer: TokenizerLike, token_id: int, special: set[int]) -> str | None:
    """判定某个 token 是否是"边界安全"的：单独 decode 且前后加字都不改变编码。"""
    if token_id in special:
        return None
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    if not text:
        return None
    # 单独编回、左侧拼接、右侧拼接都必须保持该 token 不变。
    if tokenizer.encode(text, add_special_tokens=False) != [token_id]:
        return None
    if tokenizer.encode("X" + text, add_special_tokens=False)[-1:] != [token_id]:
        return None
    if tokenizer.encode(text + "X", add_special_tokens=False)[:1] != [token_id]:
        return None
    return text


def find_boundary_safe_token_ids(tokenizer: TokenizerLike, minimum: int) -> list[int]:
    # Prefer space-prefixed tokens: in BPE tokenizers they cannot merge with
    # preceding text, so seeds built from them stay stable at every junction.
    vocab_size = len(tokenizer)  # type: ignore[arg-type]
    logger.info("[gen] find_boundary_safe_token_ids minimum=%d vocab_size=%d", minimum, vocab_size)
    special = set(getattr(tokenizer, "all_special_ids", []))
    preferred: list[int] = []
    fallback: list[int] = []
    for token_id in range(vocab_size):
        text = _safe_token_text(tokenizer, token_id, special)
        if text is None:
            continue
        # 优先收集空格开头的 token（BPE 下与前置文本不会合并，seed 更稳定）。
        if text.startswith(" "):
            preferred.append(token_id)
            if len(preferred) >= minimum:
                logger.info("[gen] find_boundary_safe_token_ids preferred=%s", preferred)
                return preferred
        else:
            fallback.append(token_id)
    combined = preferred + fallback
    if len(combined) < minimum:
        raise ArtifactValidationError(f"tokenizer has only {len(combined)} boundary-safe tokens; need {minimum}")
    result = combined[:minimum]
    logger.info("[gen] find_boundary_safe_token_ids preferred=%d fallback=%d result=%s", len(preferred), len(fallback), result)
    return result


def _seed_round_trips(tokenizer: TokenizerLike, seed: Sequence[int]) -> bool:
    """校验 seed token 序列 decode 后再 encode 能原样恢复（round-trip 安全）。"""
    text = tokenizer.decode(seed, skip_special_tokens=False)
    return tokenizer.encode(text, add_special_tokens=False) == list(seed)


def build_unique_seed(tokenizer: TokenizerLike | None, safe_ids: Sequence[int], request_id: str, seed_length: int, random_seed: int, exclude: set[tuple[int, ...]] | None = None) -> tuple[int, ...]:
    """构造一个全局唯一且 round-trip 安全的 seed token 序列（长度 seed_length）。

    用 SHA-256 派生的字节流从 safe_ids 中抽样；若与已用 seed 重复或无法
    round-trip，则换 nonce 重试。
    """
    logger.info("[gen] build_unique_seed request_id=%s seed_length=%d random_seed=%d safe_ids=%d exclude=%d", request_id, seed_length, random_seed, len(safe_ids), len(exclude) if exclude else 0)
    if seed_length < 1 or len(safe_ids) < 2:
        raise ArtifactValidationError("seed generation requires positive length and at least two safe tokens")
    used = exclude if exclude is not None else set()
    for nonce in range(4096):
        digest = hashlib.sha256(f"{random_seed}:{request_id}:{nonce}".encode()).digest()
        stream = itertools.cycle(digest)
        seed = tuple(safe_ids[next(stream) % len(safe_ids)] for _ in range(seed_length))
        if seed in used:
            logger.info("[gen] build_unique_seed retry request_id=%s nonce=%d reason=duplicate_seed", request_id, nonce)
            continue
        if tokenizer is not None and not _seed_round_trips(tokenizer, seed):
            logger.info("[gen] build_unique_seed retry request_id=%s nonce=%d reason=round_trip_failure", request_id, nonce)
            continue
        logger.info("[gen] build_unique_seed request_id=%s nonce=%d seed=%s", request_id, nonce, seed)
        return seed
    raise ArtifactValidationError(f"unable to construct a unique round-trip-safe seed for {request_id}")


def build_unique_seed_tokens(safe_ids: Sequence[int], request_ids: Sequence[str], seed_length: int, random_seed: int, tokenizer: TokenizerLike | None = None) -> dict[str, tuple[int, ...]]:
    """为一批 request_id 批量构造互不重复的唯一 seed，返回 {request_id: seed}。"""
    logger.info("[gen] build_unique_seed_tokens request_ids=%d seed_length=%d random_seed=%d", len(request_ids), seed_length, random_seed)
    result: dict[str, tuple[int, ...]] = {}
    used: set[tuple[int, ...]] = set()
    for request_id in request_ids:
        seed = build_unique_seed(tokenizer, safe_ids, request_id, seed_length, random_seed, used)
        used.add(seed)
        result[request_id] = seed
    logger.info("[gen] build_unique_seed_tokens result keys=%d", len(result))
    return result


def _repeat_tokens(records: Sequence[GSMRecord], tokenizer: TokenizerLike, target: int) -> tuple[list[int], tuple[int, ...], tuple[str, ...]]:
    """循环拼接语料记录直至 token 数达到 target，返回 tokens 及来源索引/hash。"""
    logger.info("[gen] _repeat_tokens records=%d target=%d", len(records), target)
    tokens: list[int] = []
    indices: list[int] = []
    hashes: list[str] = []
    for record in itertools.cycle(records):
        piece = tokenizer.encode((" " if tokens else "") + record.question, add_special_tokens=False)
        if not piece:
            continue
        tokens.extend(piece)
        indices.append(record.line_index)
        hashes.append(record.question_sha256)
        if len(tokens) >= target:
            logger.info("[gen] _repeat_tokens result tokens=%d indices=%d hashes=%d", len(tokens[:target]), len(indices), len(hashes))
            return tokens[:target], tuple(indices), tuple(hashes)
    raise ArtifactValidationError("cannot build tokens from empty GSM8K records")


def build_canonical_prefixes(tokenizer: TokenizerLike, group_sources: dict[str, Sequence[GSMRecord]], max_lengths: dict[str, int], block_size: int) -> dict[str, CanonicalPrefix]:
    """为每个 Prefix Group 构造 canonical 前缀（语料重复拼接至组内最大共享长度）。

    各组的首个 block 必须互不相同，否则无法区分组；冲突时用确定性组标记兜底。
    """
    logger.info("[gen] build_canonical_prefixes groups=%s max_lengths=%s block_size=%d", sorted(group_sources), max_lengths, block_size)
    result: dict[str, CanonicalPrefix] = {}
    first_blocks: set[tuple[int, ...]] = set()
    for group_position, group in enumerate(sorted(group_sources)):
        source_records = list(group_sources[group])
        if not source_records:
            raise ArtifactValidationError(f"canonical prefix source is empty for {group}")
        token_ids = indices = hashes = None
        # 尝试不同旋转起点，找到首个 block 不与其他组冲突的版本。
        for offset in range(len(source_records)):
            rotated = source_records[offset:] + source_records[:offset]
            candidate_tokens, candidate_indices, candidate_hashes = _repeat_tokens(
                rotated, tokenizer, max(max_lengths[group], block_size)
            )
            if tuple(candidate_tokens[:block_size]) not in first_blocks:
                token_ids, indices, hashes = candidate_tokens, candidate_indices, candidate_hashes
                logger.info("[gen] build_canonical_prefixes group=%s accepted rotation offset=%d", group, offset)
                break
        if token_ids is None:
            # Explicitly duplicated corpus selections can make every source rotation
            # identical. Add a deterministic group marker only in that collision case
            # so one bad override cannot abort the whole dataset generation.
            logger.info("[gen] build_canonical_prefixes group=%s all rotations collide -> adding deterministic marker", group)
            marker = tokenizer.encode(f"{group_position} prefix-cache-group-{group} ", add_special_tokens=False)
            source_tokens, source_indices, source_hashes = _repeat_tokens(
                source_records, tokenizer, max(max_lengths[group], block_size)
            )
            token_ids = marker + source_tokens
            indices, hashes = source_indices, source_hashes
            if tuple(token_ids[:block_size]) in first_blocks:
                raise ArtifactValidationError(f"canonical prefixes collide in first block for {group} after deterministic fallback")
        first_block = tuple(token_ids[:block_size])
        first_blocks.add(first_block)
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        actual = tokenizer.encode(text, add_special_tokens=False)
        # 校验前缀 decode/re-encode 后不变（round-trip 安全）。
        if actual[:max_lengths[group]] != token_ids[:max_lengths[group]]:
            raise ArtifactValidationError(f"canonical prefix does not round-trip for {group}")
        digest = hashlib.sha256(bytes(str(token_ids), "utf-8")).hexdigest()
        result[group] = CanonicalPrefix(group, text, tuple(token_ids), digest, indices, hashes)
        logger.info("[gen] build_canonical_prefixes group=%s tokens=%d text_len=%d sha256=%s gsm_indices=%s", group, len(token_ids), len(text), digest, indices)
    return result


def build_prompt(tokenizer: TokenizerLike, canonical: CanonicalPrefix, shared_prefix_tokens: int, seed: Sequence[int], suffix_records: Sequence[GSMRecord], target_tokens: int) -> tuple[str, tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    """按 共享前缀 + 唯一seed + 自然后缀 拼接出目标长度的 prompt 文本。

    返回 (文本, 实际token, 后缀来源索引, 后缀来源hash)，并校验 round-trip。
    """
    logger.info("[gen] build_prompt group=%s shared_prefix_tokens=%d seed_len=%d target_tokens=%d suffix_records=%d", canonical.group_id, shared_prefix_tokens, len(seed), target_tokens, len(suffix_records))
    suffix_len = target_tokens - shared_prefix_tokens - len(seed)
    logger.info("[gen] build_prompt suffix_len=%d", suffix_len)
    if suffix_len < 0:
        raise ArtifactValidationError("prefix and seed exceed target input length")
    suffix, indices, hashes = _repeat_tokens(suffix_records, tokenizer, suffix_len) if suffix_len else ([], (), ())
    expected = list(canonical.token_ids[:shared_prefix_tokens]) + list(seed) + suffix
    text = tokenizer.decode(expected, skip_special_tokens=False)
    actual = tokenizer.encode(text, add_special_tokens=False)
    logger.info("[gen] build_prompt group=%s expected_tokens=%d actual_tokens=%d text_len=%d suffix_indices=%d", canonical.group_id, len(expected), len(actual), len(text), len(indices))
    if actual != expected:
        raise PromptRoundTripError("prompt token layout changed after decode/re-encode")
    return text, tuple(actual), indices, hashes
