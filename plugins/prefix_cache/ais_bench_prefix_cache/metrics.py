from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from .errors import RuntimeCapabilityError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankMetrics:
    """单个 DP rank 的 Prefix Cache 计数。"""
    queries: int        # 前缀查询次数
    hits: int           # 前缀命中次数
    kv_cache_usage: float | None = None  # 显存占用百分比（可选）


@dataclass(frozen=True)
class SpecDecodeSnapshot:
    """某一时刻抓取的投机推理累计指标（各 rank 的原始计数值）。"""
    by_rank: dict[int, dict[str, float]]
    metric_names: dict[str, str]   # 逻辑名 -> 实际 Prometheus 指标名


@dataclass(frozen=True)
class MetricSnapshot:
    """某一时刻抓取的指标快照（各 rank 的累计计数）。"""
    by_rank: dict[int, RankMetrics]
    metric_names: dict[str, str]   # 逻辑名 -> 实际 Prometheus 指标名
    raw_text: str = ""             # 原始 Prometheus 文本
    spec_decode: SpecDecodeSnapshot | None = None
    spec_decode_error: str | None = None


@dataclass(frozen=True)
class ActualMetrics:
    """两次快照求差后得到的"本次运行"真实命中统计。"""
    by_rank: dict[int, RankMetrics]
    global_queries: int
    global_hits: int
    global_hit_rate: float | None


@dataclass(frozen=True)
class ActualSpecDecodeMetrics:
    """两次快照求差后得到的"本次运行"真实投机推理统计。"""
    by_rank: dict[int, dict[str, Any]]
    global_draft_tokens: int | None
    global_accepted_tokens: int | None
    global_emitted_tokens: int | None
    global_draft_acceptance_rate: float | None
    global_efficiency: float | None
    status: str                    # ok / partial / counter_regressed / no_activity / unavailable
    reason: str | None = None
    covered_ranks: tuple[int, ...] = ()
    missing_ranks: tuple[int, ...] = ()


# Prometheus 文本行的样本正则：metric_name{labels} value
_SAMPLE = re.compile(r'^([^\s{]+)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)(?:\s+\d+)?$')
# 标签键值对正则：key="value"（支持转义）
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')
# vLLM 各版本暴露的指标名别名，按优先级取第一个出现的。
_ALIASES = {
    "queries": ("vllm:prefix_cache_queries", "vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries", "vllm:gpu_prefix_cache_queries_total"),
    "hits": ("vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits", "vllm:gpu_prefix_cache_hits_total"),
    "kv": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}
# vLLM 投机推理（speculative decoding）指标别名。Counter 在不同版本中可能带
# `_total` 后缀，也可能不带；按优先级取第一个出现的。
_SPEC_ALIASES = {
    "draft_tokens": ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_draft_tokens"),
    "accepted_tokens": ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens"),
    "emitted_tokens": ("vllm:spec_decode_num_emitted_tokens_total", "vllm:spec_decode_num_emitted_tokens"),
    "draft_acceptance_rate": ("vllm:spec_decode_draft_acceptance_rate",),
    "efficiency": ("vllm:spec_decode_efficiency",),
}
_SPEC_COUNTERS = ("draft_tokens", "accepted_tokens", "emitted_tokens")


def _rank(labels: dict[str, str], dp_size: int, mapping: dict[str, int]) -> int:
    """从样本标签中解析出 DP rank：优先 engine 标签映射，其次回退到尾部数字。"""
    value = labels.get("engine")
    if value is None:
        if dp_size == 1:
            return 0
        raise RuntimeCapabilityError("metric sample is missing engine label")
    if value in mapping:
        return int(mapping[value])
    match = re.search(r"(\d+)$", value)
    if not match:
        raise RuntimeCapabilityError(f"cannot map engine label to DP rank: {value}")
    return int(match.group(1))


def _parse_spec_decode(
    samples: dict[str, list[tuple[dict[str, str], float]]],
    dp_size: int,
    mapping: dict[str, int],
) -> tuple[SpecDecodeSnapshot | None, str | None]:
    """解析投机推理指标；缺失或无法定位 rank 时返回错误说明而不影响主流程。"""
    selected = {
        logical: next((name for name in aliases if name in samples), None)
        for logical, aliases in _SPEC_ALIASES.items()
    }
    selected = {logical: name for logical, name in selected.items() if name is not None}
    if not selected:
        return None, None
    try:
        by_rank: dict[int, dict[str, float]] = {}
        for logical, name in selected.items():
            for labels, value in samples[name]:
                rank = _rank(labels, dp_size, mapping)
                if rank < 0 or rank >= dp_size:
                    raise RuntimeCapabilityError(f"metric contains out-of-range DP rank {rank}")
                row = by_rank.setdefault(rank, {})
                if logical in row:
                    # 多组标签（例如多个 model_name）同时暴露同一指标时，Counter
                    # 求和才是总工作量；Gauge 保留最后一个采样值。
                    if logical in _SPEC_COUNTERS:
                        logger.warning(
                            "[metrics] duplicate speculative metric %s for DP rank %d; summing values",
                            logical, rank,
                        )
                        row[logical] += value
                    else:
                        logger.warning(
                            "[metrics] duplicate speculative metric %s for DP rank %d; keeping last value",
                            logical, rank,
                        )
                        row[logical] = value
                else:
                    row[logical] = value
    except RuntimeCapabilityError as exc:
        logger.warning("[metrics] parse_spec_decode skipped reason=%s", exc)
        return None, str(exc)
    return SpecDecodeSnapshot(by_rank, selected), None


def parse_metrics(text: str, dp_size: int, engine_label_map: dict[str, int] | None = None) -> MetricSnapshot:
    """解析 vLLM 的 Prometheus 指标文本为 MetricSnapshot。

    逐行提取样本并按指标名聚合；为每个 DP rank 校验 queries/hits 是否齐全，
    并检查计数自洽（hits <= queries）。
    """
    mapping = engine_label_map or {}
    logger.info(
        "[metrics] parse_metrics start text_bytes=%d text_lines=%d dp_size=%d engine_label_map=%s",
        len(text.encode("utf-8")),
        len(text.splitlines()),
        dp_size,
        mapping,
    )
    samples: dict[str, list[tuple[dict[str, str], float]]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if not match:
            continue
        name, label_text, value = match.groups()
        labels = {key: bytes(val, "utf-8").decode("unicode_escape") for key, val in _LABEL.findall(label_text or "")}
        samples.setdefault(name, []).append((labels, float(value)))
    # 把逻辑名（queries/hits/kv）映射到实际出现的指标名。
    selected: dict[str, str] = {}
    for logical, aliases in _ALIASES.items():
        selected_name = next((name for name in aliases if name in samples), None)
        if logical in {"queries", "hits"} and selected_name is None:
            raise RuntimeCapabilityError(f"missing vLLM Prefix Cache {logical} metric")
        if selected_name:
            selected[logical] = selected_name
    logger.info("[metrics] parse_metrics selected_metric_names=%s", selected)
    # 按 rank 聚合各逻辑指标的取值。
    values: dict[int, dict[str, float]] = {}
    for logical, name in selected.items():
        for labels, value in samples[name]:
            rank = _rank(labels, dp_size, mapping)
            if rank < 0 or rank >= dp_size:
                raise RuntimeCapabilityError(f"metric contains out-of-range DP rank {rank}")
            rank_values = values.setdefault(rank, {})
            if logical in rank_values:
                raise RuntimeCapabilityError(f"duplicate {logical} metric for DP rank {rank}")
            rank_values[logical] = value
    missing = sorted(set(range(dp_size)) - set(values))
    if missing:
        raise RuntimeCapabilityError(f"missing DP ranks: {', '.join(map(str, missing))}")
    by_rank = {}
    for rank in range(dp_size):
        row = values[rank]
        if "queries" not in row or "hits" not in row:
            raise RuntimeCapabilityError(f"incomplete Prefix Cache metrics for DP rank {rank}")
        queries, hits = int(row["queries"]), int(row["hits"])
        if hits > queries:
            raise RuntimeCapabilityError(f"Prefix Cache hits exceed queries for DP rank {rank}")
        by_rank[rank] = RankMetrics(queries, hits, row.get("kv"))
    spec_decode, spec_decode_error = _parse_spec_decode(samples, dp_size, mapping)
    logger.info(
        "[metrics] parse_metrics complete by_dp=%s spec_decode=%s",
        {
            str(rank): {
                "queries": row.queries,
                "hits": row.hits,
                "kv_cache_usage": row.kv_cache_usage,
            }
            for rank, row in by_rank.items()
        },
        None if spec_decode is None else {"metric_names": spec_decode.metric_names, "by_dp": spec_decode.by_rank},
    )
    return MetricSnapshot(by_rank, selected, text, spec_decode, spec_decode_error)


def diff_metrics(before: MetricSnapshot, after: MetricSnapshot) -> ActualMetrics:
    """用两次快照求差得到本次运行的命中统计，并校验计数未回退、hits<=queries。"""
    logger.info(
        "[metrics] diff_metrics start before_ranks=%s after_ranks=%s before_names=%s after_names=%s",
        sorted(before.by_rank),
        sorted(after.by_rank),
        before.metric_names,
        after.metric_names,
    )
    if set(before.by_rank) != set(after.by_rank):
        raise RuntimeCapabilityError("metric snapshots contain different DP ranks")
    by_rank: dict[int, RankMetrics] = {}
    for rank in sorted(before.by_rank):
        old, new = before.by_rank[rank], after.by_rank[rank]
        queries, hits = new.queries - old.queries, new.hits - old.hits
        if queries < 0 or hits < 0:
            raise RuntimeCapabilityError(f"Prefix Cache counter regressed for DP rank {rank}")
        if hits > queries:
            raise RuntimeCapabilityError(f"Prefix Cache hit delta exceeds query delta for DP rank {rank}")
        by_rank[rank] = RankMetrics(queries, hits, new.kv_cache_usage)
        logger.info(
            "[metrics] diff_metrics rank=%d before_queries=%d after_queries=%d query_delta=%d before_hits=%d after_hits=%d hit_delta=%d hit_rate=%s kv_cache_usage_after=%s",
            rank,
            old.queries,
            new.queries,
            queries,
            old.hits,
            new.hits,
            hits,
            hits / queries if queries else None,
            new.kv_cache_usage,
        )
    total_queries = sum(value.queries for value in by_rank.values())
    total_hits = sum(value.hits for value in by_rank.values())
    result = ActualMetrics(by_rank, total_queries, total_hits, total_hits / total_queries if total_queries else None)
    logger.info(
        "[metrics] diff_metrics complete global_queries=%d global_hits=%d global_hit_rate=%s",
        result.global_queries,
        result.global_hits,
        result.global_hit_rate,
    )
    return result


def metrics_to_dict(actual: ActualMetrics) -> dict[str, Any]:
    """把 ActualMetrics 序列化为便于写入 analysis.json 的字典。"""
    return {
        "by_dp": {str(rank): {"queries": row.queries, "hits": row.hits, "hit_rate": row.hits / row.queries if row.queries else None, "kv_cache_usage": row.kv_cache_usage} for rank, row in actual.by_rank.items()},
        "global_queries": actual.global_queries,
        "global_hits": actual.global_hits,
        "global_hit_rate": actual.global_hit_rate,
    }


def _json_number(value: float) -> float | int:
    """整数计数写成 JSON 整数，Gauge 保持浮点。"""
    return int(value) if float(value).is_integer() else value


def diff_spec_decode(before: MetricSnapshot, after: MetricSnapshot) -> ActualSpecDecodeMetrics:
    """用两次快照求差得到本次运行的投机推理真实数据。

    指标缺失时返回 ``status="unavailable"``，不会让主基准流程失败；部分 rank
    缺失或计数回退时分别标记为 ``partial`` / ``counter_regressed``。
    真正速率一律用 Counter 差值计算，而不是直接采用 Gauge：
    ``acceptance = accepted_delta / draft_delta``，
    ``efficiency = emitted_delta / draft_delta``。
    """
    logger.info(
        "[metrics] diff_spec_decode start before_ranks=%s after_ranks=%s",
        sorted(before.spec_decode.by_rank) if before.spec_decode else None,
        sorted(after.spec_decode.by_rank) if after.spec_decode else None,
    )
    if before.spec_decode is None or after.spec_decode is None:
        reasons: list[str] = []
        if before.spec_decode is None:
            reasons.append(before.spec_decode_error or "baseline snapshot has no speculative-decoding metrics")
        if after.spec_decode is None:
            reasons.append(after.spec_decode_error or "after snapshot has no speculative-decoding metrics")
        result = ActualSpecDecodeMetrics(
            {}, None, None, None, None, None, "unavailable", "; ".join(reasons),
        )
        logger.info("[metrics] diff_spec_decode unavailable reason=%s", result.reason)
        return result

    before_rows = before.spec_decode.by_rank
    after_rows = after.spec_decode.by_rank
    ranks = sorted(set(before_rows) | set(after_rows))
    expected_ranks = sorted(set(before.by_rank) & set(after.by_rank))
    missing_ranks = tuple(
        rank for rank in expected_ranks
        if rank not in before_rows or rank not in after_rows
    )
    by_rank: dict[int, dict[str, Any]] = {}
    issues: list[str] = []
    has_regression = False
    has_missing_metric = False
    for rank in ranks:
        old = before_rows.get(rank, {})
        new = after_rows.get(rank, {})
        deltas: dict[str, float | int] = {}
        missing: list[str] = []
        regressed: list[str] = []
        for name in _SPEC_COUNTERS:
            if name not in old or name not in new:
                missing.append(name)
                continue
            delta = new[name] - old[name]
            if delta < 0:
                regressed.append(name)
                continue
            deltas[name] = _json_number(delta)
        if regressed:
            status = "counter_regressed"
            has_regression = True
            issues.append(f"DP rank {rank}: counter regressed ({', '.join(regressed)})")
        elif missing:
            status = "missing_metric"
            has_missing_metric = True
            issues.append(f"DP rank {rank}: missing counters ({', '.join(missing)})")
        else:
            status = "ok"
        draft = deltas.get("draft_tokens")
        accepted = deltas.get("accepted_tokens")
        emitted = deltas.get("emitted_tokens")
        by_rank[rank] = {
            "status": status,
            "before": {
                name: _json_number(old[name])
                for name in (*_SPEC_COUNTERS, "draft_acceptance_rate", "efficiency")
                if name in old
            },
            "after": {
                name: _json_number(new[name])
                for name in (*_SPEC_COUNTERS, "draft_acceptance_rate", "efficiency")
                if name in new
            },
            "deltas": deltas,
            "draft_acceptance_rate": accepted / draft if status == "ok" and draft else None,
            "efficiency": emitted / draft if status == "ok" and draft else None,
        }
    covered_ranks = tuple(rank for rank in ranks if by_rank[rank]["status"] == "ok")
    complete = [by_rank[rank] for rank in covered_ranks]
    if complete:
        draft_total: int | None = sum(int(row["deltas"]["draft_tokens"]) for row in complete)
        accepted_total: int | None = sum(int(row["deltas"]["accepted_tokens"]) for row in complete)
        emitted_total: int | None = sum(int(row["deltas"]["emitted_tokens"]) for row in complete)
    else:
        draft_total = accepted_total = emitted_total = None
    acceptance_rate = (
        accepted_total / draft_total
        if accepted_total is not None and draft_total
        else None
    )
    efficiency = (
        emitted_total / draft_total
        if emitted_total is not None and draft_total
        else None
    )
    if missing_ranks:
        issues.append(
            "no speculative-decoding metrics for DP ranks: "
            + ", ".join(map(str, missing_ranks))
        )
    if not by_rank:
        status = "unavailable"
        reason = "no speculative-decoding metrics in either snapshot"
    elif has_regression:
        status = "counter_regressed"
        reason = "; ".join(issues)
    elif has_missing_metric or missing_ranks:
        status = "partial"
        reason = "; ".join(issues)
    elif not draft_total:
        status = "no_activity"
        reason = "no draft tokens were observed during this run"
    else:
        status = "ok"
        reason = None
    result = ActualSpecDecodeMetrics(
        by_rank=by_rank,
        global_draft_tokens=draft_total,
        global_accepted_tokens=accepted_total,
        global_emitted_tokens=emitted_total,
        global_draft_acceptance_rate=acceptance_rate,
        global_efficiency=efficiency,
        status=status,
        reason=reason,
        covered_ranks=covered_ranks,
        missing_ranks=missing_ranks,
    )
    logger.info(
        "[metrics] diff_spec_decode complete status=%s draft=%s accepted=%s emitted=%s acceptance_rate=%s efficiency=%s covered_ranks=%s missing_ranks=%s reason=%s",
        result.status,
        result.global_draft_tokens,
        result.global_accepted_tokens,
        result.global_emitted_tokens,
        result.global_draft_acceptance_rate,
        result.global_efficiency,
        result.covered_ranks,
        result.missing_ranks,
        result.reason,
    )
    return result


def spec_decode_to_dict(actual: ActualSpecDecodeMetrics) -> dict[str, Any]:
    """把投机推理真实数据序列化为 analysis.json 中的 actual.spec_decode。"""
    return {
        "status": actual.status,
        "reason": actual.reason,
        "global_draft_tokens": actual.global_draft_tokens,
        "global_accepted_tokens": actual.global_accepted_tokens,
        "global_emitted_tokens": actual.global_emitted_tokens,
        "global_draft_acceptance_rate": actual.global_draft_acceptance_rate,
        "global_efficiency": actual.global_efficiency,
        "covered_ranks": list(actual.covered_ranks),
        "missing_ranks": list(actual.missing_ranks),
        "by_dp": {str(rank): row for rank, row in actual.by_rank.items()},
    }


def summarize_kv_usage(samples: list[dict[int, float | None]]) -> dict[str, Any]:
    """聚合跑分期间轮询得到的 KV 用量样本。

    每个样本是 {rank: 用量占比}；None 表示该次抓取缺失该 rank 的 kv 指标。
    返回每个 rank 的峰值/均值/有效样本数，以及跨 rank 的全局峰值与均值。
    """
    logger.info("[metrics] summarize_kv_usage start sample_count=%d", len(samples))
    ranks = sorted({rank for sample in samples for rank in sample})
    by_dp: dict[str, dict[str, Any]] = {}
    global_values: list[float] = []
    for rank in ranks:
        values = [sample[rank] for sample in samples if rank in sample and sample[rank] is not None]
        if values:
            global_values.extend(values)
        by_dp[str(rank)] = {
            "peak": max(values) if values else None,
            "avg": sum(values) / len(values) if values else None,
            "sample_count": len(values),
        }
    result = {
        "count": len(samples),
        "by_dp": by_dp,
        "global_peak": max(global_values) if global_values else None,
        "global_avg": sum(global_values) / len(global_values) if global_values else None,
    }
    logger.info("[metrics] summarize_kv_usage complete summary=%s", result)
    return result


def snapshot_to_dict(snapshot: MetricSnapshot, include_raw: bool = True) -> dict[str, Any]:
    """把 MetricSnapshot 序列化为字典（可选附带原始 Prometheus 文本用于审计）。"""
    result: dict[str, Any] = {
        "metric_names": snapshot.metric_names,
        "by_dp": {
            str(rank): {
                "queries": row.queries,
                "hits": row.hits,
                "kv_cache_usage": row.kv_cache_usage,
            }
            for rank, row in snapshot.by_rank.items()
        },
    }
    if include_raw:
        result["raw_prometheus"] = snapshot.raw_text
    if snapshot.spec_decode is None:
        result["spec_decode"] = {
            "available": False,
            "reason": snapshot.spec_decode_error,
        }
    else:
        result["spec_decode"] = {
            "available": True,
            "metric_names": snapshot.spec_decode.metric_names,
            "by_dp": {
                str(rank): {
                    name: _json_number(value)
                    for name, value in row.items()
                }
                for rank, row in snapshot.spec_decode.by_rank.items()
            },
        }
    logger.info(
        "[metrics] snapshot_to_dict ranks=%s include_raw=%s raw_bytes=%d spec_decode_available=%s",
        sorted(snapshot.by_rank),
        include_raw,
        len(snapshot.raw_text.encode("utf-8")) if include_raw else 0,
        snapshot.spec_decode is not None,
    )
    return result
