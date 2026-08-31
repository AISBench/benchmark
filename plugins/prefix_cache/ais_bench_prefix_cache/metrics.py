from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import RuntimeCapabilityError


@dataclass(frozen=True)
class RankMetrics:
    """单个 DP rank 的 Prefix Cache 计数。"""
    queries: int        # 前缀查询次数
    hits: int           # 前缀命中次数
    kv_cache_usage: float | None = None  # 显存占用百分比（可选）


@dataclass(frozen=True)
class MetricSnapshot:
    """某一时刻抓取的指标快照（各 rank 的累计计数）。"""
    by_rank: dict[int, RankMetrics]
    metric_names: dict[str, str]   # 逻辑名 -> 实际 Prometheus 指标名
    raw_text: str = ""             # 原始 Prometheus 文本


@dataclass(frozen=True)
class ActualMetrics:
    """两次快照求差后得到的"本次运行"真实命中统计。"""
    by_rank: dict[int, RankMetrics]
    global_queries: int
    global_hits: int
    global_hit_rate: float | None


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


def parse_metrics(text: str, dp_size: int, engine_label_map: dict[str, int] | None = None) -> MetricSnapshot:
    """解析 vLLM 的 Prometheus 指标文本为 MetricSnapshot。

    逐行提取样本并按指标名聚合；为每个 DP rank 校验 queries/hits 是否齐全，
    并检查计数自洽（hits <= queries）。
    """
    mapping = engine_label_map or {}
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
    return MetricSnapshot(by_rank, selected, text)


def diff_metrics(before: MetricSnapshot, after: MetricSnapshot) -> ActualMetrics:
    """用两次快照求差得到本次运行的命中统计，并校验计数未回退、hits<=queries。"""
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
    total_queries = sum(value.queries for value in by_rank.values())
    total_hits = sum(value.hits for value in by_rank.values())
    return ActualMetrics(by_rank, total_queries, total_hits, total_hits / total_queries if total_queries else None)


def metrics_to_dict(actual: ActualMetrics) -> dict[str, Any]:
    """把 ActualMetrics 序列化为便于写入 analysis.json 的字典。"""
    return {
        "by_dp": {str(rank): {"queries": row.queries, "hits": row.hits, "hit_rate": row.hits / row.queries if row.queries else None, "kv_cache_usage": row.kv_cache_usage} for rank, row in actual.by_rank.items()},
        "global_queries": actual.global_queries,
        "global_hits": actual.global_hits,
        "global_hit_rate": actual.global_hit_rate,
    }


def summarize_kv_usage(samples: list[dict[int, float | None]]) -> dict[str, Any]:
    """聚合跑分期间轮询得到的 KV 用量样本。

    每个样本是 {rank: 用量占比}；None 表示该次抓取缺失该 rank 的 kv 指标。
    返回每个 rank 的峰值/均值/有效样本数，以及跨 rank 的全局峰值与均值。
    """
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
    return {
        "count": len(samples),
        "by_dp": by_dp,
        "global_peak": max(global_values) if global_values else None,
        "global_avg": sum(global_values) / len(global_values) if global_values else None,
    }


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
    return result

