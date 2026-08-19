from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import RuntimeCapabilityError


@dataclass(frozen=True)
class RankMetrics:
    queries: int
    hits: int
    kv_cache_usage: float | None = None


@dataclass(frozen=True)
class MetricSnapshot:
    by_rank: dict[int, RankMetrics]
    metric_names: dict[str, str]
    raw_text: str = ""


@dataclass(frozen=True)
class ActualMetrics:
    by_rank: dict[int, RankMetrics]
    global_queries: int
    global_hits: int
    global_hit_rate: float | None


_SAMPLE = re.compile(r'^([^\s{]+)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)(?:\s+\d+)?$')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"])*)"')
_ALIASES = {
    "queries": ("vllm:prefix_cache_queries", "vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries", "vllm:gpu_prefix_cache_queries_total"),
    "hits": ("vllm:prefix_cache_hits", "vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits", "vllm:gpu_prefix_cache_hits_total"),
    "kv": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
}


def _rank(labels: dict[str, str], dp_size: int, mapping: dict[str, int]) -> int:
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
    selected: dict[str, str] = {}
    for logical, aliases in _ALIASES.items():
        selected_name = next((name for name in aliases if name in samples), None)
        if logical in {"queries", "hits"} and selected_name is None:
            raise RuntimeCapabilityError(f"missing vLLM Prefix Cache {logical} metric")
        if selected_name:
            selected[logical] = selected_name
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
    return {
        "by_dp": {str(rank): {"queries": row.queries, "hits": row.hits, "hit_rate": row.hits / row.queries if row.queries else None, "kv_cache_usage": row.kv_cache_usage} for rank, row in actual.by_rank.items()},
        "global_queries": actual.global_queries,
        "global_hits": actual.global_hits,
        "global_hit_rate": actual.global_hit_rate,
    }


def snapshot_to_dict(snapshot: MetricSnapshot, include_raw: bool = True) -> dict[str, Any]:
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
