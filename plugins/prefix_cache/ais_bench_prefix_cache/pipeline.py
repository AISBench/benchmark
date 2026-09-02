from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .artifacts import ArtifactPaths, artifact_paths, read_jsonl, sha256_file, validate_artifacts, write_json, write_jsonl
from .errors import ArtifactValidationError, PromptRoundTripError
from .generation import (
    RequestPlan,
    assign_cold_routes,
    assign_groups,
    build_canonical_prefixes,
    build_input_lengths,
    build_output_lengths,
    build_prompt,
    build_unique_seed,
    build_unique_seed_tokens,
    find_boundary_safe_token_ids,
    load_gsm8k,
    order_indices,
    select_gsm8k,
    simulate_theory,
    solve_prefix_lengths,
)
from .scenario import Scenario, load_scenario, new_execution_timestamp, with_execution_timestamp

logger = logging.getLogger(__name__)


def _tokenizer_loader(scenario: Scenario):
    """按场景配置加载 HuggingFace AutoTokenizer，作为默认 tokenizer 加载器。"""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ArtifactValidationError("transformers is required to load the configured tokenizer") from exc
    cfg = scenario.section("tokenizer")
    logger.info("[prepare] _tokenizer_loader path=%s revision=%s trust_remote_code=%s", cfg["path"], cfg.get("revision"), cfg.get("trust_remote_code", False))
    return AutoTokenizer.from_pretrained(cfg["path"], revision=cfg.get("revision"), trust_remote_code=cfg.get("trust_remote_code", False))


def _sha256_json(value: Any) -> str:
    """对任意 JSON 可序列化值做规范化后计算 SHA-256（键排序、紧凑分隔符）。"""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_random_seed(global_seed: int, request_id: str) -> int:
    """由全局种子 + 请求 id 派生每条请求的确定性随机种子（可复现）。"""
    digest = hashlib.sha256(f"{global_seed}:{request_id}".encode("utf-8")).digest()
    result = int.from_bytes(digest[:8], "big")
    logger.info("[prepare] _request_random_seed global_seed=%d request_id=%s -> %d", global_seed, request_id, result)
    return result


def _percentile(sorted_values: list[int], percentile: float) -> float:
    """对已排序序列做线性插值分位数（支持单元素序列）。"""
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _length_summary(values: list[int]) -> dict[str, Any]:
    """生成长度分布的摘要：min/max/mean/分位数，并分为最多 10 桶直方图。

    每个 bin 的 min/max 是该桶内实际取值的最小/最大值（count == 1 时二者相等）；
    桶按固定宽度 [bin_low, bin_high] 划分（最多 10 桶），仅输出非空桶。
    """
    logger.info("[prepare] _length_summary values_count=%d", len(values))
    ordered = sorted(values)
    low, high = ordered[0], ordered[-1]
    width = max(1, math.ceil((high - low + 1) / 10))
    bins: dict[tuple[int, int], dict[str, int]] = {}
    for value in ordered:
        bin_low = low + ((value - low) // width) * width
        bounds = (bin_low, min(high, bin_low + width - 1))
        slot = bins.get(bounds)
        if slot is None:
            bins[bounds] = slot = {"min": value, "max": value, "count": 0}
        slot["min"] = min(slot["min"], value)
        slot["max"] = max(slot["max"], value)
        slot["count"] += 1
    result = {
        "min": low,
        "max": high,
        "mean": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "bins": [
            {"min": slot["min"], "max": slot["max"], "count": slot["count"]}
            for _, slot in sorted(bins.items())
        ],
    }
    logger.info("[prepare] _length_summary result=%s", result)
    return result


def _tokenizer_manifest(tokenizer: Any, effective: dict[str, Any], block_size: int) -> dict[str, Any]:
    """生成 tokenizer 的指纹信息（路径/类/词表/特殊 id），用于工件溯源与一致性校验。"""
    special_ids = sorted(int(value) for value in getattr(tokenizer, "all_special_ids", []))
    fingerprint_source = {
        "path": effective["tokenizer"]["path"],
        "revision": effective["tokenizer"].get("revision"),
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "vocab_size": len(tokenizer),
        "special_token_ids": special_ids,
    }
    logger.info("[prepare] _tokenizer_manifest tokenizer_class=%s block_size=%d fingerprint_source=%s", fingerprint_source["class"], block_size, fingerprint_source)
    result = fingerprint_source | {
        "block_size": block_size,
        "fingerprint_sha256": _sha256_json(fingerprint_source),
    }
    logger.info("[prepare] _tokenizer_manifest result=%s", result)
    return result


def _build_prompt_with_seed_retry(tokenizer: Any, canonical: Any, prefix_len: int, seeds: dict[str, tuple[int, ...]], request_id: str, rotated_pool: list[Any], target_tokens: int, safe_ids: list[int], seed_length: int, random_seed: int):
    """构造 prompt；若 round-trip 失败则换一个唯一 seed 重试，最多 64 次。"""
    logger.info("[prepare] _build_prompt_with_seed_retry request_id=%s prefix_len=%d target_tokens=%d seed_length=%d random_seed=%d rotated_pool=%d", request_id, prefix_len, target_tokens, seed_length, random_seed, len(rotated_pool))
    for attempt in range(64):
        try:
            result = build_prompt(tokenizer, canonical, prefix_len, seeds[request_id], rotated_pool, target_tokens)
            logger.info("[prepare] _build_prompt_with_seed_retry request_id=%s attempt=%d ok", request_id, attempt)
            return result
        except PromptRoundTripError:
            # seed 可能导致 decode/re-encode 后 token 布局变化，重新生成一个不重复的 seed。
            logger.info("[prepare] _build_prompt_with_seed_retry request_id=%s attempt=%d PromptRoundTripError -> regenerating seed", request_id, attempt)
            seeds[request_id] = build_unique_seed(tokenizer, safe_ids, request_id, seed_length, random_seed + attempt * 10007 + 1, set(seeds.values()))
    raise ArtifactValidationError(f"unable to construct a round-trip-safe prompt for {request_id}")


def prepare_scenario(
    path: Path | str,
    overwrite: bool | None = None,
    tokenizer_loader: Callable[[Scenario], Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
    execution_timestamp: str | None = None,
) -> ArtifactPaths:
    """准备阶段：从场景配置生成全部请求工件（full/requests/manifest/analysis）。

    流程：解析配置 → 生成输入/输出长度、分组 → 应用 group override → 排序 →
    cold 路由 → 求解共享前缀 → 构造 canonical 前缀与每条 prompt → 理论命中率模拟 →
    落盘工件并校验。整个过程不发任何网络请求。
    """
    logger.info("[prepare] prepare_scenario path=%s overwrite=%s tokenizer_loader=%s", path, overwrite, tokenizer_loader)
    scenario = with_execution_timestamp(load_scenario(path), execution_timestamp or new_execution_timestamp())
    logger.info("[prepare] scenario run_id=%s random_seed=%d cache_mode=%s dp_size=%d block_size=%d output_dir=%s source_path=%s", scenario.run_id, scenario.random_seed, scenario.cache_mode, scenario.dp_size, scenario.block_size, scenario.output_dir, scenario.source_path)
    effective = scenario.to_effective_dict()
    logger.info("[prepare] effective keys=%s", sorted(effective))
    overwrite = effective["run"].get("overwrite", False) if overwrite is None else overwrite
    request_cfg = effective["requests"]
    pc_cfg = effective["prefix_cache"]
    corpus_cfg = effective["corpus"]
    logger.info("[prepare] overwrite=%s request_cfg=%s", overwrite, request_cfg)
    logger.info("[prepare] pc_cfg=%s", pc_cfg)
    logger.info("[prepare] corpus_cfg=%s", corpus_cfg)
    count = request_cfg["count"]
    seed = scenario.random_seed
    logger.info("[prepare] count=%d seed=%d", count, seed)
    # 阶段1：按配置生成输入/输出长度序列，并按权重把请求分配到各 Prefix Group。
    input_lengths = build_input_lengths(request_cfg["input_length"], count, seed)
    output_lengths = build_output_lengths(request_cfg["output_length"], count, seed + 1)
    groups = assign_groups(count, pc_cfg["groups"], seed + 2)
    logger.info("[prepare] input_lengths=%s", input_lengths)
    logger.info("[prepare] output_lengths=%s", output_lengths)
    logger.info("[prepare] groups=%s distribution=%s", groups, {group: groups.count(group) for group in sorted(set(groups))})
    records = load_gsm8k(Path(corpus_cfg["path"]), corpus_cfg["field"])
    logger.info("[prepare] records=%d", len(records))
    # 阶段2：应用每个组的 override（独立输入/输出长度、语料选择），构造各组的语料池。
    overrides = pc_cfg["groups"].get("overrides", {})
    logger.info("[prepare] overrides=%s", overrides)
    group_pools: dict[str, list[Any]] = {}
    for group_index, group in enumerate(sorted(set(groups))):
        group_positions = [index for index, value in enumerate(groups) if value == group]
        logger.info("[prepare] group=%s group_index=%d positions_count=%d positions=%s", group, group_index, len(group_positions), group_positions)
        override = overrides.get(group, {})
        logger.info("[prepare] group=%s override=%s", group, override)
        if "input_length" in override:
            values = build_input_lengths(override["input_length"], len(group_positions), seed + 100 + group_index)
            if len(values) != len(group_positions):
                raise ArtifactValidationError(f"{group} input_length generated {len(values)} values; expected {len(group_positions)}")
            for position, value in zip(group_positions, values):
                input_lengths[position] = value
            logger.info("[prepare] group=%s input_length override values=%s -> input_lengths=%s", group, values, input_lengths)
        if "output_length" in override:
            values = build_output_lengths(override["output_length"], len(group_positions), seed + 200 + group_index)
            for position, value in zip(group_positions, values):
                output_lengths[position] = value
            logger.info("[prepare] group=%s output_length override values=%s -> output_lengths=%s", group, values, output_lengths)
        selection = override.get("corpus_selection", corpus_cfg["selection"])
        logger.info("[prepare] group=%s selection=%s", group, selection)
        group_pools[group] = select_gsm8k(records, selection, max(2, len(group_positions)), seed + 300 + group_index)
    logger.info("[prepare] group_pools sizes=%s", {group: len(pool) for group, pool in group_pools.items()})
    # 阶段3：按配置策略重排请求顺序，长度/分组随之对齐。
    ordering = order_indices(groups, pc_cfg["order"]["strategy"], seed + 4, input_lengths)
    logger.info("[prepare] ordering=%s", ordering)
    input_lengths = [input_lengths[index] for index in ordering]
    output_lengths = [output_lengths[index] for index in ordering]
    groups = [groups[index] for index in ordering]
    logger.info("[prepare] after ordering input_lengths=%s", input_lengths)
    logger.info("[prepare] after ordering output_lengths=%s", output_lengths)
    logger.info("[prepare] after ordering groups=%s", groups)
    # 阶段4：cold 模式给每条请求分派 DP rank 与 lane；warmup 模式无需路由。
    if scenario.cache_mode == "cold":
        ranks_raw, lane_raw = assign_cold_routes(groups, scenario.dp_size)
        ranks: list[int | None] = ranks_raw
        lanes: list[int | None] = lane_raw
        logger.info("[prepare] cache_mode=cold dp_size=%d ranks=%s lanes=%s", scenario.dp_size, ranks, lanes)
    else:
        ranks = [None] * count
        lanes = [None] * count
        logger.info("[prepare] cache_mode=%s -> ranks=lanes=None for all %d requests", scenario.cache_mode, count)
    seed_length = scenario.block_size * pc_cfg["seed_blocks"]
    minimum_non_shared_length = pc_cfg["minimum_non_shared_length"]
    logger.info("[prepare] seed_length=%d minimum_non_shared_length=%d", seed_length, minimum_non_shared_length)
    # 阶段5：求解每条请求的共享前缀长度，使理论命中率逼近目标。
    solve = solve_prefix_lengths(input_lengths, output_lengths, groups, ranks, lanes, scenario.block_size, minimum_non_shared_length, scenario.cache_mode, pc_cfg["target_hit_rate"])
    logger.info("[prepare] solve shared_prefix_tokens=%s", solve.shared_prefix_tokens)
    logger.info("[prepare] solve requested_hit_tokens=%d effective_hit_tokens=%d effective_hit_rate=%.4f min_reachable_rate=%.4f max_reachable_rate=%.4f target_reachable=%s adjusted=%s reason=%s", solve.requested_hit_tokens, solve.effective_hit_tokens, solve.effective_hit_rate, solve.min_reachable_rate, solve.max_reachable_rate, solve.target_reachable, solve.adjusted, solve.reason)
    max_by_group = {group: max((prefix for prefix, current in zip(solve.shared_prefix_tokens, groups) if current == group), default=0) for group in sorted(set(groups))}
    group_sources = {group: group_pools[group] for group in sorted(set(groups))}
    logger.info("[prepare] max_by_group=%s", max_by_group)
    logger.info("[prepare] group_sources sizes=%s", {group: len(pool) for group, pool in group_sources.items()})
    # 阶段6：加载 tokenizer，为每个组构造 canonical 前缀并选出边界安全的 seed token。
    tokenizer = (tokenizer_loader or _tokenizer_loader)(scenario)
    logger.info("[prepare] tokenizer=%s vocab_size=%d", f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}", len(tokenizer))
    canonical = build_canonical_prefixes(tokenizer, group_sources, max_by_group, scenario.block_size)
    logger.info("[prepare] canonical=%s", {group: {"sha256": item.sha256, "tokens": len(item.token_ids), "gsm_indices": list(item.gsm_indices), "gsm_hashes": list(item.gsm_hashes)} for group, item in canonical.items()})
    safe_ids = find_boundary_safe_token_ids(tokenizer, max(2, min(64, len(tokenizer))))
    logger.info("[prepare] safe_ids count=%d safe_ids=%s", len(safe_ids), safe_ids)
    # 阶段7：为每条请求生成唯一 seed，再按 shared_prefix + seed + 自然后缀拼 prompt。
    request_ids = [f"request-{index:08d}" for index in range(count)]
    request_random_seeds = {
        request_id: _request_random_seed(seed + 5, request_id)
        for request_id in request_ids
    }
    logger.info("[prepare] request_ids count=%d first=%s last=%s", len(request_ids), request_ids[0], request_ids[-1])
    logger.info("[prepare] request_random_seeds=%s", request_random_seeds)
    seeds: dict[str, tuple[int, ...]] = {}
    used_seeds: set[tuple[int, ...]] = set()
    for request_id in request_ids:
        request_seed = build_unique_seed(
            tokenizer, safe_ids, request_id, seed_length,
            request_random_seeds[request_id], used_seeds,
        )
        used_seeds.add(request_seed)
        seeds[request_id] = request_seed
        logger.info("[prepare] request_id=%s seed=%s used_seeds=%d", request_id, request_seed, len(used_seeds))
    plans: list[RequestPlan] = []
    occurrences: dict[str, int] = {}
    if progress is not None:
        progress(0, count)
    for index, request_id in enumerate(request_ids):
        group = groups[index]
        occurrence = occurrences.get(group, 0)
        occurrences[group] = occurrence + 1
        prefix_len = solve.shared_prefix_tokens[index]
        pool = group_pools[group]
        # 组内循环轮换语料池，保证不同请求的后缀不同但前缀可共享。
        rotated_pool = pool[occurrence % len(pool):] + pool[:occurrence % len(pool)]
        logger.info("[prepare] build plan index=%d request_id=%s group=%s occurrence=%d prefix_len=%d pool_size=%d rotation_offset=%d", index, request_id, group, occurrence, prefix_len, len(pool), occurrence % len(pool))
        text, tokens, suffix_indices, suffix_hashes = _build_prompt_with_seed_retry(tokenizer, canonical[group], prefix_len, seeds, request_id, rotated_pool, input_lengths[index], safe_ids, seed_length, request_random_seeds[request_id])
        seed_hash = hashlib.sha256(str(seeds[request_id]).encode()).hexdigest()
        plan = RequestPlan(
            request_id, index, group, occurrence, ranks[index], lanes[index], input_lengths[index], len(tokens), output_lengths[index], prefix_len, seed_length, len(tokens) - prefix_len - seed_length,
            text, "none", suffix_indices, suffix_hashes, canonical[group].sha256, seed_hash,
            request_random_seed=request_random_seeds[request_id],
        )
        plans.append(plan)
        logger.info("[prepare] plan request_id=%s sequence_index=%d group=%s occurrence=%d dp_rank=%s lane=%s target_input_tokens=%d actual_input_tokens=%d max_tokens=%d shared_prefix_tokens=%d seed_tokens=%d natural_suffix_tokens=%d seed_sha256=%s", plan.request_id, plan.sequence_index, plan.group_id, plan.occurrence_index_within_group, plan.dp_rank, plan.lane_sequence, plan.target_input_tokens, plan.actual_input_tokens, plan.max_tokens, plan.shared_prefix_tokens, plan.seed_tokens, plan.natural_suffix_tokens, plan.seed_sha256)
        if progress is not None:
            progress(index + 1, count)
    warm_watermarks = max_by_group if scenario.cache_mode == "warmup" else None
    logger.info("[prepare] warm_watermarks=%s", warm_watermarks)
    # 阶段8：按缓存水位模拟理论命中率，并生成 full/requests 行。
    theory = simulate_theory(plans, scenario.cache_mode, warm_watermarks)
    logger.info("[prepare] theory total_input_tokens=%d total_hit_tokens=%d global_hit_rate=%.4f", theory.total_input_tokens, theory.total_hit_tokens, theory.global_hit_rate)
    logger.info("[prepare] theory group_stats=%s dp_stats=%s", theory.group_stats, theory.dp_stats)
    full_rows = [
        row.to_dict() | {
            "theoretical_hit_rate": row.theoretical_hit_tokens / row.actual_input_tokens if row.actual_input_tokens else 0.0,
            "divergence_block_sha256": row.seed_sha256,
            "divergence_unique": True,
            "collision_status": "pass",
        }
        for row in theory.rows
    ]
    request_rows = [{"question": row.question, "answer": row.answer, "max_tokens": row.max_tokens} for row in theory.rows]
    logger.info("[prepare] full_rows=%d request_rows=%d first_full_row=%s", len(full_rows), len(request_rows), full_rows[0])
    paths = artifact_paths(scenario.output_dir, scenario.run_id)
    logger.info("[prepare] paths full=%s requests=%s manifest=%s analysis=%s", paths.full, paths.requests, paths.manifest, paths.analysis)
    # 阶段9：落盘 full/requests 工件，并在 warmup 模式下生成预热计划。
    write_jsonl(paths.full, full_rows, overwrite)
    write_jsonl(paths.requests, request_rows, overwrite)
    warmup_plan = []
    if scenario.cache_mode == "warmup":
        # warmup 模式：为每个 (group, rank) 生成一条预热请求，把缓存前缀写满。
        warm_ids = [f"warmup:{group}:{rank}" for group in sorted(canonical) for rank in range(scenario.dp_size)]
        logger.info("[prepare] warmup warm_ids=%s", warm_ids)
        warm_seeds = build_unique_seed_tokens(safe_ids, warm_ids, seed_length, seed + 6, tokenizer)
        logger.info("[prepare] warmup warm_seeds=%s", warm_seeds)
        for group in sorted(canonical):
            for rank in range(scenario.dp_size):
                request_id = f"warmup:{group}:{rank}"
                prompt, tokens, _, _ = _build_prompt_with_seed_retry(tokenizer, canonical[group], max_by_group[group], warm_seeds, request_id, [], max_by_group[group] + seed_length, safe_ids, seed_length, seed + 6)
                warmup_plan.append({"request_id": request_id, "group_id": group, "dp_rank": rank, "prompt": prompt, "input_tokens": len(tokens), "shared_prefix_tokens": max_by_group[group], "max_tokens": 1, "included_in_formal_statistics": False})
                logger.info("[prepare] warmup plan item=%s", warmup_plan[-1])
    # 阶段10：生成告警（目标不可达 / 命中率偏差）并写 analysis.json。
    signed_difference_pp = (theory.global_hit_rate - pc_cfg["target_hit_rate"]) * 100
    absolute_difference_pp = abs(signed_difference_pp)
    logger.info("[prepare] signed_difference_pp=%.4f absolute_difference_pp=%.4f", signed_difference_pp, absolute_difference_pp)
    warnings = []
    if not solve.target_reachable:
        warnings.append({
            "code": "TARGET_UNREACHABLE",
            "requested_target_hit_rate": pc_cfg["target_hit_rate"],
            "reachable_min": solve.min_reachable_rate,
            "reachable_max": solve.max_reachable_rate,
        })
    if absolute_difference_pp > effective["validation"]["target_warning_pp"]:
        warnings.append({"code": "TARGET_DEVIATION", "difference_pp": absolute_difference_pp})
    validation_status = "PASS" if not warnings else "PASS_WITH_WARNING"
    logger.info("[prepare] warnings=%s validation_status=%s", warnings, validation_status)
    analysis = {
        "schema_version": "1.0",
        "run_id": scenario.run_id,
        "status": "prepared",
        "requested_target_hit_rate": pc_cfg["target_hit_rate"],
        "effective_target_hit_rate": solve.effective_hit_rate,
        "theoretical_hit_rate": theory.global_hit_rate,
        "target_difference_pp": absolute_difference_pp,
        "target_signed_difference_pp": signed_difference_pp,
        "target_absolute_difference_pp": absolute_difference_pp,
        "validation": {
            "status": validation_status,
            "target_reachable": solve.target_reachable,
            "warning_only": True,
            "affects_exit_code": False,
        },
        "theory": {"input_tokens": theory.total_input_tokens, "hit_tokens": theory.total_hit_tokens, "groups": theory.group_stats, "dp": theory.dp_stats},
        "warnings": warnings,
    }
    logger.info("[prepare] analysis=%s", analysis)
    write_json(paths.analysis, analysis, overwrite)
    manifest_effective = copy.deepcopy(effective)
    configured_api_key = bool(manifest_effective["service"].pop("api_key", ""))
    manifest_effective["service"]["api_key_configured"] = configured_api_key
    logger.info("[prepare] configured_api_key=%s", configured_api_key)
    manifest = {
        "schema_version": "1.0",
        "plugin_version": __version__,
        "run_id": scenario.run_id,
        "scenario_path": str(scenario.source_path),
        "scenario_sha256": sha256_file(scenario.source_path),
        "effective_config": manifest_effective,
        "effective_config_sha256": _sha256_json(manifest_effective),
        "corpus_sha256": sha256_file(Path(corpus_cfg["path"])),
        "tokenizer": _tokenizer_manifest(tokenizer, effective, scenario.block_size),
        "requests": {
            "count": count,
            "total_input_tokens": theory.total_input_tokens,
            "input_length_summary": _length_summary(input_lengths),
            "output_length_summary": _length_summary(output_lengths),
        },
        "prefix_cache": {
            "mode": scenario.cache_mode,
            "requested_target_hit_rate": pc_cfg["target_hit_rate"],
            "effective_target_hit_rate": solve.effective_hit_rate,
            "theoretical_hit_rate": theory.global_hit_rate,
            "reachable_min": solve.min_reachable_rate,
            "reachable_max": solve.max_reachable_rate,
            "target_reachable": solve.target_reachable,
            "minimum_non_shared_length": minimum_non_shared_length,
            "adjusted": solve.adjusted,
            "reason": solve.reason,
            "validation_status": validation_status,
            "target_signed_difference_pp": signed_difference_pp,
            "target_absolute_difference_pp": absolute_difference_pp,
        },
        "groups": {
            group: {
                "canonical_prefix_sha256": item.sha256,
                "canonical_prefix_tokens": len(item.token_ids),
                "max_shared_prefix_tokens": max_by_group[group],
                "gsm_indices": list(item.gsm_indices),
                "gsm_question_sha256": list(item.gsm_hashes),
                "reachable_min": solve.group_reachability[group]["min_reachable_rate"],
                "reachable_max": solve.group_reachability[group]["max_reachable_rate"],
                "theoretical_hit_rate": theory.group_stats[group]["hit_rate"],
            }
            for group, item in canonical.items()
        },
        "dp": {"size": scenario.dp_size, "cold_route_strategy": "group_round_robin" if scenario.cache_mode == "cold" else None},
        "warmup": {"enabled": scenario.cache_mode == "warmup", "plan": warmup_plan},
        "divergence": {
            "strategy": "globally_unique_seed_block",
            "unique_request_blocks": len({row.seed_sha256 for row in theory.rows}),
            "request_count": count,
            "collision_status": "pass",
        },
        "artifacts": {
            "full": {"name": paths.full.name, "path": str(paths.full.resolve()), "rows": count, "bytes": paths.full.stat().st_size, "sha256": sha256_file(paths.full)},
            "requests": {"name": paths.requests.name, "path": str(paths.requests.resolve()), "rows": count, "bytes": paths.requests.stat().st_size, "sha256": sha256_file(paths.requests)},
            "analysis": {"name": paths.analysis.name, "path": str(paths.analysis.resolve()), "bytes": paths.analysis.stat().st_size, "sha256_at_prepare": sha256_file(paths.analysis)},
        },
    }
    logger.info("[prepare] manifest=%s", json.dumps(manifest, ensure_ascii=False))
    write_json(paths.manifest, manifest, overwrite)
    validate_artifacts(paths.manifest)
    logger.info("[prepare] prepare_scenario done paths=%s", {key: str(value) for key, value in paths.__dict__.items()})
    return paths


def inspect_scenario(path: Path | str, tokenizer_loader: Callable[[Scenario], Any] | None = None) -> dict[str, Any]:
    """Generate a read-only summary in a temporary directory without sending requests.

    只预览场景：在临时目录中改 run_id/output_dir 后复用 prepare 流程生成工件，
    但只返回统计摘要（分组分布、长度分布、可达命中率等），不发任何真实请求。
    """
    logger.info("[inspect] inspect_scenario path=%s tokenizer_loader=%s", path, tokenizer_loader)
    scenario = load_scenario(path)
    logger.info("[inspect] scenario run_id=%s cache_mode=%s dp_size=%d block_size=%d source_path=%s", scenario.run_id, scenario.cache_mode, scenario.dp_size, scenario.block_size, scenario.source_path)
    effective = scenario.to_effective_dict()
    logger.info("[inspect] effective keys=%s", sorted(effective))
    # tokenizer 相对路径且场景目录下存在本地副本时，改用本地路径，避免远程加载。
    tokenizer_path = Path(effective["tokenizer"]["path"])
    local_tokenizer = scenario.source_path.parent / tokenizer_path
    logger.info("[inspect] tokenizer_path=%s local_tokenizer=%s", tokenizer_path, local_tokenizer)
    if not tokenizer_path.is_absolute() and local_tokenizer.exists():
        effective["tokenizer"]["path"] = str(local_tokenizer.resolve())
        logger.info("[inspect] tokenizer path is relative and local copy exists -> resolved tokenizer path=%s", effective["tokenizer"]["path"])
    else:
        logger.info("[inspect] tokenizer path kept as-is: is_absolute=%s local_exists=%s", tokenizer_path.is_absolute(), local_tokenizer.exists())
    with tempfile.TemporaryDirectory(prefix="aisbench-prefix-cache-inspect-") as folder:
        logger.info("[inspect] temporary folder=%s", folder)
        root = Path(folder)
        # 改写 run_id/output_dir，让 prepare 的产物落到临时目录且不覆盖任何真实工件。
        effective["run"]["run_id"] = "inspect"
        effective["run"]["output_dir"] = str(root / "artifacts")
        effective["run"]["overwrite"] = False
        logger.info("[inspect] effective run overrides run_id=%s output_dir=%s overwrite=%s", effective["run"]["run_id"], effective["run"]["output_dir"], effective["run"]["overwrite"])
        temporary_scenario = root / "scenario.json"
        temporary_scenario.write_text(json.dumps(effective, ensure_ascii=False), encoding="utf-8")
        logger.info("[inspect] temporary_scenario=%s", temporary_scenario)
        paths = prepare_scenario(temporary_scenario, tokenizer_loader=tokenizer_loader)
        logger.info("[inspect] prepare_scenario paths=%s", {key: str(value) for key, value in paths.__dict__.items()})
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        logger.info("[inspect] manifest keys=%s run_id=%s plugin_version=%s", sorted(manifest), manifest["run_id"], manifest["plugin_version"])
        rows = read_jsonl(paths.full)
        logger.info("[inspect] rows=%d first_row=%s", len(rows), rows[0] if rows else None)
    group_counts: dict[str, int] = {}
    dp_counts: dict[str, int] = {}
    for row in rows:
        # 统计各分组与各 DP rank 的请求数量。
        group_counts[row["group_id"]] = group_counts.get(row["group_id"], 0) + 1
        if row["dp_rank"] is not None:
            key = str(row["dp_rank"])
            dp_counts[key] = dp_counts.get(key, 0) + 1
    logger.info("[inspect] group_counts=%s", group_counts)
    logger.info("[inspect] dp_counts=%s", dp_counts)
    input_lengths = [int(row["actual_input_tokens"]) for row in rows]
    output_lengths = [int(row["max_tokens"]) for row in rows]
    prefix = manifest["prefix_cache"]
    logger.info("[inspect] input_lengths=%s total=%d", input_lengths, sum(input_lengths))
    logger.info("[inspect] output_lengths=%s total=%d", output_lengths, sum(output_lengths))
    logger.info("[inspect] prefix manifest=%s", prefix)
    result = {
        "run_id": scenario.run_id,
        "mode": prefix["mode"],
        "requested_target_hit_rate": prefix["requested_target_hit_rate"],
        "effective_target_hit_rate": prefix["effective_target_hit_rate"],
        "theoretical_hit_rate": prefix["theoretical_hit_rate"],
        "reachable_min": prefix["reachable_min"],
        "reachable_max": prefix["reachable_max"],
        "target_reachable": prefix["target_reachable"],
        "group_reachability": {
            group: {"reachable_min": value["reachable_min"], "reachable_max": value["reachable_max"]}
            for group, value in manifest["groups"].items()
        },
        "groups": group_counts,
        "input_tokens": manifest["requests"]["input_length_summary"] | {"total": sum(input_lengths)},
        "output_tokens": manifest["requests"]["output_length_summary"] | {"total": sum(output_lengths)},
        "dp_route_counts": dp_counts,
        "sends_requests": False,
    }
    logger.info("[inspect] result=%s", json.dumps(result, ensure_ascii=False))
    return result
