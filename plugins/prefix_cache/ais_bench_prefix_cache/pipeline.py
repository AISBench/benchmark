from __future__ import annotations

import copy
import hashlib
import json
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
from .scenario import Scenario, load_scenario


def _tokenizer_loader(scenario: Scenario):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ArtifactValidationError("transformers is required to load the configured tokenizer") from exc
    cfg = scenario.section("tokenizer")
    return AutoTokenizer.from_pretrained(cfg["path"], revision=cfg.get("revision"), trust_remote_code=cfg.get("trust_remote_code", False))


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_manifest(tokenizer: Any, effective: dict[str, Any], block_size: int) -> dict[str, Any]:
    special_ids = sorted(int(value) for value in getattr(tokenizer, "all_special_ids", []))
    fingerprint_source = {
        "path": effective["tokenizer"]["path"],
        "revision": effective["tokenizer"].get("revision"),
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "vocab_size": len(tokenizer),
        "special_token_ids": special_ids,
    }
    return fingerprint_source | {
        "block_size": block_size,
        "fingerprint_sha256": _sha256_json(fingerprint_source),
    }


def _build_prompt_with_seed_retry(tokenizer: Any, canonical: Any, prefix_len: int, seeds: dict[str, tuple[int, ...]], request_id: str, rotated_pool: list[Any], target_tokens: int, safe_ids: list[int], seed_length: int, random_seed: int):
    for attempt in range(64):
        try:
            return build_prompt(tokenizer, canonical, prefix_len, seeds[request_id], rotated_pool, target_tokens)
        except PromptRoundTripError:
            seeds[request_id] = build_unique_seed(tokenizer, safe_ids, request_id, seed_length, random_seed + attempt * 10007 + 1, set(seeds.values()))
    raise ArtifactValidationError(f"unable to construct a round-trip-safe prompt for {request_id}")


def prepare_scenario(path: Path | str, overwrite: bool | None = None, tokenizer_loader: Callable[[Scenario], Any] | None = None) -> ArtifactPaths:
    scenario = load_scenario(path)
    effective = scenario.to_effective_dict()
    overwrite = effective["run"].get("overwrite", False) if overwrite is None else overwrite
    request_cfg = effective["requests"]
    pc_cfg = effective["prefix_cache"]
    corpus_cfg = effective["corpus"]
    count = request_cfg["count"]
    seed = scenario.random_seed
    input_lengths = build_input_lengths(request_cfg["input_length"], count, seed)
    output_lengths = build_output_lengths(request_cfg["output_length"], count, seed + 1)
    groups = assign_groups(count, pc_cfg["groups"], seed + 2)
    records = load_gsm8k(Path(corpus_cfg["path"]), corpus_cfg["field"])
    overrides = pc_cfg["groups"].get("overrides", {})
    group_pools: dict[str, list[Any]] = {}
    for group_index, group in enumerate(sorted(set(groups))):
        group_positions = [index for index, value in enumerate(groups) if value == group]
        override = overrides.get(group, {})
        if "input_length" in override:
            values = build_input_lengths(override["input_length"], len(group_positions), seed + 100 + group_index)
            if len(values) != len(group_positions):
                raise ArtifactValidationError(f"{group} input_length generated {len(values)} values; expected {len(group_positions)}")
            for position, value in zip(group_positions, values):
                input_lengths[position] = value
        if "output_length" in override:
            values = build_output_lengths(override["output_length"], len(group_positions), seed + 200 + group_index)
            for position, value in zip(group_positions, values):
                output_lengths[position] = value
        selection = override.get("corpus_selection", corpus_cfg["selection"])
        group_pools[group] = select_gsm8k(records, selection, max(2, len(group_positions)), seed + 300 + group_index)
    ordering = order_indices(groups, pc_cfg["order"]["strategy"], seed + 4)
    input_lengths = [input_lengths[index] for index in ordering]
    output_lengths = [output_lengths[index] for index in ordering]
    groups = [groups[index] for index in ordering]
    if scenario.cache_mode == "cold":
        ranks_raw, lane_raw = assign_cold_routes(groups, scenario.dp_size)
        ranks: list[int | None] = ranks_raw
        lanes: list[int | None] = lane_raw
    else:
        ranks = [None] * count
        lanes = [None] * count
    seed_length = scenario.block_size * pc_cfg["seed_blocks"]
    solve = solve_prefix_lengths(input_lengths, output_lengths, groups, ranks, lanes, scenario.block_size, seed_length, scenario.cache_mode, pc_cfg["target_hit_rate"])
    max_by_group = {group: max((prefix for prefix, current in zip(solve.shared_prefix_tokens, groups) if current == group), default=0) for group in sorted(set(groups))}
    group_sources = {group: group_pools[group] for group in sorted(set(groups))}
    tokenizer = (tokenizer_loader or _tokenizer_loader)(scenario)
    canonical = build_canonical_prefixes(tokenizer, group_sources, max_by_group, scenario.block_size)
    safe_ids = find_boundary_safe_token_ids(tokenizer, max(2, min(64, len(tokenizer))))
    request_ids = [f"request-{index:08d}" for index in range(count)]
    seeds = build_unique_seed_tokens(safe_ids, request_ids, seed_length, seed + 5, tokenizer)
    plans: list[RequestPlan] = []
    occurrences: dict[str, int] = {}
    for index, request_id in enumerate(request_ids):
        group = groups[index]
        occurrence = occurrences.get(group, 0)
        occurrences[group] = occurrence + 1
        prefix_len = solve.shared_prefix_tokens[index]
        pool = group_pools[group]
        rotated_pool = pool[occurrence % len(pool):] + pool[:occurrence % len(pool)]
        text, tokens, suffix_indices, suffix_hashes = _build_prompt_with_seed_retry(tokenizer, canonical[group], prefix_len, seeds, request_id, rotated_pool, input_lengths[index], safe_ids, seed_length, seed + 5)
        seed_hash = hashlib.sha256(str(seeds[request_id]).encode()).hexdigest()
        plans.append(RequestPlan(
            request_id, index, group, occurrence, ranks[index], lanes[index], input_lengths[index], len(tokens), output_lengths[index], prefix_len, seed_length, len(tokens) - prefix_len - seed_length,
            text, "none", suffix_indices, suffix_hashes, canonical[group].sha256, seed_hash,
        ))
    warm_watermarks = max_by_group if scenario.cache_mode == "warmup" else None
    theory = simulate_theory(plans, scenario.cache_mode, warm_watermarks)
    full_rows = [row.to_dict() | {"theoretical_hit_rate": row.theoretical_hit_tokens / row.actual_input_tokens if row.actual_input_tokens else 0.0} for row in theory.rows]
    request_rows = [{"question": row.question, "answer": row.answer, "max_tokens": row.max_tokens} for row in theory.rows]
    paths = artifact_paths(scenario.output_dir, scenario.run_id)
    write_jsonl(paths.full, full_rows, overwrite)
    write_jsonl(paths.requests, request_rows, overwrite)
    warmup_plan = []
    if scenario.cache_mode == "warmup":
        warm_ids = [f"warmup:{group}:{rank}" for group in sorted(canonical) for rank in range(scenario.dp_size)]
        warm_seeds = build_unique_seed_tokens(safe_ids, warm_ids, seed_length, seed + 6, tokenizer)
        for group in sorted(canonical):
            for rank in range(scenario.dp_size):
                request_id = f"warmup:{group}:{rank}"
                prompt, tokens, _, _ = _build_prompt_with_seed_retry(tokenizer, canonical[group], max_by_group[group], warm_seeds, request_id, [], max_by_group[group] + seed_length, safe_ids, seed_length, seed + 6)
                warmup_plan.append({"request_id": request_id, "group_id": group, "dp_rank": rank, "prompt": prompt, "input_tokens": len(tokens), "shared_prefix_tokens": max_by_group[group], "max_tokens": 1, "included_in_formal_statistics": False})
    analysis = {
        "schema_version": "1.0",
        "run_id": scenario.run_id,
        "status": "prepared",
        "requested_target_hit_rate": pc_cfg["target_hit_rate"],
        "effective_target_hit_rate": solve.effective_hit_rate,
        "theoretical_hit_rate": theory.global_hit_rate,
        "target_difference_pp": abs(theory.global_hit_rate - pc_cfg["target_hit_rate"]) * 100,
        "theory": {"input_tokens": theory.total_input_tokens, "hit_tokens": theory.total_hit_tokens, "groups": theory.group_stats, "dp": theory.dp_stats},
        "runtime": {},
        "warnings": ([{"code": "TARGET_DEVIATION", "difference_pp": abs(theory.global_hit_rate - pc_cfg["target_hit_rate"]) * 100}] if abs(theory.global_hit_rate - pc_cfg["target_hit_rate"]) * 100 > effective["validation"]["target_warning_pp"] else []),
    }
    write_json(paths.analysis, analysis, overwrite)
    manifest_effective = copy.deepcopy(effective)
    configured_api_key = bool(manifest_effective["service"].pop("api_key", ""))
    manifest_effective["service"]["api_key_configured"] = configured_api_key
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
        "requests": {"count": count, "total_input_tokens": theory.total_input_tokens},
        "prefix_cache": {"mode": scenario.cache_mode, "requested_target_hit_rate": pc_cfg["target_hit_rate"], "effective_target_hit_rate": solve.effective_hit_rate, "theoretical_hit_rate": theory.global_hit_rate, "reachable_min": solve.min_reachable_rate, "reachable_max": solve.max_reachable_rate, "adjusted": solve.adjusted, "reason": solve.reason},
        "groups": {
            group: {
                "canonical_prefix_sha256": item.sha256,
                "canonical_prefix_tokens": len(item.token_ids),
                "max_shared_prefix_tokens": max_by_group[group],
                "gsm_indices": list(item.gsm_indices),
                "gsm_question_sha256": list(item.gsm_hashes),
            }
            for group, item in canonical.items()
        },
        "dp": {"size": scenario.dp_size, "cold_route_strategy": "group_round_robin" if scenario.cache_mode == "cold" else None},
        "warmup": {"enabled": scenario.cache_mode == "warmup", "plan": warmup_plan},
        "artifacts": {
            "full": {"name": paths.full.name, "path": str(paths.full.resolve()), "rows": count, "bytes": paths.full.stat().st_size, "sha256": sha256_file(paths.full)},
            "requests": {"name": paths.requests.name, "path": str(paths.requests.resolve()), "rows": count, "bytes": paths.requests.stat().st_size, "sha256": sha256_file(paths.requests)},
            "analysis": {"name": paths.analysis.name, "path": str(paths.analysis.resolve()), "bytes": paths.analysis.stat().st_size, "sha256_at_prepare": sha256_file(paths.analysis)},
        },
    }
    write_json(paths.manifest, manifest, overwrite)
    validate_artifacts(paths.manifest)
    return paths


def inspect_scenario(path: Path | str, tokenizer_loader: Callable[[Scenario], Any] | None = None) -> dict[str, Any]:
    """Generate a read-only summary in a temporary directory without sending requests."""
    scenario = load_scenario(path)
    effective = scenario.to_effective_dict()
    tokenizer_path = Path(effective["tokenizer"]["path"])
    local_tokenizer = scenario.source_path.parent / tokenizer_path
    if not tokenizer_path.is_absolute() and local_tokenizer.exists():
        effective["tokenizer"]["path"] = str(local_tokenizer.resolve())
    with tempfile.TemporaryDirectory(prefix="aisbench-prefix-cache-inspect-") as folder:
        root = Path(folder)
        effective["run"]["run_id"] = "inspect"
        effective["run"]["output_dir"] = str(root / "artifacts")
        effective["run"]["overwrite"] = False
        temporary_scenario = root / "scenario.json"
        temporary_scenario.write_text(json.dumps(effective, ensure_ascii=False), encoding="utf-8")
        paths = prepare_scenario(temporary_scenario, tokenizer_loader=tokenizer_loader)
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        rows = read_jsonl(paths.full)
    group_counts: dict[str, int] = {}
    dp_counts: dict[str, int] = {}
    for row in rows:
        group_counts[row["group_id"]] = group_counts.get(row["group_id"], 0) + 1
        if row["dp_rank"] is not None:
            key = str(row["dp_rank"])
            dp_counts[key] = dp_counts.get(key, 0) + 1
    input_lengths = [int(row["actual_input_tokens"]) for row in rows]
    output_lengths = [int(row["max_tokens"]) for row in rows]
    prefix = manifest["prefix_cache"]
    return {
        "run_id": scenario.run_id,
        "mode": prefix["mode"],
        "requested_target_hit_rate": prefix["requested_target_hit_rate"],
        "effective_target_hit_rate": prefix["effective_target_hit_rate"],
        "theoretical_hit_rate": prefix["theoretical_hit_rate"],
        "reachable_min": prefix["reachable_min"],
        "reachable_max": prefix["reachable_max"],
        "groups": group_counts,
        "input_tokens": {"min": min(input_lengths), "max": max(input_lengths), "total": sum(input_lengths)},
        "output_tokens": {"min": min(output_lengths), "max": max(output_lengths), "total": sum(output_lengths)},
        "dp_route_counts": dp_counts,
        "sends_requests": False,
    }
