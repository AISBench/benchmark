from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from .errors import ArtifactValidationError, ScenarioValidationError


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str: ...


@dataclass(frozen=True)
class GSMRecord:
    line_index: int
    question: str
    question_sha256: str


@dataclass(frozen=True)
class CanonicalPrefix:
    group_id: str
    text: str
    token_ids: tuple[int, ...]
    sha256: str
    gsm_indices: tuple[int, ...]
    gsm_hashes: tuple[str, ...]


@dataclass(frozen=True)
class RequestPlan:
    request_id: str
    sequence_index: int
    group_id: str
    occurrence_index_within_group: int
    dp_rank: int | None
    lane_sequence: int | None
    target_input_tokens: int
    actual_input_tokens: int
    max_tokens: int
    shared_prefix_tokens: int
    seed_tokens: int
    natural_suffix_tokens: int
    question: str = ""
    answer: str = "none"
    gsm_indices: tuple[int, ...] = ()
    gsm_hashes: tuple[str, ...] = ()
    canonical_prefix_sha256: str = ""
    seed_sha256: str = ""
    watermark_before: int = 0
    theoretical_hit_tokens: int = 0
    watermark_after: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TheorySummary:
    rows: tuple[RequestPlan, ...]
    total_input_tokens: int
    total_hit_tokens: int
    global_hit_rate: float
    group_stats: dict[str, dict[str, float | int]]
    dp_stats: dict[int, dict[str, float | int]]


@dataclass(frozen=True)
class SolveResult:
    shared_prefix_tokens: tuple[int, ...]
    requested_hit_tokens: int
    effective_hit_tokens: int
    effective_hit_rate: float
    min_reachable_rate: float
    max_reachable_rate: float
    adjusted: bool
    reason: str | None


def normalize_question(value: str) -> str:
    return " ".join(value.strip().split())


def load_gsm8k(path: Path, field: str = "question") -> list[GSMRecord]:
    records: list[GSMRecord] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ScenarioValidationError(f"cannot read GSM8K {path}: {exc}") from exc
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
    return records


def select_gsm8k(records: Sequence[GSMRecord], config: dict[str, Any], count: int, seed: int) -> list[GSMRecord]:
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
        return selected[:count]
    values = config.get("values")
    if values is None:
        values = config.get("indices" if mode == "indices" else "question_sha256", [])
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
        selected = []
        index_values = config.get("indices", [])
        hash_values = config.get("question_sha256", [])
        if index_values:
            selected.extend(select_gsm8k(records, {"mode": "indices", "values": index_values}, len(index_values), seed))
        if hash_values:
            selected.extend(select_gsm8k(records, {"mode": "question_sha256", "values": hash_values}, len(hash_values), seed))
    if not selected:
        raise ScenarioValidationError("specified GSM8K selection is empty")
    return [selected[i % len(selected)] for i in range(count)]


def _csv_values(path: str, aliases: Sequence[str]) -> list[int]:
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    fieldnames = rows[0].keys() if rows else []
    column = next((name for name in aliases if name in fieldnames), None)
    if column is None:
        raise ScenarioValidationError(f"CSV requires one of columns {list(aliases)}")
    values = [int(row[column]) for row in rows]
    if not values or any(value < 1 for value in values):
        raise ScenarioValidationError(f"CSV column {column} must contain positive integers")
    return values


def build_input_lengths(config: dict[str, Any], count: int, seed: int) -> list[int]:
    mode = config["mode"]
    if mode == "fixed":
        return [int(config["value"])] * count
    if mode == "csv":
        values = _csv_values(config["path"], ("input_prompt_tokens", "content_tokens", "input_tokens"))
        if len(values) != count:
            raise ScenarioValidationError("input CSV row count must equal requests.count")
        return values
    rng = random.Random(seed)
    values = [rng.randint(int(item["min"]), int(item["max"])) for item in config["ranges"] for _ in range(int(item["count"]))]
    return values


def build_output_lengths(config: dict[str, Any], count: int, seed: int) -> list[int]:
    mode = config["mode"]
    if mode == "fixed":
        return [int(config["value"])] * count
    if mode == "csv":
        values = _csv_values(config["path"], ("output_tokens",))
        if len(values) != count:
            raise ScenarioValidationError("output CSV row count must equal requests.count")
        return values
    low, high = int(config["min"]), int(config["max"])
    rng = random.Random(seed)
    if mode == "uniform":
        return [rng.randint(low, high) for _ in range(count)]
    if low == high:
        return [low] * count
    mean = float(config.get("mean", (low + high) / 2))
    std = float(config.get("std", max(1.0, (high - low) / 4)))
    values: list[int] = []
    attempts = 0
    while len(values) < count and attempts < max(1000, count * 100):
        value = int(round(rng.gauss(mean, std)))
        if low <= value <= high:
            values.append(value)
        attempts += 1
    if len(values) != count:
        raise ScenarioValidationError("truncated_normal could not produce enough values")
    return values


def assign_groups(count: int, config: dict[str, Any], seed: int) -> list[str]:
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
    total = sum(weights)
    quotas = [count * value / total for value in weights]
    allocations = [math.floor(value) for value in quotas]
    remaining = count - sum(allocations)
    order = sorted(range(group_count), key=lambda index: (-(quotas[index] - allocations[index]), index))
    for index in order[:remaining]:
        allocations[index] += 1
    groups = [f"group-{index}" for index, amount in enumerate(allocations) for _ in range(amount)]
    if mode == "zipf":
        random.Random(seed).shuffle(groups)
    return groups


def order_indices(group_ids: Sequence[str], strategy: str, seed: int) -> list[int]:
    indices = list(range(len(group_ids)))
    rng = random.Random(seed)
    if strategy == "sequential":
        return indices
    if strategy == "global_shuffle":
        rng.shuffle(indices)
        return indices
    buckets: dict[str, list[int]] = {}
    for index, group in enumerate(group_ids):
        buckets.setdefault(group, []).append(index)
    if strategy == "within_group_shuffle":
        result: list[int] = []
        for group in sorted(buckets):
            rng.shuffle(buckets[group])
            result.extend(buckets[group])
        return result
    result = []
    for row in itertools.zip_longest(*(buckets[group] for group in sorted(buckets))):
        result.extend(index for index in row if index is not None)
    return result


def assign_cold_routes(group_ids: Sequence[str], dp_size: int, explicit: Sequence[int] | None = None) -> tuple[list[int], list[int]]:
    if explicit is not None:
        if len(explicit) != len(group_ids) or any(rank < 0 or rank >= dp_size for rank in explicit):
            raise ScenarioValidationError("explicit DP routes are invalid")
        ranks = list(explicit)
    else:
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
    return ranks, lane_sequences


def simulate_theory(plans: Sequence[RequestPlan], mode: str, warmup_watermarks: dict[str, int] | None = None) -> TheorySummary:
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
        group_totals.setdefault(plan.group_id, [0, 0])
        group_totals[plan.group_id][0] += plan.actual_input_tokens
        group_totals[plan.group_id][1] += hit
        if plan.dp_rank is not None:
            dp_totals.setdefault(plan.dp_rank, [0, 0])
            dp_totals[plan.dp_rank][0] += plan.actual_input_tokens
            dp_totals[plan.dp_rank][1] += hit
    total_input = sum(row.actual_input_tokens for row in rows)
    total_hit = sum(row.theoretical_hit_tokens for row in rows)
    stats = lambda values: {"input_tokens": values[0], "hit_tokens": values[1], "hit_rate": values[1] / values[0] if values[0] else 0.0}
    return TheorySummary(tuple(rows), total_input, total_hit, total_hit / total_input if total_input else 0.0, {key: stats(value) for key, value in group_totals.items()}, {key: stats(value) for key, value in dp_totals.items()})


def _plans_for_prefixes(input_lengths: Sequence[int], output_lengths: Sequence[int], group_ids: Sequence[str], ranks: Sequence[int | None], lane_sequences: Sequence[int | None], prefixes: Sequence[int]) -> list[RequestPlan]:
    occurrences: dict[str, int] = {}
    plans = []
    for index, (length, out_len, group, rank, lane_seq, prefix) in enumerate(zip(input_lengths, output_lengths, group_ids, ranks, lane_sequences, prefixes)):
        occurrence = occurrences.get(group, 0)
        occurrences[group] = occurrence + 1
        plans.append(RequestPlan(f"request-{index:08d}", index, group, occurrence, rank, lane_seq, length, length, out_len, prefix, 0, length - prefix))
    return plans


def solve_prefix_lengths(input_lengths: Sequence[int], output_lengths: Sequence[int], group_ids: Sequence[str], ranks: Sequence[int | None], lane_sequences: Sequence[int | None], block_size: int, seed_tokens: int, mode: str, target_hit_rate: float) -> SolveResult:
    candidates = [list(range(0, max(0, ((length - seed_tokens) // block_size) * block_size) + 1, block_size)) for length in input_lengths]
    total_input = sum(input_lengths)
    target_tokens = int(total_input * target_hit_rate + 0.5)

    def score(prefixes: Sequence[int]) -> tuple[int, int]:
        plans = _plans_for_prefixes(input_lengths, output_lengths, group_ids, ranks, lane_sequences, prefixes)
        warm = {group: max((prefix for prefix, current in zip(prefixes, group_ids) if current == group), default=0) for group in set(group_ids)} if mode == "warmup" else None
        hit = simulate_theory(plans, mode, warm).total_hit_tokens
        return abs(hit - target_tokens), hit

    candidate_space = math.prod(len(values) for values in candidates)
    if candidate_space <= 200_000:
        chosen = min(
            itertools.product(*candidates),
            key=lambda trial: (score(trial)[0], abs(sum(trial) - target_tokens), tuple(trial)),
        )
        prefixes = list(chosen)
        best_error, best_hit = score(prefixes)
    elif mode == "warmup":
        desired = min(sum(values[-1] for values in candidates), max(0, int(round(target_tokens / block_size)) * block_size))
        prefixes = [0] * len(candidates)
        remaining = desired
        for index in reversed(range(len(candidates))):
            value = min(candidates[index][-1], (remaining // block_size) * block_size)
            prefixes[index] = value
            remaining -= value
        best_error, best_hit = score(prefixes)
    else:
        prefixes = [min(values, key=lambda value: abs(value - length * target_hit_rate)) for values, length in zip(candidates, input_lengths)]
        best_error, best_hit = score(prefixes)
        lanes: dict[tuple[str, int], list[int]] = {}
        for index, (group, rank) in enumerate(zip(group_ids, ranks)):
            lanes.setdefault((group, int(rank or 0)), []).append(index)
        changed = True
        while changed:
            changed = False
            best_move = None
            moves: list[tuple[int, ...]] = [(index,) for index in range(len(prefixes))]
            moves.extend((left, right) for lane in lanes.values() for left, right in zip(lane, lane[1:]))
            for move in moves:
                for direction in (-1, 1):
                    trial = prefixes.copy()
                    valid = True
                    for index in move:
                        values = candidates[index]
                        next_pos = values.index(trial[index]) + direction
                        if not 0 <= next_pos < len(values):
                            valid = False
                            break
                        trial[index] = values[next_pos]
                    if not valid:
                        continue
                    error, hit = score(trial)
                    key = (error, len(move), tuple(move), tuple(trial))
                    if error < best_error and (best_move is None or key < best_move[0]):
                        best_move = (key, trial, hit)
            if best_move is not None:
                _, prefixes, best_hit = best_move
                best_error = abs(best_hit - target_tokens)
                changed = True
    zero_hit = score([0] * len(prefixes))[1]
    max_prefixes = [values[-1] for values in candidates]
    max_hit = score(max_prefixes)[1]
    effective_rate = best_hit / total_input if total_input else 0.0
    adjusted = best_hit != target_tokens
    return SolveResult(tuple(prefixes), target_tokens, best_hit, effective_rate, zero_hit / total_input if total_input else 0.0, max_hit / total_input if total_input else 0.0, adjusted, "block alignment and cache-watermark constraints" if adjusted else None)


def find_boundary_safe_token_ids(tokenizer: TokenizerLike, minimum: int) -> list[int]:
    vocab_size = len(tokenizer)  # type: ignore[arg-type]
    special = set(getattr(tokenizer, "all_special_ids", []))
    safe: list[int] = []
    for token_id in range(vocab_size):
        if token_id in special:
            continue
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        standalone = tokenizer.encode(text, add_special_tokens=False) == [token_id]
        contextual = tokenizer.encode("X" + text, add_special_tokens=False)[-1:] == [token_id]
        if standalone and (text.startswith(" ") or contextual):
            safe.append(token_id)
            if len(safe) >= minimum:
                return safe
    raise ArtifactValidationError(f"tokenizer has only {len(safe)} boundary-safe tokens; need {minimum}")


def build_unique_seed_tokens(safe_ids: Sequence[int], request_ids: Sequence[str], seed_length: int, random_seed: int) -> dict[str, tuple[int, ...]]:
    if seed_length < 1 or len(safe_ids) < 2:
        raise ArtifactValidationError("seed generation requires positive length and at least two safe tokens")
    result: dict[str, tuple[int, ...]] = {}
    used: set[tuple[int, ...]] = set()
    for request_id in request_ids:
        nonce = 0
        while True:
            digest = hashlib.sha256(f"{random_seed}:{request_id}:{nonce}".encode()).digest()
            stream = itertools.cycle(digest)
            seed = tuple(safe_ids[next(stream) % len(safe_ids)] for _ in range(seed_length))
            if seed not in used:
                used.add(seed)
                result[request_id] = seed
                break
            nonce += 1
            if nonce > len(request_ids) * 10 + 100:
                raise ArtifactValidationError("unable to construct globally unique seeds")
    return result


def _repeat_tokens(records: Sequence[GSMRecord], tokenizer: TokenizerLike, target: int) -> tuple[list[int], tuple[int, ...], tuple[str, ...]]:
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
            return tokens[:target], tuple(indices), tuple(hashes)
    raise ArtifactValidationError("cannot build tokens from empty GSM8K records")


def build_canonical_prefixes(tokenizer: TokenizerLike, group_sources: dict[str, Sequence[GSMRecord]], max_lengths: dict[str, int], block_size: int) -> dict[str, CanonicalPrefix]:
    result: dict[str, CanonicalPrefix] = {}
    first_blocks: set[tuple[int, ...]] = set()
    for group_position, group in enumerate(sorted(group_sources)):
        source_records = list(group_sources[group])
        if not source_records:
            raise ArtifactValidationError(f"canonical prefix source is empty for {group}")
        token_ids = indices = hashes = None
        for offset in range(len(source_records)):
            rotated = source_records[offset:] + source_records[:offset]
            candidate_tokens, candidate_indices, candidate_hashes = _repeat_tokens(
                rotated, tokenizer, max(max_lengths[group], block_size)
            )
            if tuple(candidate_tokens[:block_size]) not in first_blocks:
                token_ids, indices, hashes = candidate_tokens, candidate_indices, candidate_hashes
                break
        if token_ids is None:
            # Explicitly duplicated corpus selections can make every source rotation
            # identical. Add a deterministic group marker only in that collision case
            # so one bad override cannot abort the whole dataset generation.
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
        if actual[:max_lengths[group]] != token_ids[:max_lengths[group]]:
            raise ArtifactValidationError(f"canonical prefix does not round-trip for {group}")
        digest = hashlib.sha256(bytes(str(token_ids), "utf-8")).hexdigest()
        result[group] = CanonicalPrefix(group, text, tuple(token_ids), digest, indices, hashes)
    return result


def build_prompt(tokenizer: TokenizerLike, canonical: CanonicalPrefix, shared_prefix_tokens: int, seed: Sequence[int], suffix_records: Sequence[GSMRecord], target_tokens: int) -> tuple[str, tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    suffix_len = target_tokens - shared_prefix_tokens - len(seed)
    if suffix_len < 0:
        raise ArtifactValidationError("prefix and seed exceed target input length")
    suffix, indices, hashes = _repeat_tokens(suffix_records, tokenizer, suffix_len) if suffix_len else ([], (), ())
    expected = list(canonical.token_ids[:shared_prefix_tokens]) + list(seed) + suffix
    text = tokenizer.decode(expected, skip_special_tokens=False)
    actual = tokenizer.encode(text, add_special_tokens=False)
    if actual != expected:
        raise ArtifactValidationError("prompt token layout changed after decode/re-encode")
    return text, tuple(actual), indices, hashes
