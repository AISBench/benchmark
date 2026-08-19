# AISBench Prefix Cache Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independently installable AISBench plugin that generates target-driven GSM8K Prefix Cache datasets, runs cold or warmup tests against one vLLM endpoint with multi-DP support, and persists auditable theoretical and actual hit-rate results.

**Architecture:** All new production code lives under `plugins/prefix_cache`; AISBench core files remain unchanged. A deterministic domain layer prepares four artifacts, AISBench plugin classes add route metadata and per-request DP headers, and a runtime controller surrounds the AISBench subprocess with vLLM reset/warmup/metrics phases.

**Tech Stack:** Python 3.10+, setuptools entry points, dataclasses, Hugging Face Transformers tokenizer, Hugging Face Datasets, aiohttp/requests, prometheus-client parser, pytest/unittest.mock.

## Global Constraints

- Do not modify existing files under `ais_bench/benchmark`; add an independently installable plugin only.
- Support one `inference_url`, one `metrics_url`, optional one `reset_url`, and internal `dp_size >= 1`; reject multi-instance configuration.
- Support fixed/range/CSV input lengths and fixed/uniform/truncated_normal/CSV output lengths.
- Use GSM8K `question` only; selection supports deterministic random, zero-based line index, and normalized-question SHA-256.
- Each Prefix Group owns one canonical prefix and independent cache watermarks.
- Place a deterministic, globally unique, boundary-safe seed between shared prefix and natural suffix; default seed length is one block.
- `target_hit_rate` is primary; unreachable targets use the nearest reachable rate and record requested/effective values and reason.
- Support `sequential`, `within_group_shuffle`, `interleave`, and `global_shuffle`; `interleave` is the default.
- Support cold-start and warmup. Warmup requests never enter AISBench formal requests or formal metric/performance denominators.
- For multi-DP warmup, warm every `Prefix Group × DP rank` using `X-data-parallel-rank`.
- For multi-DP cold-start, route every formal request deterministically and maintain watermarks per `(group_id, dp_rank)` while preserving order within each lane.
- Prefer vLLM V1 metrics and support named legacy aliases. Aggregate global actual hit rate from summed hit/query token deltas, never by averaging percentages.
- Theory/target deviations over 1 pp and theory/actual deviations over 5 pp are warnings only and never change an otherwise successful exit code.
- Persist `<run_id>.full.jsonl`, `<run_id>.requests.jsonl`, `<run_id>.manifest.json`, and `<run_id>.analysis.json` using atomic writes.
- Use TDD for every task. Do not commit `.superpowers/`, generated output data, caches, or credentials.

## File Map

```text
plugins/prefix_cache/
├── README.md
├── setup.py
├── config_examples/
│   ├── prefix_cache_perf.py
│   └── scenario.example.json
├── ais_bench_prefix_cache/
│   ├── __init__.py
│   ├── cli.py
│   ├── errors.py
│   ├── records.py
│   ├── scenario.py
│   ├── config.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── lengths.py
│   │   ├── corpus.py
│   │   ├── grouping.py
│   │   ├── tokens.py
│   │   ├── theory.py
│   │   ├── solver.py
│   │   ├── artifacts.py
│   │   └── pipeline.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── vllm.py
│   │   └── orchestrator.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── prefix_cache_dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── vllm_prefix_cache_api.py
│   └── openicl/icl_inferencer/
│       ├── __init__.py
│       └── prefix_cache_gen_inferencer.py
└── tests/
    ├── conftest.py
    ├── test_scenario.py
    ├── test_lengths_corpus_grouping.py
    ├── test_tokens.py
    ├── test_theory_solver.py
    ├── test_artifacts_pipeline.py
    ├── test_metrics_vllm.py
    ├── test_aisbench_integration.py
    └── test_orchestrator_cli.py
```

---

### Task 1: Plugin scaffold, typed records, and strict scenario loading

**Files:**
- Create: `plugins/prefix_cache/setup.py`
- Create: `plugins/prefix_cache/README.md`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/__init__.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/errors.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/records.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/scenario.py`
- Create: `plugins/prefix_cache/tests/conftest.py`
- Create: `plugins/prefix_cache/tests/test_scenario.py`

**Interfaces:**
- Produces: `PrefixCacheError`, `ScenarioValidationError`, `ArtifactValidationError`, `RuntimeCapabilityError`.
- Produces: immutable config dataclasses `RunConfig`, `TokenizerConfig`, `CorpusConfig`, `RequestConfig`, `PrefixCacheConfig`, `ServiceConfig`, `ValidationConfig`, and `Scenario`.
- Produces: immutable domain dataclasses `GSMRecord`, `CanonicalPrefix`, `PromptBuildResult`, `RequestPlan`, `TheoryResult`, `TheorySummary`, `SolveProblem`, `SolveResult`, `ArtifactPaths`, `ValidationReport`, `RankMetrics`, `MetricSnapshot`, `ActualMetrics`, `PhaseResult`, and `AnalysisResult`.
- Produces: `load_scenario(path: Path) -> Scenario` and `Scenario.to_effective_dict() -> dict[str, object]`.
- Test support in `conftest.py`: `FakeTokenizer`, fixtures `fake_tokenizer`, `repo_root`, and helpers `read_jsonl(path)`, `write_jsonl(path, rows)`, `write_scenario_and_gsm(tmp_path, *, mode="cold", dp_size=2)`, and `write_three_consistent_artifacts(tmp_path)`.

- [ ] **Step 1: Write failing setup and strict-config tests**

```python
# plugins/prefix_cache/tests/test_scenario.py
import json
from pathlib import Path
import pytest

from ais_bench_prefix_cache.scenario import load_scenario
from ais_bench_prefix_cache.errors import ScenarioValidationError


def minimal_scenario(tmp_path: Path) -> dict:
    return {
        "schema_version": "1.0",
        "run": {"run_id": "pc-test", "random_seed": 42, "output_dir": str(tmp_path)},
        "tokenizer": {"path": "fake-tokenizer", "block_size": 16},
        "corpus": {"path": str(tmp_path / "gsm8k.jsonl"), "field": "question", "selection": {"mode": "random"}},
        "requests": {"count": 4, "input_length": {"mode": "fixed", "value": 64}, "output_length": {"mode": "fixed", "value": 8}},
        "prefix_cache": {"mode": "cold", "target_hit_rate": 0.5, "seed_blocks": 1, "groups": {"count": 2, "assignment": {"mode": "uniform"}}, "order": {"strategy": "interleave"}},
        "service": {"inference_url": "http://127.0.0.1:8000/v1/completions", "metrics_url": "http://127.0.0.1:8000/metrics", "reset_url": "http://127.0.0.1:8000/reset_prefix_cache", "model": "m", "dp_size": 2, "assume_empty_cache": False},
        "validation": {"target_warning_pp": 1.0, "actual_warning_pp": 5.0},
    }


def test_load_scenario_expands_defaults(tmp_path):
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(minimal_scenario(tmp_path)), encoding="utf-8")
    scenario = load_scenario(path)
    assert scenario.run.run_id == "pc-test"
    assert scenario.prefix_cache.order.strategy == "interleave"
    assert scenario.service.dp_size == 2


def test_unknown_field_is_rejected(tmp_path):
    data = minimal_scenario(tmp_path)
    data["service"]["instances"] = ["forbidden"]
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ScenarioValidationError, match="service.instances"):
        load_scenario(path)
```

- [ ] **Step 2: Run tests and verify the missing package failure**

Run: `python -m pytest plugins/prefix_cache/tests/test_scenario.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'ais_bench_prefix_cache'`.

- [ ] **Step 3: Add packaging metadata and record types**

```python
# plugins/prefix_cache/setup.py
from setuptools import find_packages, setup

setup(
    name="ais-bench-prefix-cache",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["ais-bench-benchmark", "prometheus-client>=0.20", "requests>=2.31", "transformers"],
    entry_points={
        "ais_bench.benchmark_plugins": ["prefix_cache = ais_bench_prefix_cache"],
        "console_scripts": ["ais-bench-prefix-cache = ais_bench_prefix_cache.cli:main"],
    },
)
```

```python
# plugins/prefix_cache/ais_bench_prefix_cache/errors.py
class PrefixCacheError(Exception):
    """Base error with a stable user-facing message."""


class ScenarioValidationError(PrefixCacheError):
    pass


class ArtifactValidationError(PrefixCacheError):
    pass


class RuntimeCapabilityError(PrefixCacheError):
    pass
```

Define `RequestPlan` in `records.py` with these exact fields: `request_id`, `sequence_index`, `group_id`, `occurrence_index_within_group`, `dp_rank`, `lane_sequence`, `target_input_tokens`, `actual_input_tokens`, `max_tokens`, `shared_prefix_tokens`, `seed_tokens`, `natural_suffix_tokens`, `question`, `gsm_indices`, `gsm_hashes`, `canonical_prefix_sha256`, `seed_sha256`, `watermark_before`, `theoretical_hit_tokens`, and `watermark_after`.

Define the remaining shared record shapes exactly as follows:

```python
@dataclass(frozen=True)
class CanonicalPrefix:
    group_id: str
    text: str
    token_ids: tuple[int, ...]
    sha256: str
    gsm_indices: tuple[int, ...]
    gsm_hashes: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactPaths:
    full: Path
    requests: Path
    manifest: Path
    analysis: Path


@dataclass(frozen=True)
class RankMetrics:
    queries: int
    hits: int
    kv_cache_usage: float | None


@dataclass(frozen=True)
class MetricSnapshot:
    by_rank: dict[int, RankMetrics]
    metric_names: dict[str, str]
    raw_text: str


@dataclass(frozen=True)
class ActualMetrics:
    by_rank: dict[int, RankMetrics]
    global_queries: int
    global_hits: int
    global_hit_rate: float | None
```

`TheorySummary`, `SolveResult`, and `AnalysisResult` expose token totals and rates as typed attributes rather than unstructured dictionaries; their `to_dict()` methods are the only serialization boundary.

- [ ] **Step 4: Implement strict dataclass parsing and validation**

```python
# plugins/prefix_cache/ais_bench_prefix_cache/scenario.py
def load_scenario(path: Path) -> Scenario:
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require_exact_keys(raw, _TOP_LEVEL_KEYS, "")
    scenario = Scenario.from_dict(raw, source_path=path.resolve())
    scenario.validate()
    return scenario
```

Validation must reject unknown keys, ratios outside `[0, 1]`, nonpositive lengths/counts/block size/DP size, unsupported modes, a CSV mode without a CSV path, and any multi-instance key. Validation messages include the full dotted field path.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest plugins/prefix_cache/tests/test_scenario.py -v`

Expected: all scenario tests pass.

- [ ] **Step 6: Commit Task 1**

```text
git add plugins/prefix_cache/setup.py plugins/prefix_cache/README.md plugins/prefix_cache/ais_bench_prefix_cache plugins/prefix_cache/tests/conftest.py plugins/prefix_cache/tests/test_scenario.py
git commit -m "feat(prefix-cache): scaffold plugin and scenario model"
```

### Task 2: Deterministic lengths, GSM8K selection, grouping, and ordering

**Files:**
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/generation/__init__.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/generation/lengths.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/generation/corpus.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/generation/grouping.py`
- Create: `plugins/prefix_cache/tests/test_lengths_corpus_grouping.py`

**Interfaces:**
- Consumes: `Scenario.requests`, `Scenario.corpus`, `Scenario.prefix_cache.groups`, and top-level random seed.
- Produces: `build_input_lengths(config, count, seed) -> list[int]`.
- Produces: `build_output_lengths(config, count, seed) -> list[int]`.
- Produces: `load_gsm8k(path: Path, field: str = "question") -> list[GSMRecord]`.
- Produces: `select_gsm8k(records, selection, count, seed) -> list[GSMRecord]`.
- Produces: `assign_groups(count, config, seed) -> list[str]` and `order_indices(group_ids, strategy, seed) -> list[int]`.

- [ ] **Step 1: Write failing deterministic generation tests**

```python
def test_range_lengths_are_deterministic():
    cfg = {"mode": "range", "ranges": [{"min": 10, "max": 12, "count": 4}]}
    assert build_input_lengths(cfg, 4, 7) == build_input_lengths(cfg, 4, 7)
    assert all(10 <= value <= 12 for value in build_input_lengths(cfg, 4, 7))


def test_hash_selection_uses_normalized_question(tmp_path):
    path = tmp_path / "gsm.jsonl"
    path.write_text('{"question":"  two plus two?  "}\n', encoding="utf-8")
    records = load_gsm8k(path)
    chosen = select_gsm8k(records, {"mode": "question_sha256", "values": [records[0].question_sha256]}, 1, 9)
    assert chosen[0].question == "two plus two?"


def test_weighted_groups_use_largest_remainder():
    groups = assign_groups(10, {"count": 3, "assignment": {"mode": "weights", "weights": [0.5, 0.3, 0.2]}}, 42)
    assert [groups.count(f"group-{i}") for i in range(3)] == [5, 3, 2]


def test_interleave_round_robins_groups():
    group_ids = ["group-0", "group-0", "group-1", "group-1"]
    assert order_indices(group_ids, "interleave", 42) == [0, 2, 1, 3]


@pytest.mark.parametrize("strategy", ["sequential", "within_group_shuffle", "interleave", "global_shuffle"])
def test_all_order_strategy_names_are_supported(strategy):
    assert sorted(order_indices(["g0", "g0", "g1", "g1"], strategy, 42)) == [0, 1, 2, 3]
```

- [ ] **Step 2: Run tests and verify missing-function failures**

Run: `python -m pytest plugins/prefix_cache/tests/test_lengths_corpus_grouping.py -v`

Expected: tests fail because generation modules do not exist.

- [ ] **Step 3: Implement lengths and CSV aliases**

Implement fixed/range/CSV input modes and fixed/uniform/truncated_normal/CSV output modes. CSV input accepts `input_prompt_tokens`, `content_tokens`, or `input_tokens`; output accepts `output_tokens`. Use `random.Random(seed)` and rejection sampling for truncated normal with a bounded attempt count; a degenerate range returns the fixed endpoint.

- [ ] **Step 4: Implement GSM8K indexing and selection**

```python
@dataclass(frozen=True)
class GSMRecord:
    line_index: int
    question: str
    question_sha256: str


def normalize_question(value: str) -> str:
    return " ".join(value.strip().split())
```

Raise `ScenarioValidationError` with line numbers for malformed rows, and reject missing/ambiguous hashes rather than substituting random records.

- [ ] **Step 5: Implement group allocation and all four order strategies**

Use group IDs `group-0` through `group-{count-1}`. Uniform allocation differs by at most one request. Zipf and explicit weights convert fractional counts through largest remainder with stable group-ID tie-breaking.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest plugins/prefix_cache/tests/test_lengths_corpus_grouping.py -v`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```text
git add plugins/prefix_cache/ais_bench_prefix_cache/generation plugins/prefix_cache/tests/test_lengths_corpus_grouping.py
git commit -m "feat(prefix-cache): add deterministic data dimensions"
```

### Task 3: Canonical prefixes, natural suffixes, and globally unique seeds

**Files:**
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/generation/tokens.py`
- Create: `plugins/prefix_cache/tests/test_tokens.py`

**Interfaces:**
- Consumes: a Hugging Face-compatible tokenizer, selected `GSMRecord` values, group IDs, requested token lengths, block size, and random seed.
- Produces: `find_boundary_safe_token_ids(tokenizer, minimum: int) -> list[int]`.
- Produces: `build_canonical_prefixes(tokenizer, group_sources, max_lengths, block_size) -> dict[str, CanonicalPrefix]`.
- Produces: `build_unique_seed_tokens(safe_ids, request_ids, seed_length, random_seed) -> dict[str, tuple[int, ...]]`.
- Produces: `build_prompt(tokenizer, canonical, shared_prefix_tokens, seed, suffix_records, target_tokens) -> PromptBuildResult`.

- [ ] **Step 1: Write a tiny round-trip tokenizer fixture and failing tests**

```python
def test_seed_sequences_are_global_unique_and_deterministic(fake_tokenizer):
    safe = find_boundary_safe_token_ids(fake_tokenizer, minimum=4)
    first = build_unique_seed_tokens(safe, ["r0", "r1", "r2"], 2, 42)
    second = build_unique_seed_tokens(safe, ["r0", "r1", "r2"], 2, 42)
    assert first == second
    assert len(set(first.values())) == 3


def test_groups_have_different_first_block(fake_tokenizer):
    sources = {"group-0": [GSMRecord(0, "alpha question", "h0")], "group-1": [GSMRecord(1, "beta question", "h1")]}
    prefixes = build_canonical_prefixes(fake_tokenizer, sources, {"group-0": 8, "group-1": 8}, 4)
    assert prefixes["group-0"].token_ids[:4] != prefixes["group-1"].token_ids[:4]


def test_prompt_layout_round_trips(fake_tokenizer):
    canonical = CanonicalPrefix("group-0", "canonical text", tuple(range(20)), "canonical-hash", (0,), ("h0",))
    suffixes = [GSMRecord(0, "natural suffix question", "h0")]
    result = build_prompt(fake_tokenizer, canonical, 8, (91, 92, 93, 94), suffixes, 20)
    assert len(fake_tokenizer.encode(result.text, add_special_tokens=False)) == 20
    assert result.token_ids[:8] == canonical.token_ids[:8]
    assert result.token_ids[8:12] == (91, 92, 93, 94)
```

- [ ] **Step 2: Run tests and verify missing implementation**

Run: `python -m pytest plugins/prefix_cache/tests/test_tokens.py -v`

Expected: tests fail because `tokens.py` is absent.

- [ ] **Step 3: Implement safe-token and multi-token seed generation**

Derive each seed from `SHA-256(f"{random_seed}:{request_id}")`, encode the digest as a base-N sequence over boundary-safe token IDs, and extend the sequence deterministically when `seed_length` exceeds one digest chunk. Verify uniqueness after construction; duplicate token tuples raise `ArtifactValidationError`.

- [ ] **Step 4: Implement canonical-prefix isolation**

Build each canonical prefix from deterministic GSM8K question material plus a group-specific boundary-safe discriminator. Keep extending natural material until the group maximum is reached. Re-encode the final text and reject any group pair whose first full block matches.

- [ ] **Step 5: Implement exact prompt construction**

Prompt construction uses `[canonical slice][seed][natural suffix]`. It appends GSM8K question tokens until the target is reached, truncates by token ID, decodes, re-encodes, and performs bounded boundary-safe correction. Return actual tokens and GSM source references; never silently accept a changed prefix or seed boundary.

- [ ] **Step 6: Run token tests**

Run: `python -m pytest plugins/prefix_cache/tests/test_tokens.py -v`

Expected: all tests pass, including a vocabulary-too-small case that succeeds with multi-token seeds.

- [ ] **Step 7: Commit Task 3**

```text
git add plugins/prefix_cache/ais_bench_prefix_cache/generation/tokens.py plugins/prefix_cache/tests/test_tokens.py
git commit -m "feat(prefix-cache): build canonical prompts and unique seeds"
```

### Task 4: DP-aware theory simulation and target-driven solver

**Files:**
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/generation/theory.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/generation/solver.py`
- Create: `plugins/prefix_cache/tests/test_theory_solver.py`

**Interfaces:**
- Consumes: ordered request identities, input lengths, group IDs, cold routes, block size, seed lengths, cache mode, and target rate.
- Produces: `assign_cold_routes(group_ids, dp_size, explicit=None) -> list[int | None]`.
- Produces: `simulate_theory(plans, mode, warmup_watermarks=None) -> TheorySummary`.
- Produces: `solve_prefix_lengths(problem: SolveProblem) -> SolveResult`.
- `SolveResult` includes `shared_prefix_tokens`, `requested_hit_tokens`, `effective_hit_tokens`, `effective_hit_rate`, `min_reachable_rate`, `max_reachable_rate`, `adjusted`, and `reason`.

- [ ] **Step 1: Write failing watermark and reachability tests**

```python
def make_plans(prefixes, groups, ranks):
    return [
        RequestPlan.minimal_for_theory(
            request_id=f"r{i}", sequence_index=i, group_id=groups[i],
            dp_rank=ranks[i], shared_prefix_tokens=prefix,
            actual_input_tokens=prefix + 16,
        )
        for i, prefix in enumerate(prefixes)
    ]


def test_cold_dp_watermarks_are_per_group_and_rank():
    plans = make_plans(prefixes=[16, 16, 16, 16], groups=["g0"] * 4, ranks=[0, 1, 0, 1])
    result = simulate_theory(plans, mode="cold")
    assert [row.theoretical_hit_tokens for row in result.rows] == [0, 0, 16, 16]


def test_warmup_initializes_every_group_watermark():
    plans = make_plans(prefixes=[16, 32], groups=["g0", "g0"], ranks=[None, None])
    result = simulate_theory(plans, mode="warmup", warmup_watermarks={"g0": 32})
    assert [row.theoretical_hit_tokens for row in result.rows] == [16, 32]


def test_group_local_round_robin_route():
    assert assign_cold_routes(["g0", "g1", "g0", "g1"], 2) == [0, 0, 1, 1]


def test_solver_chooses_nearest_reachable_rate():
    problem = SolveProblem.from_dimensions(
        input_lengths=[40, 40], group_ids=["g0", "g0"], dp_ranks=[0, 0],
        block_size=16, seed_lengths=[16, 16], mode="cold", target_hit_rate=0.31,
    )
    result = solve_prefix_lengths(problem)
    candidates = exhaustive_reachable_rates(problem)
    assert abs(result.effective_hit_rate - 0.31) == min(abs(value - 0.31) for value in candidates)
```

Implement `RequestPlan.minimal_for_theory(...)` as a test-oriented classmethod with deterministic empty/default values for non-theory fields. Implement `SolveProblem.from_dimensions(...)` as the validated public constructor used by both the pipeline and tests. Define `exhaustive_reachable_rates(problem)` inside the test module with `itertools.product` and `simulate_theory`.

- [ ] **Step 2: Run tests and verify missing modules**

Run: `python -m pytest plugins/prefix_cache/tests/test_theory_solver.py -v`

Expected: import or missing-function failures.

- [ ] **Step 3: Implement route and theory primitives**

Use `(group_id, dp_rank)` as the cold watermark key and `group_id` as the warmup key. For each row compute `hit = min(shared_prefix, watermark_before)` and then `watermark_after = max(watermark_before, shared_prefix)`. Aggregate request, group, cold-DP, and global values by summing tokens before division.

- [ ] **Step 4: Implement deterministic nearest-target solving**

Create candidates `0, B, ..., floor((input_tokens - seed_tokens) / B) * B`. Start from zero, add one block at a time to the request whose simulated marginal hit gain most reduces absolute target-token error, recomputing affected lane watermarks after each change. Once no addition improves the primary objective, test one-block removals and swaps. Break ties by group-balance error, changed-request count, then `(sequence_index, group_id, dp_rank)`.

- [ ] **Step 5: Add exhaustive small-case oracle tests**

For problems with at most four requests and three candidates each, enumerate every prefix vector in the test and assert the solver's primary absolute error equals the exhaustive optimum. Test cold single-DP, cold multi-DP, and warmup.

- [ ] **Step 6: Run theory/solver tests**

Run: `python -m pytest plugins/prefix_cache/tests/test_theory_solver.py -v`

Expected: all tests pass and repeated runs return identical vectors.

- [ ] **Step 7: Commit Task 4**

```text
git add plugins/prefix_cache/ais_bench_prefix_cache/generation/theory.py plugins/prefix_cache/ais_bench_prefix_cache/generation/solver.py plugins/prefix_cache/tests/test_theory_solver.py
git commit -m "feat(prefix-cache): solve and simulate DP-aware hit rates"
```

### Task 5: Artifact writer, prepare pipeline, and prepare/validate/inspect CLI

**Files:**
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/generation/artifacts.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/generation/pipeline.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/cli.py`
- Create: `plugins/prefix_cache/tests/test_artifacts_pipeline.py`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: `prepare_scenario(path: Path, overwrite: bool = False) -> ArtifactPaths`.
- Produces: `write_artifacts(output_dir, run_id, rows, manifest_seed) -> ArtifactPaths`.
- Produces: `validate_artifacts(manifest_path: Path) -> ValidationReport`.
- CLI subcommands: `prepare`, `validate`, and `inspect`.

- [ ] **Step 1: Write failing four-artifact contract test**

```python
def test_prepare_writes_four_consistent_artifacts(tmp_path, fake_tokenizer_loader):
    paths = prepare_scenario(write_scenario_and_gsm(tmp_path))
    assert paths.full.name.endswith(".full.jsonl")
    assert paths.requests.name.endswith(".requests.jsonl")
    assert paths.manifest.name.endswith(".manifest.json")
    assert paths.analysis.name.endswith(".analysis.json")
    request_rows = read_jsonl(paths.requests)
    assert set(request_rows[0]) == {"question", "answer", "max_tokens"}
    report = validate_artifacts(paths.manifest)
    assert report.ok


def test_cold_first_row_can_have_prefix_layout_but_zero_hit(tmp_path, fake_tokenizer_loader):
    paths = prepare_scenario(write_cold_scenario(tmp_path))
    first = read_jsonl(paths.full)[0]
    assert first["shared_prefix_tokens"] >= 0
    assert first["theoretical_hit_tokens"] == 0
```

- [ ] **Step 2: Run tests and verify missing pipeline**

Run: `python -m pytest plugins/prefix_cache/tests/test_artifacts_pipeline.py -v`

Expected: tests fail because artifact/pipeline functions are missing.

- [ ] **Step 3: Implement atomic JSON/JSONL writes**

Write a sibling `.<name>.tmp-<pid>` file, flush and close it, validate row counts, and use `os.replace`. Reject overwrite unless the caller explicitly enabled it and all resolved targets remain beneath the configured output directory. Manifest records hashes for full/requests/analysis; analysis records the Manifest hash after Manifest completion to avoid recursive hashing.

- [ ] **Step 4: Implement prepare pipeline**

Execute stages in this order: load scenario → load/select GSM8K → build lengths/groups/order/routes → solve prefix lengths → build canonical prefixes/seeds/prompts → re-simulate theory from actual token lengths → write full and requests → write generation-only analysis → write Manifest and final analysis hash references → validate all artifacts.

- [ ] **Step 5: Implement CLI exit handling**

```python
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args)
    except PrefixCacheError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
```

The console entry point calls `raise SystemExit(main())`. `inspect` prints reachability and distribution summaries without sending HTTP requests.

- [ ] **Step 6: Run artifact tests**

Run: `python -m pytest plugins/prefix_cache/tests/test_artifacts_pipeline.py -v`

Expected: all tests pass, and a second prepare without `--overwrite` fails safely.

- [ ] **Step 7: Commit Task 5**

```text
git add plugins/prefix_cache/ais_bench_prefix_cache/generation/artifacts.py plugins/prefix_cache/ais_bench_prefix_cache/generation/pipeline.py plugins/prefix_cache/ais_bench_prefix_cache/cli.py plugins/prefix_cache/tests/test_artifacts_pipeline.py
git commit -m "feat(prefix-cache): prepare and validate auditable artifacts"
```

### Task 6: vLLM Prometheus metrics, capability probing, reset, and per-DP warmup

**Files:**
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/runtime/__init__.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/runtime/metrics.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/runtime/vllm.py`
- Create: `plugins/prefix_cache/tests/test_metrics_vllm.py`

**Interfaces:**
- Consumes: service config, group warmup plans, and requests/aiohttp-compatible transports.
- Produces: `parse_metrics(text, dp_size, engine_label_map=None) -> MetricSnapshot`.
- Produces: `diff_metrics(before, after) -> ActualMetrics`.
- Produces: `VLLMClient.probe()`, `.reset()`, `.snapshot()`, `.send_completion()`, and `.warm_every_group_rank()`.

- [ ] **Step 1: Write failing metric and HTTP behavior tests**

```python
V1_METRICS_TWO_ENGINES = """\
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries{engine="0",model_name="m"} 100
vllm:prefix_cache_queries{engine="1",model_name="m"} 100
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits{engine="0",model_name="m"} 50
vllm:prefix_cache_hits{engine="1",model_name="m"} 60
"""


def test_v1_metrics_are_parsed_per_engine():
    snap = parse_metrics(V1_METRICS_TWO_ENGINES, dp_size=2)
    assert snap.by_rank[0].queries == 100
    assert snap.by_rank[1].hits == 60


def test_global_actual_rate_sums_tokens_before_division():
    actual = diff_metrics(BASELINE, AFTER)
    assert actual.global_hits == 90
    assert actual.global_queries == 120
    assert actual.global_hit_rate == 0.75


def test_missing_rank_is_fatal():
    with pytest.raises(RuntimeCapabilityError, match="missing DP ranks: 1"):
        parse_metrics(V1_METRICS_ONE_ENGINE, dp_size=2)


def test_warmup_sends_every_group_to_every_rank(mock_transport):
    client = VLLMClient(SERVICE, transport=mock_transport)
    client.warm_every_group_rank({"g0": "p0", "g1": "p1"}, max_tokens=1)
    assert mock_transport.dp_headers == [0, 1, 0, 1]
```

The test module defines `BASELINE` and `AFTER` as concrete `MetricSnapshot` values, `V1_METRICS_ONE_ENGINE` as rank-0-only text, `SERVICE` as a `ServiceConfig`, and `MockTransport` with recorded `dp_headers` and deterministic JSON responses.

- [ ] **Step 2: Run tests and verify missing runtime modules**

Run: `python -m pytest plugins/prefix_cache/tests/test_metrics_vllm.py -v`

Expected: import failures.

- [ ] **Step 3: Implement V1 and legacy Prometheus parsing**

Use `prometheus_client.parser.text_string_to_metric_families`. Prefer `vllm:prefix_cache_queries`, `vllm:prefix_cache_hits`, and `vllm:kv_cache_usage_perc`; fall back only to an explicit alias map. Normalize `engine="0"`, `engine="engine_0"`, or configured labels to integer ranks. Reject missing/duplicate ranks, counter regressions, hits greater than queries, and mixed metric families.

- [ ] **Step 4: Implement local-header HTTP requests**

```python
def rank_headers(base: Mapping[str, str], dp_rank: int | None) -> dict[str, str]:
    headers = dict(base)
    if dp_rank is not None:
        headers["X-data-parallel-rank"] = str(dp_rank)
    return headers
```

Never mutate shared base headers. Probe every configured rank with a minimal completion, then verify the matching engine metric changed; reset occurs after probes. Reset failure is fatal unless `assume_empty_cache` is true, in which case return a structured warning.

- [ ] **Step 5: Implement all-group/all-rank warmup and baseline separation**

Warmup sends one plan per Cartesian pair in stable group/rank order, collects success and latency, fails if any pair fails, waits for metrics stability with a bounded timeout, and only then returns the formal baseline snapshot.

- [ ] **Step 6: Run runtime tests**

Run: `python -m pytest plugins/prefix_cache/tests/test_metrics_vllm.py -v`

Expected: all tests pass for V1, legacy aliases, reset fallback, header probing, and warmup coverage.

- [ ] **Step 7: Commit Task 6**

```text
git add plugins/prefix_cache/ais_bench_prefix_cache/runtime plugins/prefix_cache/tests/test_metrics_vllm.py plugins/prefix_cache/setup.py
git commit -m "feat(prefix-cache): collect vLLM multi-DP cache metrics"
```

### Task 7: AISBench Dataset, lane-preserving Inferencer, and DP-routed API Model

**Files:**
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/datasets/__init__.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/datasets/prefix_cache_dataset.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/openicl/__init__.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/openicl/icl_inferencer/__init__.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/openicl/icl_inferencer/prefix_cache_gen_inferencer.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/models/__init__.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/models/vllm_prefix_cache_api.py`
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/config.py`
- Create: `plugins/prefix_cache/tests/test_aisbench_integration.py`

**Interfaces:**
- Consumes: validated requests/full/Manifest artifacts.
- Produces registered `PrefixCacheDataset`, `PrefixCacheGenInferencer`, and `VLLMPrefixCacheAPI`.
- Produces `build_dataset_config(scenario_path: Path) -> dict` and `build_model_config(scenario_path: Path) -> dict`.
- Adds internal sample fields `dp_rank`, `group_id`, `lane_sequence`, and `cache_mode`; they do not appear in requests JSONL.

- [ ] **Step 1: Write failing plugin integration tests**

```python
def test_dataset_joins_route_metadata_by_sequence(tmp_path):
    dataset = PrefixCacheDataset.load(**write_three_consistent_artifacts(tmp_path))
    assert dataset[0]["max_out_len"] == 8
    assert dataset[0]["dp_rank"] == 0
    assert dataset[0]["lane_sequence"] == 0


@pytest.mark.asyncio
async def test_model_uses_request_local_dp_header(aiohttp_post_mock):
    model = make_model(aiohttp_post_mock)
    await model.generate("prompt", 1, RequestOutput(True), dp_rank=1)
    assert aiohttp_post_mock.call_args.kwargs["headers"]["X-data-parallel-rank"] == "1"
    assert "X-data-parallel-rank" not in model.headers


@pytest.mark.asyncio
async def test_lane_sequencer_blocks_second_request_until_first_finishes():
    sequencer = LaneSequencer()
    events = await exercise_out_of_order_completion(sequencer, lane=("g0", 0), sequences=[0, 1])
    assert events == ["start-0", "finish-0", "start-1", "finish-1"]
```

The test module defines `make_model(post_mock)` using `VLLMPrefixCacheAPI(model="m", url="http://server/")`. It defines `exercise_out_of_order_completion` with two asyncio tasks and events so the lane behavior is observable without real HTTP.

- [ ] **Step 2: Run tests and verify missing registered classes**

Run: `python -m pytest plugins/prefix_cache/tests/test_aisbench_integration.py -v`

Expected: import/registry failures.

- [ ] **Step 3: Implement artifact-joining Dataset**

Register with `@LOAD_DATASET.register_module()`. Load and validate all artifacts before returning `datasets.Dataset.from_list(rows)`. Convert `max_tokens` to `max_out_len`; reject sampling, repetition, row-count mismatch, changed order, or a Manifest hash mismatch.

- [ ] **Step 4: Implement metadata-enriching Inferencer and lane sequencer**

Subclass `GenInferencer`. `get_data_list` calls `super()`, reads the same-index raw test row, and copies the four internal fields. `do_request` bypasses sequencing for warmup mode; for cold mode it waits on `LaneSequencer.wait_turn((group_id, dp_rank), lane_sequence)`, calls `super().do_request`, and advances the lane in `finally`. Generated AISBench config must set one process worker so lane state is not split across processes; request concurrency remains available inside that process.

- [ ] **Step 5: Implement request-local Header Model**

Subclass `VLLMCustomAPI`. Add an internal body key in `get_request_body`, then override text and stream HTTP methods to remove that key from a copied payload and construct a local header dictionary. Preserve BaseAPIModel retry, Output timing, response parsing, and error semantics. A retry must retain the same DP rank.

- [ ] **Step 6: Implement config builders**

`build_dataset_config` uses `PrefixCacheDataset`, `ZeroRetriever`, and `PrefixCacheGenInferencer`. `build_model_config` uses `VLLMPrefixCacheAPI`. The generated config rejects `max_num_workers != 1` in cold mode and sets input columns to `question` and `max_out_len` without templating extra text around the prompt.

- [ ] **Step 7: Run AISBench integration tests**

Run: `python -m pytest plugins/prefix_cache/tests/test_aisbench_integration.py -v`

Expected: all tests pass and concurrent rank headers never leak between requests.

- [ ] **Step 8: Commit Task 7**

```text
git add plugins/prefix_cache/ais_bench_prefix_cache/datasets plugins/prefix_cache/ais_bench_prefix_cache/openicl plugins/prefix_cache/ais_bench_prefix_cache/models plugins/prefix_cache/ais_bench_prefix_cache/config.py plugins/prefix_cache/tests/test_aisbench_integration.py
git commit -m "feat(prefix-cache): integrate DP routing with AISBench"
```

### Task 8: End-to-end runtime orchestration, analysis, and warning-only thresholds

**Files:**
- Create: `plugins/prefix_cache/ais_bench_prefix_cache/runtime/orchestrator.py`
- Modify: `plugins/prefix_cache/ais_bench_prefix_cache/cli.py`
- Create: `plugins/prefix_cache/tests/test_orchestrator_cli.py`

**Interfaces:**
- Consumes: prepared artifacts, `VLLMClient`, AISBench executable/config, and subprocess runner.
- Produces: `run_scenario(scenario_path: Path, aisbench_config: Path) -> AnalysisResult`.
- Produces: `analyze_run(manifest_path, baseline, after, phase_results) -> AnalysisResult`.
- Adds CLI subcommands `run` and `analyze`.

- [ ] **Step 1: Write failing phase-order and exit-code tests**

```python
def test_warmup_run_orders_phases_and_excludes_warmup(mock_runtime):
    result = run_scenario(mock_runtime.scenario, mock_runtime.config)
    assert mock_runtime.events == ["probe", "reset", "warm-all", "baseline", "aisbench", "after"]
    assert result.actual.global_queries == mock_runtime.formal_query_delta


def test_cold_run_has_no_warmup_and_uses_baseline_after_reset(mock_runtime):
    run_scenario(mock_runtime.cold_scenario, mock_runtime.config)
    assert mock_runtime.events == ["probe", "reset", "baseline", "aisbench", "after"]


def test_large_deviation_is_warning_with_zero_exit(mock_runtime, capsys):
    mock_runtime.actual_rate = 0.10
    mock_runtime.theory_rate = 0.90
    assert cli_main(["run", "--scenario", str(mock_runtime.scenario), "--config", str(mock_runtime.config)]) == 0
    assert "WARNING" in capsys.readouterr().err
```

`test_orchestrator_cli.py` defines a concrete `FakeRuntime` that records each method call, returns fixed `MetricSnapshot` values, and replaces subprocess execution with a successful `CompletedProcess`. The `mock_runtime` fixture writes both cold and warmup scenarios and exposes their paths.

- [ ] **Step 2: Run tests and verify missing orchestration**

Run: `python -m pytest plugins/prefix_cache/tests/test_orchestrator_cli.py -v`

Expected: missing-function failures.

- [ ] **Step 3: Implement explicit phase state machine**

Use phases `PRECHECK`, `RESET`, `WARMUP`, `BASELINE`, `FORMAL`, `AFTER`, `ANALYZE`, and `COMPLETE`. Warmup skips only in cold mode. Invoke AISBench with `subprocess.run([sys.executable, "-m", "ais_bench.benchmark.cli.main", config, "--mode", "perf"], check=False, env=...)`; do not use `shell=True`. Nonzero AISBench status is fatal.

- [ ] **Step 4: Implement analysis output and warning policy**

Store raw baseline/after snapshots, per-DP deltas, global sums, group theory, cold-DP theory, phase results, warmup matrix, AISBench output path, and structured warnings. Threshold warnings never alter status or exit code. Data/schema/runtime/metrics/AISBench failures remain nonzero.

- [ ] **Step 5: Make run resumability explicit**

`run` always validates existing artifacts. It may reuse prepared artifacts only when hashes match. It never reuses a prior runtime baseline or after snapshot. Analysis writes atomically after every completed runtime phase so a crash leaves an auditable incomplete state.

- [ ] **Step 6: Run orchestration tests**

Run: `python -m pytest plugins/prefix_cache/tests/test_orchestrator_cli.py -v`

Expected: all tests pass for cold/warmup phase order, fatal failures, and warning-only deviations.

- [ ] **Step 7: Commit Task 8**

```text
git add plugins/prefix_cache/ais_bench_prefix_cache/runtime/orchestrator.py plugins/prefix_cache/ais_bench_prefix_cache/cli.py plugins/prefix_cache/tests/test_orchestrator_cli.py
git commit -m "feat(prefix-cache): orchestrate cold and warmup benchmark runs"
```

### Task 9: Example configuration, operator documentation, and full verification

**Files:**
- Create: `plugins/prefix_cache/config_examples/scenario.example.json`
- Create: `plugins/prefix_cache/config_examples/prefix_cache_perf.py`
- Modify: `plugins/prefix_cache/README.md`
- Test: all files under `plugins/prefix_cache/tests/`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: copyable prepare/run commands, artifact descriptions, vLLM prerequisites, DP limitations, and optional live-E2E instructions.

- [ ] **Step 1: Write executable example-config tests**

```python
def test_example_scenario_passes_schema_validation(repo_root):
    scenario = load_scenario(repo_root / "plugins/prefix_cache/config_examples/scenario.example.json")
    assert scenario.prefix_cache.mode in {"cold", "warmup"}


def test_example_python_config_imports_with_plugin_installed(repo_root, monkeypatch):
    monkeypatch.setenv("AISBENCH_PREFIX_CACHE_SCENARIO", str(repo_root / "plugins/prefix_cache/config_examples/scenario.example.json"))
    module = runpy.run_path(str(repo_root / "plugins/prefix_cache/config_examples/prefix_cache_perf.py"))
    assert module["datasets"]
    assert module["models"]
```

- [ ] **Step 2: Add the scenario and native AISBench Python config**

The Python config reads `AISBENCH_PREFIX_CACHE_SCENARIO`, calls the two config builders, selects `PrefixCacheGenInferencer`, and sets a single process worker. It contains no machine-specific absolute path or credentials.

- [ ] **Step 3: Complete operator README**

Document installation, `prepare`, `inspect`, `validate`, `run`, and `analyze`; vLLM `--enable-prefix-caching`, metrics, reset dev-mode requirement, `X-data-parallel-rank`, cold lane semantics, every-group/every-rank warmup, `assume_empty_cache`, four artifacts, warning-only deviations, and the explicit lack of multi-instance support.

- [ ] **Step 4: Install plugin editable and run unit/component suite**

Run: `python -m pip install -e plugins/prefix_cache`

Run: `python -m pytest plugins/prefix_cache/tests -v`

Expected: all plugin tests pass.

- [ ] **Step 5: Run regression checks against touched integration surfaces**

Run: `python -m pytest tests/UT/test_registry.py tests/UT/datasets/test_custom.py tests/UT/models/api_models/test_base_api.py -v`

Expected: all selected AISBench regression tests pass.

- [ ] **Step 6: Run static artifact and source checks**

Run: `python -m compileall -q plugins/prefix_cache/ais_bench_prefix_cache`

Run: `rg -n "TO[D]O|T[B]D|FIX[M]E|shell=True|instances\s*=|instance_urls" plugins/prefix_cache/ais_bench_prefix_cache plugins/prefix_cache/tests`

Expected: compile succeeds; the scan has no implementation placeholders, unsafe subprocess shell use, or multi-instance configuration.

- [ ] **Step 7: Record optional live vLLM E2E commands without making CI depend on them**

Document four opt-in cases: DP1 cold, DP1 warmup, DP2 cold with routed headers, and DP2 warmup with every group/rank. Each command requires explicit service URLs and writes to a new output directory.

- [ ] **Step 8: Commit Task 9**

```text
git add plugins/prefix_cache
git commit -m "docs(prefix-cache): add examples and verification guide"
```

## Final Verification Gate

- [ ] Run `python -m pytest plugins/prefix_cache/tests -v` and confirm zero failures.
- [ ] Run selected AISBench registry/custom-dataset/base-API regressions and confirm zero failures.
- [ ] Run `python -m compileall -q plugins/prefix_cache/ais_bench_prefix_cache`.
- [ ] Generate a small deterministic cold DP2 dataset twice into separate directories and confirm full/requests content SHA-256 values match.
- [ ] Generate a small warmup DP2 dataset and inspect Manifest warmup matrix contains every group/rank pair.
- [ ] Verify requests JSONL rows contain exactly `question`, `answer`, and `max_tokens`.
- [ ] Verify analysis threshold deviations are warnings and a mocked otherwise-successful run returns zero.
- [ ] Verify `git diff --name-only` contains only `plugins/prefix_cache`, this plan, and any explicitly approved documentation; it must not contain files under `ais_bench/benchmark`.
- [ ] Invoke `superpowers:requesting-code-review` before claiming implementation complete.
- [ ] Invoke `superpowers:verification-before-completion` before final delivery.

## Spec Coverage Self-Review

| Approved specification area | Implemented by |
|---|---|
| Scope, strict scenario, single endpoint/no multi-instance | Tasks 1 and 9 |
| GSM8K random/index/hash selection and natural suffixes | Tasks 2 and 3 |
| Input/output length modes | Task 2 |
| Uniform/Zipf/explicit-weight Prefix Groups | Task 2 |
| Canonical prefixes and globally unique seed | Task 3 |
| Target-driven nearest-reachable solver | Task 4 |
| cold/warmup theory and four order strategies | Tasks 2 and 4 |
| Four atomic artifacts and reproducibility | Task 5 |
| vLLM V1/legacy metrics, reset, DP probes, every-group/every-rank warmup | Task 6 |
| AISBench plugin registration, minimal requests, cold lane order, local DP Header | Task 7 |
| Runtime phase ordering, actual/theory analysis, warning-only thresholds | Task 8 |
| Tests, examples, operator docs, optional live E2E | Task 9 and Final Verification Gate |
