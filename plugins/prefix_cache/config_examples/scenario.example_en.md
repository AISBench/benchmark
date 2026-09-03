# Scenario Configuration Reference

[中文](scenario.example.md) | English

This document explains every field in [scenario.example.json](scenario.example.json), including optional modes not expanded in the example.

## 1. General rules

- The file is strict JSON: no comments, trailing commas, or unknown fields.
- Relative paths are resolved from the Scenario file's directory.
- Rates use `0.0–1.0`; for example, `0.6` means 60%.
- Token lengths are measured with the tokenizer loaded from `tokenizer.path`.
- `{}` and `[]` in examples are JSON objects and arrays, not parameters.
- Every field may be omitted; source defaults match the current `scenario.example.json`.

## 2. Top-level fields

| Field | Required | Purpose |
|---|---:|---|
| `schema_version` | No | Configuration contract version; currently `"1.0"`. |
| `run` | No | Run ID, random seed, and artifact directory. |
| `tokenizer` | No | Token counting and Block alignment. |
| `corpus` | No | GSM8K source and sample selection. |
| `requests` | No | Formal request count and input/output lengths. |
| `output` | No | Optional third field in the minimal requests file. |
| `prefix_cache` | No | Cache mode, target rate, Groups, and order. |
| `service` | No | vLLM endpoints, metrics, reset, DP routing, and timeouts. |
| `validation` | No | Warning thresholds for rate differences. |
| `aisbench` | No | AISBench config, work directory, and extra CLI arguments used by `run`. |

Nested objects also use strict allowlists. In particular, `corpus.selection` accepts `mode`, `values`, `indices`, and `question_sha256`; input/output length objects accept the modes and fields described below; `aisbench.dataset` accepts `abbr`, `input_columns`, `output_column`, `prompt_template`, and `pred_role`; and `aisbench.model` accepts `abbr`, `attr`, `stream`, `max_out_len`, `retry`, `batch_size`, and `generation_kwargs`.

## 3. `schema_version`

```json
"schema_version": "1.0"
```

Prevents the plugin from interpreting a file with an incompatible structure. Other versions fail immediately.

## 4. `run`

```json
"run": {
  "run_id": "gsm8k-prefix-cache-60",
  "random_seed": 42,
  "output_dir": "./outputs/gsm8k-prefix-cache-60"
}
```

| Field | Default | Purpose |
|---|---|---|
| `run_id` | `gsm8k-prefix-cache-60` | Base run name. Execution appends `_YYYYMMDD_HHMMSS` and uses it as the artifact prefix. |
| `random_seed` | `42` | Controls corpus sampling, lengths, Group assignment, ordering, and unique seeds. |
| `output_dir` | `./outputs/gsm8k-prefix-cache-60` | Base artifact directory. The same timestamp is appended to its final component. |
| `overwrite` | `false` | Compatibility field. Use `prepare --overwrite` to rebuild formal artifacts. |

An execution may produce `outputs/gsm8k-prefix-cache-60_20260825_123456/log/...` and a sibling `result/` directory containing `full.jsonl`, `requests.jsonl`, `manifest.json`, and `analysis.json`. `inspect` creates a lightweight Manifest only; it no longer creates a standalone `.inspect.json`. Matching Manifests are reused by `prepare`/`run`; a Scenario hash change creates a new timestamp.

## 5. `tokenizer`

```json
"tokenizer": {
  "path": "/home/weights/Qwen3.6-27B",
  "block_size": 16,
  "trust_remote_code": false
}
```

| Field | Default | Purpose |
|---|---|---|
| `path` | `/home/weights/Qwen3.6-27B` | Local tokenizer directory or Hugging Face ID passed to `AutoTokenizer.from_pretrained`; must match vLLM. |
| `block_size` | `16` | Tokens per Prefix Cache Block; prefixes and seeds are aligned to it and it must match the server. |
| `revision` | `null` | Tokenizer branch, tag, or commit. |
| `trust_remote_code` | `false` | Whether to execute custom tokenizer code; enable only for trusted repositories. |

With `block_size=16` and `seed_blocks=1`, each request contains a 16-token unique seed between its public prefix and natural suffix.

## 6. `corpus`

```json
"corpus": {
  "path": "./GSM8K.jsonl",
  "field": "question",
  "selection": {"mode": "random"}
}
```

| Field | Default | Purpose |
|---|---|---|
| `path` | `./GSM8K.jsonl` | GSM8K JSONL path; every non-empty line must be an object. |
| `field` | `question` | Field containing natural-language text; standard answers are not concatenated. |
| `selection` | `{"mode":"random"}` | Selects canonical-prefix and natural-suffix samples. |

Questions are trimmed and consecutive whitespace is collapsed. `question_sha256` is computed from the normalized UTF-8 text.

### 6.1 `selection.mode=random`

```json
"selection": {"mode": "random"}
```

Deterministically shuffles rows with `random_seed`. If more rows are needed than the corpus contains, a new shuffle cycle starts.

### 6.2 `selection.mode=indices`

```json
"selection": {"mode": "indices", "values": [0, 15, 72]}
```

Uses zero-based row numbers (`0` is the first row). `values` may also be written as `indices`; a short list is cyclically reused, while a missing row fails.

### 6.3 `selection.mode=question_sha256`

```json
"selection": {
  "mode": "question_sha256",
  "values": ["64-character SHA-256 of a normalized question"]
}
```

`values` may also be written as `question_sha256`. Every hash must match exactly one corpus row; zero or multiple matches fail.

### 6.4 `selection.mode=mixed`

```json
"selection": {
  "mode": "mixed",
  "indices": [0, 15],
  "question_sha256": ["question SHA-256"]
}
```

Index-selected rows are added first, followed by hash-selected rows. If their combined count is smaller than required, the merged order is cyclically reused. At least one list must be non-empty; otherwise the plugin reports `specified GSM8K selection is empty`.

## 7. `requests`

```json
"requests": {
  "count": 100,
  "input_length": {"mode": "fixed", "value": 1024},
  "output_length": {"mode": "fixed", "value": 32}
}
```

### 7.1 `count`

Number of formal requests (default `100`), a positive integer. Warmup requests do not count and are not written to `requests.jsonl`.

### 7.2 `input_length`

Target total input tokens per formal request:

```text
shared prefix + globally unique seed + GSM8K natural suffix
```

Omitting the object defaults to `{"mode":"fixed","value":1024}`.

#### Fixed

```json
"input_length": {"mode": "fixed", "value": 1024}
```

`value` is a positive integer and applies to every request.

#### Closed-interval sampling

```json
"input_length": {
  "mode": "range",
  "ranges": [
    {"min": 512, "max": 1024, "count": 80},
    {"min": 2048, "max": 4096, "count": 20}
  ]
}
```

Bounds are inclusive. Each `count` is the number sampled from that interval and all counts must sum to the applicable request count. Sampling uses `random_seed`.

#### Explicit list

```json
"input_length": {"mode": "explicit", "values": [512, 768, 1024, 2048]}
```

All values must be positive integers. The list length must equal the global request count or the actual count of a Group override.

#### Truncated normal

```json
"input_length": {
  "mode": "truncated_normal",
  "min": 512,
  "max": 2048,
  "mean": 1024,
  "std": 256
}
```

Only integer samples in `[min,max]` are retained. `mean` defaults to the midpoint; `std` is derived from the width unless explicitly set, in which case it must be positive. `min=max` is fixed length.

#### CSV

```json
"input_length": {"mode": "csv", "path": "./input_lengths.csv"}
```

The row count must equal the applicable request count. The CSV must contain a positive-integer column named `input_prompt_tokens`, `content_tokens`, or `input_tokens`.

### 7.3 `output_length`

The sampled value becomes `max_tokens` in the full audit row. Omitting the object or fixed `value` defaults to 32.

Fixed:

```json
"output_length": {"mode": "fixed", "value": 32}
```

Uniform:

```json
"output_length": {"mode": "uniform", "min": 16, "max": 64}
```

`min` and `max` are positive, inclusive, and satisfy `max >= min`.

Truncated normal:

```json
"output_length": {
  "mode": "truncated_normal",
  "min": 16,
  "max": 128,
  "mean": 64,
  "std": 16
}
```

The same truncation/default rules as input length apply. A CSV mode requires an `output_tokens` positive-integer column and exactly `requests.count` rows.

### 7.4 Top-level `output`

```json
"output": {"output_key": null}
```

| Value | Effect |
|---|---|
| `null` | `requests.jsonl` contains only `question` and `answer`. |
| `"max_tokens"` | Append a `max_tokens` field. |
| `"output_tokens"` | Append an `output_tokens` field with the same value. |

This matches `extract_qa.py --output-key`. `full.jsonl.max_tokens` is always retained, so omitting the third public field does not change the load test.

## 8. `prefix_cache`

```json
"prefix_cache": {
  "mode": "warmup",
  "target_hit_rate": 0.6,
  "seed_blocks": 1,
  "minimum_non_shared_length": 16,
  "groups": {"count": 1, "assignment": {"mode": "uniform"}},
  "order": {"strategy": "interleave"}
}
```

### 8.1 `mode`

- `cold`: route formal requests by `(Prefix Group, DP rank)` lane and simulate each watermark from zero;
- `warmup`: create a warmup plan for every Group × DP rank in `warmup.plan`; formal requests are not pinned to a DP.

`prepare` only creates data and the plan. `run` sends each warmup before the formal baseline; warmup is excluded from formal request count, performance metrics, theoretical denominators, and actual counter deltas. Default: `warmup`.

### 8.2 `target_hit_rate`

Global token-weighted target in `[0,1]` (default `0.6`). It is the solver target, not a fixed per-request prefix percentage. Block alignment, order, Groups, watermarks, and cold DP routing are considered. The nearest reachable total is selected; among equal totals, the solver minimizes cumulative overshoot and fallback before approaching the target. Requested/effective/theoretical values and reasons are recorded.

Strictly monotonic cumulative rates are not guaranteed: a later lane's first cold request must miss, a request may lack capacity, and Block granularity is discrete. The solver minimizes unavoidable oscillation while preserving target-driven total accuracy.

### 8.3 `seed_blocks`

Positive integer number of Blocks in the unique seed (default `1`):

```text
seed tokens = seed_blocks × tokenizer.block_size
```

The seed lies between public prefix and natural suffix and is globally unique. Scenario loading verifies every input-length mode can contain the non-shared region.

### 8.4 `minimum_non_shared_length`

Minimum non-shared tokens per formal request. Defaults to `seed_blocks × block_size` and cannot be smaller. The maximum public prefix is `floor((input_length - minimum_non_shared_length) / block_size) × block_size`. Extra space beyond the seed is filled with the natural GSM8K suffix.

### 8.5 `groups.count`

Number of Prefix Groups (default `1`), producing `group-0`, `group-1`, etc. Each Group independently builds its canonical prefix, maintains its watermark, reports its theoretical rate, and receives per-DP warmup.

### 8.6 `groups.assignment`

Defaults to `{"mode":"uniform"}`.

```json
{"mode": "uniform"}
```

Distributes requests as evenly as possible; remainders follow stable Group order.

```json
{"mode": "zipf", "exponent": 1.0}
```

Group popularity is proportional to `1/rank^exponent`; exponent must be positive and larger values concentrate traffic in hot Groups.

```json
{"mode": "weights", "weights": [0.5, 0.3, 0.15, 0.05]}
```

Weight count must equal `groups.count`, weights cannot be negative, and their sum must be positive. Pre-normalization is not required.

### 8.7 `groups.overrides`

Per-Group overrides (default `{}`):

```json
"overrides": {
  "group-0": {
    "input_length": {"mode": "fixed", "value": 2048},
    "output_length": {"mode": "fixed", "value": 64},
    "corpus_selection": {"mode": "indices", "values": [0, 1, 2]}
  }
}
```

IDs must be `group-0` through `group-(count-1)`. Input/output support all global modes; corpus selection supports random, indices, question SHA-256, and mixed. Group range/CSV counts must equal the actual number assigned to that Group.

### 8.8 `order.strategy`

- `sequential`: retain assignment-stage order;
- `within_group_shuffle`: shuffle within each Group, then output by Group;
- `interleave`: rotate across Groups (default, useful for multi-tenant traffic);
- `global_shuffle`: deterministic global shuffle;
- `input_len_asc`: sort each Group short-to-long, then rotate across Groups; equal lengths retain stable order.

Theoretical watermarks are always re-simulated in final send order. To model a cold cache growing from short to long requests, combine `mode="cold"` and `strategy="input_len_asc"`. `LaneSequencer` serializes each `(group_id, dp_rank)` lane while independent Groups/DPs remain concurrent.

## 9. `service`

| Field | Default | Purpose |
|---|---|---|
| `inference_url` | `http://127.0.0.1:8000/v1/completions` | vLLM Completions endpoint for probes, warmup, and formal requests. |
| `metrics_url` | `http://127.0.0.1:8000/metrics` | Prometheus baseline/after endpoint. |
| `reset_url` | `http://127.0.0.1:8000/reset_prefix_cache` | Clears Prefix Cache before formal statistics. Empty/failed reset requires `assume_empty_cache`. |
| `model` | `model-name` | Model name in completion requests. |
| `dp_size` | `2` | DP ranks behind the one HTTP endpoint; used offline and for online probes/warmup/metric validation. |
| `assume_empty_cache` | `false` | Continue when reset is unavailable, recording `ASSUME_EMPTY_CACHE`. |
| `engine_label_map` | `{}` | Explicit Prometheus `engine` to DP-rank mapping; otherwise the trailing number is parsed. |
| `timeout_seconds` | `30` | HTTP timeout for probes, reset, warmup, and metrics. |
| `api_key` | `""` | Optional Bearer token. Manifest stores only whether it was configured. |
| `poll_interval_seconds` | `5.0` | KV gauge polling interval during formal scoring; `0` disables polling while baseline/after remain enabled. |

Offline commands do not contact the service. The plugin supports one endpoint with internal multiple DP ranks, not multiple independent vLLM instances.

## 10. `validation`

| Field | Default | Purpose |
|---|---:|---|
| `target_warning_pp` | `1.0` | Emit `TARGET_DEVIATION` when theory/target differs by more than this many percentage points. |
| `actual_warning_pp` | `5.0` | Emit `ACTUAL_DEVIATION` when actual/theory differs by more than this many percentage points. |

Units are percentage points, not relative percent. Both are warning-only and do not change a successful exit code. Analysis also records signed/absolute differences, reachability, and `PASS`/`PASS_WITH_WARNING`.

## 11. `aisbench`

| Field | Default | Purpose |
|---|---|---|
| `config` | `./plugins/prefix_cache/config_examples/prefix_cache_perf.py` | AISBench Python template rendered by `run`; `--config` can override it for one invocation. |
| `work_dir` | `./outputs/default` | AISBench base directory; child logs are written below its timestamped `logs/infer/`. |
| `extra_args` | `[]` | String arguments appended to the AISBench perf command. Example `['--num-warmups','0']` disables AISBench's own warmup, not plugin warmup. |
| `dataset` | See below | Dataset reader, prompt, and evaluation role. |
| `model` | See below | API streaming, retry, concurrency, and generation settings. |

`dataset` defaults and constraints:

| Field | Default | Meaning |
|---|---|---|
| `abbr` | `null` | Display name; null uses the timestamped run ID. |
| `input_columns` | `['question','max_out_len']` | Reader input columns; retain this contract. |
| `output_column` | `answer` | Reference-answer column; retain `answer`. |
| `prompt_template` | `{question}` | Prompt template; changing it breaks theory/actual token equivalence. |
| `pred_role` | `BOT` | AISBench prediction role; any non-empty string is allowed. |

`model` defaults and constraints:

| Field | Default | Meaning |
|---|---|---|
| `abbr` | `null` | Display name; null uses `<run_id>-vllm`. |
| `attr` | `service` | Must be `service` to enable service metrics such as TTFT. |
| `stream` | `true` | SSE streaming. False still measures Prefix Cache but cannot produce complete TTFT/TPOT/ITL. |
| `max_out_len` | `1` | Fallback maximum output; full-row `max_tokens` takes precedence. |
| `retry` | `2` | Non-negative API retry count. |
| `batch_size` | `1` | Positive concurrency base; a cold lane remains serialized by the plugin. |
| `generation_kwargs` | `{'temperature':0,'ignore_eos':true}` | JSON generation parameters merged into vLLM requests. |

The complete `aisbench` section may be omitted and old Scenarios receive current defaults. `config`/`work_dir` must be non-empty strings and `extra_args` a string list. Plugin types, artifact paths, routing, and inferencer contracts are not replaceable through Scenario.

## 12. Meaning of the example

The example generates 100 fixed-1024-token formal requests, one uniform Group, a 60% target, one unique 16-token seed, and a 16-token non-shared reserve. Requests are interleaved, warmup mode is enabled, and two DP ranks share one vLLM endpoint. Two per-DP warmup requests are planned and excluded from formal statistics. `--num-warmups 0` disables AISBench's own warmup. Differences above 1 pp (target) or 5 pp (actual) are warnings only.

## 13. Recommended checking order

```bash
ais-bench-prefix-cache inspect --scenario ./scenario.json
ais-bench-prefix-cache prepare --scenario ./scenario.json
ais-bench-prefix-cache validate --manifest <manifest-path>
ais-bench-prefix-cache run --scenario ./scenario.json
```

Review requested/effective/theoretical rates, reachability, Group distribution, `warmup.plan`, warnings, length summaries, per-request `request_random_seed`, and divergence collision status.

## 14. CLI behavior and return fields

### 14.1 `inspect`

Loads tokenizer/GSM8K and computes reachability without sending requests. It creates a fresh timestamp, writes `log/<run_id_timestamp>.inspect.log`, writes a lightweight `status="inspected"` Manifest under `result/`, and returns its `log` and `manifest` paths.

### 14.2 `prepare`

Reuses the latest matching inspect timestamp or creates a new one. It upgrades the lightweight Manifest to `prepared`, shows prompt progress on stderr, and returns:

| Field | Meaning |
|---|---|
| `full` | Full audit JSONL path. |
| `requests` | Minimal requests JSONL path. |
| `manifest` | Manifest path. |
| `analysis` | Theoretical analysis path. |
| `log` | Prepare log path. |

`--overwrite` affects only the four formal artifacts in the selected timestamp directory.

### 14.3 `validate`

Checks rows, fields, order, correspondence, and SHA-256 without generating or contacting a service. It returns `ok`, `rows`, and `run_id`; details go to the Manifest timestamp's validate log.

### 14.4 `run`

Reuses or auto-prepares artifacts, then performs per-DP probes, reset, optional Group × DP warmup, baseline, AISBench perf, after capture, and counter deltas. Plugin logs go to `.run.log`; AISBench child stdout/stderr remains visible in the terminal. Stdout returns complete analysis.

### 14.5 `analyze`

`analyze --manifest <path> --baseline <before.prom> --after <after.prom>` performs offline counter-delta analysis and updates analysis without contacting the service. It shares the validate log filename.

Successful execution returns 0. Scenario, generation, or artifact errors return 2. Reachability and rate differences are warnings only.

## 15. Request artifact fields

### 15.1 `<run_id>.requests.jsonl`

Each row writes `question`, then `answer`, then the optional `max_tokens`/`output_tokens` selected by `output.output_key`. `full.jsonl` always contains `max_tokens` and all routing/audit fields.

### 15.2 `<run_id>.full.jsonl`

| Field | Meaning |
|---|---|
| `request_id`, `sequence_index` | Stable ID and final zero-based send order. |
| `group_id`, `occurrence_index_within_group` | Prefix Group and occurrence number. |
| `dp_rank`, `lane_sequence` | Cold target DP and lane sequence; null when not applicable. |
| `target_input_tokens`, `actual_input_tokens`, `max_tokens` | Configured/retokenized input and output lengths. |
| `shared_prefix_tokens`, `seed_tokens`, `natural_suffix_tokens` | Three prompt components. |
| `question`, `answer` | Final prompt and AISBench placeholder (`none`). |
| `gsm_indices`, `gsm_hashes` | Natural-suffix corpus rows and normalized-question hashes. |
| `canonical_prefix_sha256`, `seed_sha256` | Group prefix and unique-seed fingerprints. |
| `request_random_seed` | Seed used for this request's difference block. |
| `watermark_before`, `theoretical_hit_tokens`, `watermark_after` | Lane cache state and theoretical hit. |
| `theoretical_hit_rate` | Hit tokens divided by actual input tokens. |
| `divergence_block_sha256`, `divergence_unique`, `collision_status` | Difference fingerprint, uniqueness check, and collision result. |

## 16. Complete Manifest fields

| Field | Meaning |
|---|---|
| `schema_version`, `plugin_version` | Contract and plugin versions. |
| `run_id`, `scenario_path`, `scenario_sha256` | Timestamped ID, original Scenario path, and hash. |
| `effective_config`, `effective_config_sha256` | Defaults-filled, resolved configuration and hash. |
| `corpus_sha256` | GSM8K file hash. |
| `tokenizer` | Tokenizer source, class, vocabulary, special IDs, Block size, and fingerprint. |
| `requests` | Count, total tokens, and length summaries. |
| `prefix_cache` | Mode, targets, theoretical values, reachability, adjustments, and validation. |
| `groups` | Per-Group canonical source, limits, reachability, and rates. |
| `dp` | DP size and cold routing. |
| `warmup` | Enabled flag and per-Group/per-DP plan. |
| `divergence` | Unique seed policy and collision status. |
| `artifacts` | File names, paths, rows, sizes, and hashes. |

`requests.input_length_summary` and `output_length_summary` contain `min`, `max`, `mean`, `p50`, `p90`, `p95`, `p99`, and up to ten non-empty bins. Each bin has `min`, `max`, and `count`; the fixed width is `max(1, ceil((global_max-global_min+1)/10))`.

`effective_config` records timestamped run/output values and resolved paths. `service.api_key` is replaced by the boolean `api_key_configured`; plaintext secrets are never stored. Group entries include `canonical_prefix_sha256`, `canonical_prefix_tokens`, `max_shared_prefix_tokens`, GSM8K indices/hashes, reachability, and theoretical rate. `warmup.plan` entries contain `request_id`, `group_id`, `dp_rank`, `prompt`, `input_tokens`, `shared_prefix_tokens`, `max_tokens`, and `included_in_formal_statistics=false`.

Artifact metadata uses `name`, `path`, `rows` (JSONL only), `bytes`, and `sha256`; analysis uses `sha256_at_prepare` because run/analyze legitimately rewrites it.

## 17. Complete `analysis.json` fields

| Field | Meaning |
|---|---|
| `schema_version`, `run_id`, `status` | Analysis contract, timestamped ID, and `prepared`/`complete`/`analyzed` state. |
| `requested_target_hit_rate`, `effective_target_hit_rate`, `theoretical_hit_rate` | Requested, nearest reachable, and final theoretical rates. |
| `target_difference_pp`, `target_signed_difference_pp`, `target_absolute_difference_pp` | Target deviations; the first currently equals absolute deviation. |
| `validation` | Status, reachability, warning-only flag, and exit-code effect. |
| `theory` | Global, Group, and cold-DP input/hit token totals and rates. |
| `runtime` | Online phases, probes, reset/warmup, AISBench exit code, and metric snapshots. |
| `actual` | Formal after-baseline query/hit deltas and KV usage. |
| `theory_actual_difference_pp` and signed/absolute variants | Actual-versus-theory differences. |
| `warnings` | `TARGET_UNREACHABLE`, `TARGET_DEVIATION`, `ACTUAL_DEVIATION`, or `ASSUME_EMPTY_CACHE`. |

### 17.1 Validation and warnings

`validation.status` is `PASS` with no warnings and `PASS_WITH_WARNING` otherwise. `warning_only=true` and `affects_exit_code=false` are fixed. `validation.actual_status` is added after run/analyze. Warning objects include code and the relevant bounds/difference.

### 17.2 Theory

`theory.input_tokens`/`hit_tokens` are global totals. `theory.groups.<id>` and `theory.dp.<rank>` contain the same fields and token-weighted `hit_rate`; warmup formal requests do not have fixed DP ranks, so `theory.dp` may be empty.

### 17.3 Runtime and metric snapshots

`runtime.phases` normally lists `precheck`, `reset`, optional `warmup`, `baseline`, `formal`, and `after`. `runtime.precheck` records success, recognized DP ranks, and matched metric names. `runtime.warmup[]` records Group, DP, success, and elapsed seconds. `runtime.aisbench_exit_code` is the child process code.

`metrics_baseline` is captured after reset/warmup and before formal perf; `metrics_after` is captured afterward. Both include metric names, per-DP cumulative `queries`/`hits`, instantaneous `kv_cache_usage`, and raw Prometheus text. Formal counters use `after - baseline`; KV gauges are not subtracted.

`runtime.kv_cache_polling` records interval, successful sample count, elapsed samples, per-DP average/peak/sample count, and global average/peak. `poll_interval_seconds=0` disables it; a failed sample is skipped.

### 17.4 Actual

`actual.by_dp.<rank>` contains formal query/hit deltas, `hit_rate`, after-snapshot KV usage, and polling average/peak. `actual.global_queries`, `global_hits`, and `global_hit_rate` aggregate all DP ranks. KV values are ratios from 0 to 1 (multiply by 100 for percent display).

### 17.5 Interpreting a result

Baseline counters may include probe requests because Prometheus counters do not reset; subtracting baseline removes them. `kv_cache_usage` at after is a single instant, while polling average/peak describe the entire formal run and can therefore be non-zero even when after usage is zero.

## 18. Inspect summary and lightweight Manifest

Inspect stdout contains `run_id`, `mode`, requested/effective/theoretical rates, reachability, `group_reachability`, `groups`, input/output summaries plus totals, `dp_route_counts`, `sends_requests=false`, `log`, and `manifest`.

The lightweight Manifest has `schema_version`, `plugin_version`, `status="inspected"`, timestamped `run_id`, Scenario path/hash, `effective_config`, and `inspect` (`timestamp`, `base_run_id`, `base_output_dir`, `sends_requests`, and `summary`). `prepare` upgrades it in place to a full `prepared` Manifest. Reuse checks the Manifest version, status, timestamp format, timestamped run/output, and Scenario SHA-256.

中文说明：[scenario.example.md](scenario.example.md)  
English
