# AISBench Prefix Cache Data Generation and Load-Testing Plugin

[中文](README.md) | English

This standalone AISBench plugin builds datasets with controllable shared prefixes and compares theoretical and measured vLLM Prefix Cache hit rates. It provides offline `inspect`, `prepare`, and `validate` commands, online `run`, and offline `analyze` from saved Prometheus snapshots.

The plugin adds code only under `plugins/prefix_cache`; AISBench core logic is unchanged.

See the [Scenario example](config_examples/scenario.example.json), [complete field reference](config_examples/scenario.example.md), [architecture](ARCHITECTURE.md), [module responsibilities](MODULES.md), and [architecture visualization](../../prefix_cache_architecture.html).

## 1. Prerequisites

The environment must provide:

- Python 3.10 or newer;
- An AISBench checkout whose dependencies can be imported;
- `transformers` for loading the tokenizer that matches the vLLM model;
- `datasets` and `aiohttp` for AISBench Dataset and API load testing;
- A GSM8K JSONL file with at least a `question` field on every line;
- For online `run`, a vLLM Completions service with Prefix Cache enabled.

The tokenizer used during data generation must be identical to the model tokenizer used by vLLM. Otherwise local token lengths, block boundaries, and theoretical hit rates will not match server behavior.

## 2. Installation (with command explanations)

The commands below assume the current directory is the AISBench repository root containing `setup.py`, `ais_bench/`, and `plugins/`.

### 2.1 Create an isolated environment (recommended)

```bash
python -m venv .venv
```

Creates an isolated Python environment in the repository, preventing plugin dependencies from interfering with system packages.

```bash
source .venv/bin/activate
```

Activates the environment in Linux Bash. Subsequent `python` and `pip` commands use `.venv`; alternatively call `.venv/bin/python` explicitly.

### 2.2 Install AISBench

```bash
python -m pip install -e .
```

Installs the checkout and its dependencies in editable mode. `-e` points imports at the working tree, so source changes normally do not require reinstalling. Skip this step if a matching `ais-bench-benchmark` is already installed.

### 2.3 Install the Prefix Cache plugin

```bash
python -m pip install -e ./plugins/prefix_cache
```

This installs the `ais_bench_prefix_cache` package, registers the Prefix Cache Dataset, Inferencer, and vLLM API Model entry points, and installs the `ais-bench-prefix-cache` CLI.

### 2.4 Verify the installation

```bash
ais-bench-prefix-cache --help
```

Verifies the CLI entry point and lists `inspect`, `prepare`, `validate`, `run`, and `analyze`.

If the executable is not on `PATH`, use:

```bash
python -m ais_bench_prefix_cache.cli --help
```

## 3. First use

### 3.1 Copy and edit a Scenario

```bash
cp ./plugins/prefix_cache/config_examples/scenario.example.json ./scenario.json
```

Creates an editable Scenario. Check at least:

- `tokenizer.path`: tokenizer matching vLLM;
- `corpus.path`: local GSM8K JSONL;
- `tokenizer.block_size`: the server Prefix Cache block size;
- `service.dp_size`: target DP count when simulating cold multi-DP routing or creating a warmup plan;
- `service.inference_url`, `metrics_url`, `reset_url`, and `model`: used by online `run`;
- `aisbench.config`: the AISBench Python configuration used for the load test;
- `aisbench.dataset` and `aisbench.model`: all user-visible AISBench Dataset/Model settings, including the reader contract, prompt, `pred_role`, `attr`, streaming, retry, batch size, and generation kwargs. Users do not need to edit the plugin `config.py`.

`inspect`, `prepare`, and `validate` do not contact the service. `run` uses the `service` section for probing, reset, per-DP warmup, the formal load test, and metric collection. One HTTP endpoint may front a single-DP or multi-DP service; orchestration of multiple independent vLLM instances is not supported.

Defaults, constraints, and modes are documented in the [Scenario field reference](config_examples/scenario.example.md). Omitting a field uses the current value from `scenario.example.json`, including the default run/tokenizer/GSM8K path, 100 fixed 1024-token requests, `output.output_key=null` (no output-length field in `requests.jsonl`), a 60% warmup target, one uniform Prefix Group, and DP size 2. `minimum_non_shared_length` is a safety exception: it is derived dynamically from `seed_blocks × block_size`; it remains 16 with the example values.

### 3.2 Prefix Cache data-construction parameters

The Scenario uses a strict allowlist; unknown fields are rejected. The complete hierarchy is:

- `schema_version`;
- `run`: `run_id`, `random_seed`, `output_dir`, `overwrite`;
- `tokenizer`: `path`, `block_size`, `revision`, `trust_remote_code`;
- `corpus`: `path`, `field`, `selection`; selection supports `mode`, `values`, `indices`, and `question_sha256`;
- `requests`: `count`, `input_length`, `output_length`;
  - `input_length` supports `mode`, `value`, `values`, `ranges`, `min`, `max`, `mean`, `std`, and `path`; each `ranges` item allows only `min`, `max`, and `count`;
  - `output_length` supports `mode`, `value`, `min`, `max`, `mean`, `std`, and `path`;
- `output`: `output_key`, which may be `null`, `"max_tokens"`, or `"output_tokens"`;
- `prefix_cache`: `mode`, `target_hit_rate`, `seed_blocks`, `minimum_non_shared_length`, `groups`, and `order`;
  - `groups` supports `count`, `assignment`, and `overrides`;
  - `assignment` supports `mode`, `exponent`, and `weights`;
  - each `overrides.group-N` supports `input_length`, `output_length`, and `corpus_selection`;
  - `order` supports `strategy`;
- `service`: `inference_url`, `metrics_url`, `reset_url`, `model`, `dp_size`, `assume_empty_cache`, `engine_label_map`, `timeout_seconds`, `api_key`, and `poll_interval_seconds`;
- `validation`: `target_warning_pp` and `actual_warning_pp`;
- `aisbench`: `config`, `work_dir`, `extra_args`, `dataset`, and `model`. Offline commands ignore this section; `run` renders it and starts AISBench perf.
  - `dataset`: `abbr`, `input_columns`, `output_column`, `prompt_template`, and `pred_role`. To keep theoretical and actual prompts token-identical, retain the example values for the first three contract fields; `abbr` and `pred_role` may be changed.
  - `model`: `abbr`, `attr`, `stream`, `max_out_len`, `retry`, `batch_size`, and `generation_kwargs`. Older Scenarios receive the example defaults automatically. `attr` currently must be `"service"` to enable AISBench service performance collection, including TTFT.

See the [Scenario complete field reference](config_examples/scenario.example.md) for each field.

#### `requests.jsonl` output fields

```json
"output": {"output_key": null}
```

This follows `extract_qa.py --output-key`:

- `null` (default): write only `question` and `answer`;
- `"max_tokens"`: append `max_tokens`;
- `"output_tokens"`: append `output_tokens`, whose value still comes from this request's `max_tokens`.

The audit `full.jsonl` always retains `max_tokens`, and the AISBench Dataset reads `max_out_len` from it, so omitting the public third field does not change generation length.

#### Input-length modes

`requests.input_length` controls the total input-token length of every formal request. It is composed of the shared prefix, globally unique seed, and GSM8K natural suffix.

Fixed length:

```json
"input_length": {"mode": "fixed", "value": 1024}
```

Explicit list:

```json
"input_length": {
  "mode": "explicit",
  "values": [512, 768, 1024, 2048]
}
```

`values` must contain positive integers. Globally, its length must equal `requests.count`; for a Prefix Group override, it must equal that group's request count.

Multiple closed intervals:

```json
"input_length": {
  "mode": "range",
  "ranges": [
    {"min": 512, "max": 1024, "count": 80},
    {"min": 2048, "max": 4096, "count": 20}
  ]
}
```

Each interval contains `min` and `max`; the sum of all `count` values must equal the applicable request count. Sampling is controlled by `run.random_seed`, so the same configuration reproduces the same sequence.

Truncated normal distribution:

```json
"input_length": {
  "mode": "truncated_normal",
  "min": 512,
  "max": 2048,
  "mean": 1024,
  "std": 256
}
```

Only integer samples in `[min,max]` are accepted. `mean` defaults to the interval midpoint; `std` is derived from the width unless set explicitly, in which case it must be positive. `min=max` is equivalent to fixed length.

CSV length file:

```json
"input_length": {"mode": "csv", "path": "./input_lengths.csv"}
```

The CSV row count must equal the applicable request count and contain one of `input_prompt_tokens`, `content_tokens`, or `input_tokens`.

#### Minimum non-shared length and unique seed

```json
"prefix_cache": {
  "seed_blocks": 1,
  "minimum_non_shared_length": 16
}
```

The unique seed length is:

```text
seed_tokens = seed_blocks × tokenizer.block_size
```

`minimum_non_shared_length` defaults to `seed_tokens` and cannot be smaller. The maximum public-prefix length is:

```text
floor((input_length - minimum_non_shared_length) / block_size) × block_size
```

If the minimum non-shared length exceeds the seed length, the remaining space is filled with the GSM8K natural suffix. Every formal request receives a deterministic `request_random_seed`; its difference seed is globally unique so that tokens after the shared prefix cannot be accidentally reused.

Scenario loading checks that every input-length mode can accommodate the non-shared region and fails before generation when it cannot.

#### Request-order strategies

```json
"order": {"strategy": "input_len_asc"}
```

Supported strategies:

- `sequential`: retain the stable data-generation order;
- `within_group_shuffle`: deterministically shuffle within each Prefix Group, then output by group;
- `interleave`: interleave different Prefix Groups round by round;
- `global_shuffle`: deterministically shuffle all requests;
- `input_len_asc`: sort each Prefix Group from short to long, then rotate across groups; equal lengths retain original order.

Theoretical hit rate is always re-simulated in final send order. With `input_len_asc` + `cold`, the short-to-long sequence passes through four stages: `prepare` reorders lengths and Groups; `requests.jsonl`/`full.jsonl` persist that order; Dataset validates `sequence_index`; and the Inferencer releases requests serially by `(Prefix Group, DP rank)` `lane_sequence`. A request on a lane is sent only after its predecessor completes, so AISBench task concurrency still models a cold first request followed by progressively longer requests that build Cache. Groups and DP ranks have independent watermarks and may run concurrently; global serialization is not required.

#### Reachability and length statistics

The solver computes global and per-Group `reachable_min`/`reachable_max`, whether `target_reachable`, requested/effective/theoretical hit rates, and signed/absolute theoretical-minus-target differences.

If the target is outside the reachable interval, the corresponding boundary solution is selected. Otherwise the nearest block-granular hit total is used. Hit-token precision is a hard constraint; then the solver optimizes the trajectory: warmup balances prefixes by cumulative input, while cold independently searches `(Prefix Group, DP rank)` watermarks, minimizing maximum overshoot, total overshoot, and fallback before approaching the target. This avoids concentrating most prefixes in the first few requests and correcting with zero/short prefixes at the end.

Strict monotonicity is not always possible: a first request on a later Group/DP lane necessarily misses, and some requests lack prefix capacity. The solver therefore prioritizes the smallest overshoot and fallback for the exact total; in extreme search spaces it falls back to deterministic exact lane construction so trajectory optimization cannot change the final effective/theoretical rate.

Manifest input/output length summaries contain `min`, `max`, `mean`, `p50`, `p90`, `p95`, `p99`, and up to ten bins. `inspect` shows these summaries and Group-level reachability.

#### Validation status and exit codes

`analysis.json` uses `PASS` or `PASS_WITH_WARNING`:

- targets outside the reachable interval record `TARGET_UNREACHABLE`;
- theoretical/target differences above `target_warning_pp` record `TARGET_DEVIATION`.

These are display-only: `warning_only=true` and `affects_exit_code=false`. Actual/theoretical differences above `actual_warning_pp` similarly record `ACTUAL_DEVIATION`; configuration, artifact, service-capability, and AISBench execution failures are fatal.

### 3.3 `inspect`: inspect configuration and theoretical range

```bash
ais-bench-prefix-cache inspect --scenario ./scenario.json
```

This command loads the tokenizer and GSM8K, constructs data in a temporary directory, calculates reachability, and displays requested/effective/theoretical hit rates, Group distribution, input/output lengths, and cold DP routing. It never contacts vLLM, sends requests, or leaves the four formal data artifacts in the Scenario `output_dir`.

It still creates a `_YYYYMMDD_HHMMSS` timestamp, writes detailed logs to `output_dir_timestamp/log/<run_id_timestamp>.inspect.log`, and writes a lightweight `manifest.json` under `output_dir_timestamp/result/` with `status="inspected"` and the summary in `inspect.summary`. No standalone `<output_dir>.inspect.json` is created. The JSON summary includes `log` and `manifest` paths.

### 3.4 `prepare`: generate formal artifacts

```bash
ais-bench-prefix-cache prepare --scenario ./scenario.json
```

Deterministically generates and validates:

- `result/<run_id_timestamp>.full.jsonl`;
- `result/<run_id_timestamp>.requests.jsonl`;
- `result/<run_id_timestamp>.manifest.json`;
- `result/<run_id_timestamp>.analysis.json`.

Prompt-generation progress is shown on stderr and advances once per successfully generated prompt:

```text
Generate prompts [###############---------------] 50/100  50%
Generate prompts [##############################] 100/100 100%
{"full":"...","requests":"...","manifest":"...","analysis":"...","log":"..."}
```

The final result JSON is printed on stdout for scripts to parse.

Timestamps use `_YYYYMMDD_HHMMSS`. A standalone `prepare` creates a new timestamp unless it can reuse a matching inspected Manifest. A matching `status="inspected"` Manifest is upgraded in place to `status="prepared"`; `run` can reuse a matching inspected/prepared Manifest so preview, generation, load test, and validation share one timestamp directory.

For example:

```text
run_id:    gsm8k-prefix-cache-60
output_dir: ./outputs/gsm8k-prefix-cache-60
```

May produce:

```text
./outputs/gsm8k-prefix-cache-60_20260825_123456/
├── log/
│   └── gsm8k-prefix-cache-60_20260825_123456.prepare.log
└── result/
    ├── gsm8k-prefix-cache-60_20260825_123456.full.jsonl
    ├── gsm8k-prefix-cache-60_20260825_123456.requests.jsonl
    ├── gsm8k-prefix-cache-60_20260825_123456.manifest.json
    └── gsm8k-prefix-cache-60_20260825_123456.analysis.json
```

Normal workflows do not require manually changing `run_id` or `output_dir`, and no extra `.inspect.json` appears beside the base output directory. The plugin scans `<output_dir>_timestamp/result/<run_id_timestamp>.manifest.json` and validates Manifest version, status, timestamped run/output, and `scenario_sha256`; changing the Scenario automatically invalidates old Manifests and creates a new timestamp.

Files are not overwritten by default. To rebuild explicitly:

```bash
ais-bench-prefix-cache prepare --scenario ./scenario.json --overwrite
```

`--overwrite` replaces only the four files for this run in the current timestamp directory; it does not clear the entire output directory. An inspected-only matching Manifest is upgraded automatically, while a prepared Manifest is not reused as an inspect placeholder.

### 3.5 `validate`: validate existing artifacts

```bash
ais-bench-prefix-cache validate --manifest ./outputs/gsm8k-prefix-cache-60_<timestamp>/result/gsm8k-prefix-cache-60_<timestamp>.manifest.json
```

Without generating data or contacting vLLM, this checks that Manifest/full/requests row counts agree, `sequence_index` is contiguous, requests contain exactly `question`, `answer`, and the optional field selected by `output.output_key`, rows correspond one-to-one, and full/requests SHA-256 values match the Manifest. It detects manual edits, truncation, reordering, or the wrong artifact version.

Detailed logs are written to `log/<run_id_timestamp>.validate.log`; the terminal prints only the validation result JSON.

### 3.6 `run`: execute a vLLM Prefix Cache load test

```bash
ais-bench-prefix-cache run --scenario ./scenario.json
```

The end-to-end flow validates or prepares timestamped artifacts, probes each DP, resets Prefix Cache (or records `ASSUME_EMPTY_CACHE`), performs per-Group × DP warmup in warmup mode, captures the formal baseline, runs AISBench `perf`, captures after metrics, calculates per-DP/global actual hit rates, and writes runtime data, actual/theoretical differences, and warnings to `result/<run_id>.analysis.json`. Warmup completes before baseline and is excluded from formal throughput, latency, and hit-rate statistics.

Plugin flow logs go only to `output_dir_timestamp/log/<run_id_timestamp>.run.log`; they are not echoed to the CLI. AISBench child stdout/stderr remains inherited and visible in the CLI. Logs include execution context, artifact reuse/auto-prepare, probes, reset, every Group × DP warmup, baseline/after metrics, rendered AISBench config and command, KV polling, per-DP deltas, global hit rate, and warnings. Only prompt length and SHA-256 are logged; prompt text, API keys, Authorization headers, and raw request bodies are not.

Dataset, Model, and Inferencer code runs inside the formal AISBench child process and uses the AISBench `AISLogger`/handle with debug-level details. With the default `aisbench.work_dir="./outputs/default"`, AISBench writes child output to `./outputs/default/<AISBench timestamp>/logs/infer/*.out`; changing `work_dir` moves these logs. AISBench's global level is INFO by default, so set DEBUG to persist debug messages.

Formal requests use vLLM SSE streaming by default because `aisbench.model.stream=true`. Start time, first chunk, and later chunk timestamps feed `DefaultPerfSummarizer` metrics TTFT, TPOT, ITL, E2EL, and throughput; summaries are under `aisbench.work_dir/performances/<model-abbr>/`. With `stream=false`, Prefix Cache hit rate still works but full TTFT/TPOT/ITL cannot be computed. DP probes and plugin warmup use separate non-streaming requests before the formal baseline and are excluded from these metrics.

Plugin Group × DP warmup and AISBench's `--num-warmups` are independent. The former is controlled by `prefix_cache.mode="warmup"` and runs before baseline; the latter belongs to the AISBench perf child. To ensure only formal requests follow baseline, use `"extra_args": ["--num-warmups", "0"]`.

Temporarily override the AISBench config:

```bash
ais-bench-prefix-cache run --scenario ./scenario.json --config ./my_prefix_cache_perf.py
```

Multi-DP uses one HTTP endpoint and requires `X-data-parallel-rank` routing plus per-DP Prometheus metrics with an `engine` label. Every DP is warmed independently; multi-instance orchestration is not supported.

### 3.7 `analyze`: recompute from saved metric snapshots

```bash
ais-bench-prefix-cache analyze \
  --manifest ./outputs/gsm8k-prefix-cache-60_<timestamp>/result/gsm8k-prefix-cache-60_<timestamp>.manifest.json \
  --baseline ./baseline.prom \
  --after ./after.prom
```

This command does not connect to vLLM or run AISBench. It parses two Prometheus text snapshots, recomputes formal counter deltas, and writes analysis next to the Manifest. The current CLI writes `analyze` and `validate` plugin logs to the same `log/<run_id>.validate.log`; a later invocation recreates that log.

## 4. Recommended workflow

```bash
ais-bench-prefix-cache inspect --scenario ./scenario.json
ais-bench-prefix-cache prepare --scenario ./scenario.json
ais-bench-prefix-cache validate --manifest <manifest-path>
ais-bench-prefix-cache run --scenario ./scenario.json
```

This permits manual review before load testing. `prepare` and `run` reuse matching Manifests; `run` directly reuses a matching formal artifact in the timestamp directory.

## 5. `cold` and `warmup`

### cold

- Every `(group_id, dp_rank)` starts at watermark zero;
- requests in a Group use round-robin targeted DP routing;
- the plugin preserves order within each lane;
- strict per-DP theoretical hit rates can be reported.

### warmup

- Each Prefix Group and each DP rank is warmed independently;
- warmup is excluded from `requests.jsonl`, theoretical denominators, and formal metric deltas;
- the global theoretical rate is valid; per-DP values primarily show measured metrics.

The plan is stored in `warmup.plan` in `result/<run_id_timestamp>.manifest.json`. `prepare` only creates the plan; `run` sends warmup requests and captures baseline after all warmups finish.

## 6. Layered artifacts

Formal data is under the timestamped `result/` directory and detailed logs under its sibling `log/` directory.

### `<run_id>.full.jsonl`

Full audit data contains these fixed fields:

| Field | Meaning |
|---|---|
| `request_id` | Stable request ID. |
| `sequence_index` | Zero-based final send order. |
| `group_id` | Prefix Group. |
| `occurrence_index_within_group` | Occurrence number within the Group. |
| `dp_rank` | Target DP rank in cold mode; `null` for warmup formal requests. |
| `lane_sequence` | Sequence within a cold `(group_id, dp_rank)` lane; `null` for warmup. |
| `target_input_tokens` | Configured input length. |
| `actual_input_tokens` | Tokenizer re-encoded length. |
| `max_tokens` | Maximum output tokens. |
| `shared_prefix_tokens` | Shared-prefix tokens used by this request. |
| `seed_tokens` | Globally unique seed length. |
| `natural_suffix_tokens` | GSM8K natural suffix after the seed. |
| `question` | Final complete prompt. |
| `answer` | AISBench-compatible placeholder, currently `"none"`. |
| `gsm_indices` | Zero-based GSM8K rows used for the natural suffix. |
| `gsm_hashes` | SHA-256 values of normalized questions. |
| `canonical_prefix_sha256` | Canonical-prefix fingerprint for the Group. |
| `seed_sha256` | Fingerprint of this request's unique seed. |
| `request_random_seed` | Deterministic seed used to construct the request seed. |
| `watermark_before` | Lane watermark before the request. |
| `theoretical_hit_tokens` | Theoretical hit tokens. |
| `watermark_after` | Lane watermark after completion. |
| `theoretical_hit_rate` | `theoretical_hit_tokens / actual_input_tokens`. |
| `divergence_block_sha256` | Difference-block fingerprint, currently equal to `seed_sha256`. |
| `divergence_unique` | Whether the difference block passed global uniqueness checks. |
| `collision_status` | Collision-check status; successful artifacts use `"pass"`. |

### `<run_id>.requests.jsonl`

Minimal input rows always write `question`, then `answer`, and optionally the third field selected by `output.output_key`:

- `question`: final complete prompt;
- `answer`: currently `"none"`;
- `max_tokens` or `output_tokens`: optional maximum output length.

`full.jsonl` always retains `max_tokens`. Routing and audit fields remain there and do not pollute the generic request format.

### `<run_id>.manifest.json`

The Manifest is the reproducibility and validation entry point.

| Field | Meaning |
|---|---|
| `schema_version`, `plugin_version` | Manifest and plugin contract versions. |
| `run_id` | Timestamped run ID. |
| `scenario_path`, `scenario_sha256` | Absolute Scenario path and source hash. |
| `effective_config`, `effective_config_sha256` | Defaults-filled, path-resolved config and hash. |
| `corpus_sha256` | GSM8K file hash. |
| `tokenizer` | Tokenizer source/class/vocabulary/special tokens/block/fingerprint. |
| `requests` | Count, total input tokens, and length summaries. |
| `prefix_cache` | Mode, target, theoretical values, reachability, adjustments, and validation. |
| `groups` | Per-Group canonical source, maximum prefix, reachability, and theoretical rate. |
| `dp` | DP count and cold routing strategy. |
| `warmup` | Enabled flag and per-Group/per-DP warmup plan. |
| `divergence` | Unique difference-block policy, count, and collision status. |
| `artifacts` | Names, paths, rows, sizes, and hashes for full/requests/analysis. |

Important nested fields include tokenizer metadata; request count and `*_length_summary` (`min`, `max`, `mean`, `p50`, `p90`, `p95`, `p99`, `bins`); prefix-cache target/reachability/validation fields; per-Group canonical and GSM8K fields; DP `size`/`cold_route_strategy`; warmup `enabled`/`plan`; divergence strategy/count/status; and artifact metadata. Each bin has `min`, `max`, and `count`.

`api_key` is never written in plaintext: `effective_config.service` contains only `api_key_configured`.

### `<run_id>.analysis.json`

Fixed fields include `schema_version`, `run_id`, `status`, requested/effective/theoretical hit rates, target differences (`target_difference_pp` currently equals absolute difference), `validation` (`status`, `target_reachable`, `warning_only`, `affects_exit_code`), `theory` (`input_tokens`, `hit_tokens`, `groups`, `dp`), and a `warnings` array. `TARGET_UNREACHABLE` carries the requested target and reachable bounds; `TARGET_DEVIATION` carries `difference_pp`.

Generation finishes with `status="prepared"`; `run` changes it to `"complete"`; offline `analyze` uses `"analyzed"`. Online runs add `runtime.metrics_baseline/metrics_after` (raw Prometheus and per-DP cumulative values), `actual`, theory/actual differences, and `validation.actual_status`. Warnings affect presentation only and not a successful exit code.

During formal scoring, `run` polls `metrics_url` every `service.poll_interval_seconds` for KV usage. KV is an instantaneous gauge that may return to zero after scoring, so sampling must happen during the run. Samples are stored in `runtime.kv_cache_polling` with interval, count, per-DP peak/average/sample count, global peak/average, and detailed elapsed samples. These values are also copied to `actual.by_dp.*.kv_cache_usage_peak/avg` and global fields; `kv_cache_usage` remains the instantaneous after-snapshot value. Set the interval to 0 to disable polling; an individual failed sample is skipped without aborting the run.

### `inspect` Manifest and CLI result fields

The `inspect` terminal JSON includes `run_id`, `mode`, requested/effective/theoretical rates, reachability, `group_reachability`, `groups`, input/output tokens, `dp_route_counts`, `sends_requests` (always `false`), `log`, and `manifest`.

The lightweight inspect Manifest contains `schema_version`, `plugin_version`, `status="inspected"`, `run_id`, Scenario path/hash, `effective_config`, and `inspect` (`timestamp`, `base_run_id`, `base_output_dir`, `sends_requests`, `summary`). `prepare` upgrades it in place to a full `status="prepared"` Manifest with requests, Groups, DP, warmup, and artifacts.

The final CLI JSON contains `full`, `requests`, `manifest`, `analysis`, and `log` for `prepare`; inspect summary plus `log`/`manifest` for `inspect`; `ok`, `rows`, and `run_id` for `validate`; updated full analysis for `run`; and offline analysis for `analyze`. `validate`, `run`, and `analyze` write logs; their returned JSON does not add a separate `log` field.

## 7. Exit codes

- Theory/target difference above `target_warning_pp`: `TARGET_DEVIATION`;
- target outside the reachable interval: `TARGET_UNREACHABLE`;
- actual/theoretical difference above `actual_warning_pp`: `ACTUAL_DEVIATION`;
- all three are warnings only and do not change an otherwise successful exit code;
- configuration errors, damaged artifacts, insufficient service capability, and AISBench failures return non-zero.

## 8. Frequently asked questions

### Why is the target hit rate not exactly equal?

Public prefixes are block-aligned, and cold mode is constrained by order, Groups, DP routing, and watermarks. The plugin selects the nearest reachable result and records requested, effective, theoretical, and the reason for any difference.

If multiple Prefix Groups select the same first GSM8K sample, the plugin rotates Group samples first; only if all rotations collide does it use a deterministic Group marker fallback. Small corpora or duplicate indices therefore do not make `prepare` fail immediately.

### Why is warmup excluded from formal statistics?

Warmup only establishes Cache. Counting it in request totals, throughput, latency, or hit rate would mix setup cost with the formal result.

### Why usually do I not need to change `run_id` after editing a Scenario?

A standalone `prepare` gets a new second-level timestamp. In the recommended `inspect → prepare → run` workflow, later commands reuse the matching Manifest. The timestamp is appended to both `run_id` and `output_dir`, so manual renaming is unnecessary; use `--overwrite` only when explicitly rebuilding the same timestamp directory.

中文说明：[README.md](README.md)  
English
