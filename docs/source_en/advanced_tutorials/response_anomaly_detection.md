# Response Anomaly Detection

## Overview

AISBench integrates msProbe's `ILLDetector` to automatically detect generation anomalies in LLM responses while running inference evaluations. Detection results cover the following types:

| `anomaly_type` | `anomaly_type_name` | Meaning |
| -------------- | ------------------- | ------- |
| 0 | `normal` | Normal |
| 1 | `rare_character` | Rare character |
| 2 | `garbled` | Garbled text |
| 3 | `repetition` | Repetition |
| 4 | `nan_value` | NaN value |

**Anomaly detection results do not affect the original evaluation metrics**: anomalous Cases are not rewritten as inference failures; accuracy/performance metrics are computed as usual, and anomaly information is an independent audit result.

Detection runs through msProbe's `ILLDetector(config_path, mtype_path, tk2cat_path).run(...)`; all three file paths can be configured in AISBench (see [Configuration](#configuration)).

---

## Prerequisites

1. **Inference backend**: Response anomaly detection currently supports only the vLLM Chat API model configurations `vllm_api_general_chat`, `vllm_api_stream_chat`, and `vllm_api_stream_chat_multiturn`. Other model backends are not supported yet.
2. **Evaluation modes**: Only the `all`, `infer`, and `infer_judge` generation chains are supported; the `perf` / `perf_viz` performance modes, as well as Agent / function-call and other custom chains, do not support this feature, and enabling it raises an explicit error during config initialization.
3. **Service requirements**: The service response must contain `token_ids` (or `tokens`) and `topk_logprobs` fields; Cases missing these fields are recorded with a `skipped` status.
4. **Optional dependency**: The `response_anomaly` extra must be installed (see the next section).

> 💡 Detection is serially bound to the inference stage: after inference finishes, the workflow starts detection and waits for it to complete (the dedicated status board prints the final result) before entering the subsequent Judge / Eval / Summary stages, guaranteeing that detection results and payload archives are on disk when the inference stage exits. See the [mode documentation](../base_tutorials/all_params/mode.md) for the full mode support matrix.

---

## Installing Dependencies

Response anomaly detection relies on the optional package `mindstudio-probe`, installed through the AISBench extra:

```bash
pip install 'ais-bench-benchmark[response_anomaly]'
```

During installation, pip downloads and builds the pinned msProbe source from GitCode, so the environment needs Git and network access.

Without this dependency, evaluation still runs normally, but all Cases are marked with the `unavailable` status in the detection results (see [Detection Results](#detection-results)); re-run after installation to restore detection.

---

## Quick Start

**1. Add the `response_anomaly` field to the model config** (the minimal configuration only needs one of `model_name` or `model_path`):

```python
models = [
    dict(
        abbr='qwen3-30b',
        attr='service',
        response_anomaly=dict(
            model_name='Qwen3-30B-A3B',   # or provide only model_path; the model name is taken from it
        ),
    ),
]
```

**2. Run the evaluation with `--response-anomaly` on the command line**:

```bash
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --response-anomaly
```

> ⚠️ The **feature switch is command-line only**: add `--response-anomaly` to the command to enable detection (omit it to disable). The `response_anomaly` entry in the config file carries only non-switch settings (`payload_retention`, `payload_storage`, etc.).

**3. Inspect the detection results**: after inference and detection finish, the results are written to `<work_dir>/response_anomaly/<model abbr>/<dataset abbr>.jsonl`, one Case per line (see [Runtime Flow and On-Disk Layout](#runtime-flow-and-on-disk-layout) for the full layout).

---

## Configuration

### Global Configuration

The global `response_anomaly` config controls the payload retention policy and storage format:

```python
response_anomaly = dict(
    payload_retention='anomalies',  # all | anomalies | none
    payload_storage=dict(
        format='jsonl',
        compression='zstd',
        compression_level=3,
        rows_per_shard=2000,
    ),
)
```

`payload_retention` determines which payloads are kept after detection:

| Value | Behavior |
| ----- | -------- |
| `all` | Keeps every payload and atomically promotes the staging data to the official archive after detection without re-compressing |
| `anomalies` (default) | Keeps only detected anomalies plus detection-failed/unavailable Cases |
| `none` | Keeps no payload |

All three modes keep the standalone detection results. The command-line parameter `--response-anomaly-payload-retention` overrides the config file value.

> 💡 Runtime details: results are written to disk in batches, and the status is refreshed at most once per second; msProbe token-category maps are cached per model and EOS token to avoid re-parsing large JSON files for every Case.

### Model-Level Configuration (msProbe)

Model-specific msProbe configuration goes into the model config:

```python
models = [
    dict(
        abbr='qwen3-30b',
        attr='service',
        response_anomaly=dict(
            model_name="",   # Model name, for example Qwen3-30B-A3B
            model_path="",   # Local model directory, for example /home/Qwen3-30B-A3B; optional, used to auto-generate configs
            msprobe_config_path="",  # Optional; msProbe algorithm-threshold config.yaml path for manual threshold tuning
            msprobe_mtype_path="",  # Optional; msProbe mtype_config.json path mapping model names to BOS/EOS token ids
            msprobe_token2category_dir="",  # Optional; msProbe token2category directory holding per-model token-id-to-character-category maps
        ),
    ),
]
```

**Rules for `model_name`**: when it is not configured explicitly, the **model name is taken from the model path** (`model_path`, or the model `path` field; e.g. `/home/Qwen3-30B-A3B` → `Qwen3-30B-A3B`), matching the config generator's default. When neither `model_name` nor a model path is available (e.g. only the explicit msProbe resource paths are configured), the task fails fast at startup and asks for an explicit `model_name`, instead of silently running detection with a wrong model name.

`model_name` must be consistent with msProbe's `mtype_config.json` and the token-category mapping.

When `msprobe_mtype_path` / `msprobe_token2category_dir` are not provided, the default files inside the msProbe package are used.

### Auto-Generation and Manual Generation of msProbe Configs

When `model_path` is configured, the msProbe configs are auto-generated into `<work_dir>/response_anomaly_config/<model abbr>/` (auto-generation never overwrites an existing `config.yaml`, so manually tuned thresholds are preserved). They can also be generated manually:

```bash
ais_bench-gen-response-anomaly-config \
  --model-path /home/Qwen3-30B-A3B \
  --model-name Qwen3-30B-A3B \
  --output-dir ./msprobe_configs
```

---

## Runtime Flow and On-Disk Layout

### Automatic Request Parameter Injection

When enabled, AISBench adds `logprobs=True` and a fixed `top_logprobs=20` to the service inference requests; the value is constrained by the detection algorithm and cannot be configured externally. For vLLM backends, `return_token_ids=True` and `return_tokens_as_token_ids=True` are also appended to obtain token ids; if the server version is too old to support these parameters, requests may fail — upgrade vLLM in that case.

### Detection Flow

1. **Inference stage**: the full payload is written directly to `response_anomaly/<model>/payload_staging/<dataset>/*.jsonl.zst`, and predictions only keep lightweight results from the start;
2. **Detection stage**: after inference finishes, the detection thread streams and decompresses the staging data and calls msProbe; detection results are written to `response_anomaly/<model>/<dataset>.jsonl`;
3. **Archive finalization**: after detection, the staging data is retained or cleaned according to `payload_retention`.

The status panel shows the config preparation, detector loading, streaming detection, and archive finalization stages.

### On-Disk Layout

Files produced by detection are laid out under `<work_dir>` as follows:

```text
<work_dir>/
├── predictions/<model abbr>/<dataset abbr>.jsonl        # Inference results (lightweight, no token/logprob payload)
├── response_anomaly/
│   └── <model abbr>/
│       ├── <dataset abbr>.jsonl                          # Detection results, one Case per line
│       ├── payload_staging/<dataset abbr>/               # Transient staging during inference; cleaned after detection
│       │   └── part-*.jsonl.zst
│       └── payload/<dataset abbr>/                       # Payload archive; absent when payload_retention is none
│           ├── payload_manifest.json                     # Archive manifest (per-shard rows, sizes, sha256)
│           └── part-*.jsonl.zst                          # Compressed payload shards (at most rows_per_shard Cases each)
├── response_anomaly_config/                              # Present only when auto-generated via model_path
│   └── <model abbr>/
│       ├── configs/
│       │   ├── config.yaml                               # Detection algorithm thresholds (never overwritten once present)
│       │   └── mtype_config.json                         # Model name to BOS/EOS token id mapping
│       └── token2category/
│           └── <model name>_<vocab size>.json            # Token id to character-category mapping
└── logs/
    └── response_anomaly/<model abbr>/<dataset abbr>.out  # Detection-specific log
```

Path-by-path notes:

- **Detection results** `response_anomaly/<model abbr>/<dataset abbr>.jsonl`: one Case per line; field semantics are described in [Detection Results](#detection-results).
- **Payload archive** `response_anomaly/<model abbr>/payload/<dataset abbr>/`: `all` keeps every Case, `anomalies` keeps only anomalous plus detection-failed/unavailable Cases, and `none` keeps nothing (the directory does not exist). To read the data, decompress the `part-*.jsonl.zst` shards with zstandard and parse each line as JSON; `payload_manifest.json` records per-shard row counts, sizes, and sha256 checksums for integrity verification. Note: under `anomalies`, even when there is nothing to retain, an archive directory containing only an empty manifest is still published to indicate the archiving flow completed successfully — it is not a leftover file.
- **Transient files**: `payload_staging/` receives payload records during inference and is cleaned automatically after detection; `.<dataset>.payload-build-*` build directories left by an interrupted detection are cleaned automatically when the next detection starts; `status_tmp/tmp_ResponseAnomaly.json` is a runtime status file (detection progress and per-type statistics) removed together with the status directory when the workflow ends.
- **Auto-generated msProbe configs** `response_anomaly_config/<model abbr>/`: generated only when `model_path` is configured and explicit mtype/token2category paths are absent. An existing `config.yaml` is never overwritten (manually tuned thresholds are preserved), and `mtype_config.json` supports multi-model merging across repeated generations.
- **Detection log** `logs/response_anomaly/<model abbr>/<dataset abbr>.out`: records the detection run for the model/dataset group, including detector initialization failures and per-Case failure reasons.

---

## Detection Results

The detection results `response_anomaly/<model abbr>/<dataset abbr>.jsonl` contain one Case per line with `id`, `uuid`, `is_anomaly`, `anomaly_type` (0: normal, 1: rare character, 2: garbled, 3: repetition, 4: NaN value), `anomaly_type_name` (the type-name string such as `normal`/`garbled`/`repetition`, which is more convenient for statistics), and `detection_status`.

The `detection_status` values in the detection results are:

| Status | Meaning | Troubleshooting |
| --- | --- | --- |
| `completed` | msProbe was invoked and returned a detection result | Nothing to do |
| `skipped` | The inference response did not carry token ids or top-k logprobs | Check whether the service supports and returns `logprobs` / `top_logprobs` / token id fields |
| `unavailable` | `mindstudio-probe` (the response_anomaly extra) is not installed | Install the optional dependency per the [Installation Guide](../get_started/install.md) and re-run |
| `failed` | An exception occurred during invocation or input conversion | Check the `reason` field of the Case result (error type and summary) and the detection logs |

Detection-specific logs are located at `<work_dir>/logs/response_anomaly/<model>/<dataset>.out`; detection progress and per-type statistics can also be found in the `<work_dir>/status_tmp/tmp_ResponseAnomaly.json` status file.

---

## Resuming Runs

When using `--reuse`, existing detection results are inherited by matching both the Case `id` and `uuid` (a changed `uuid` means the Case was re-inferred, so it never gets a stale result):

- Cases with `completed` status are not re-detected;
- Cases with `skipped` / `failed` / `unavailable` status are re-detected on resume.

`--reuse` runs must keep the payload retention policy of the original work directory.
