# AgentX Weka Trace Dataset Performance Benchmarking

This document describes how to benchmark the AgentX Weka Trace dataset in AISBench, covering the introduction, quick start, and notes. For the full parameter reference, see [CLI arguments](../../base_tutorials/all_params/cli_args.md).

## 1. Introduction

### 1.1 What is AgentX

AgentX is a performance stress-testing tool for realistic agent workloads. It replays multi-turn conversation DAGs (including tool calls and context growth) based on real coding-agent session traces collected by Weka (`traces.jsonl`), performs warmup and profiling in a separate process, and reports metrics such as TTFT, ITL, throughput, and effective concurrency.

### 1.2 Why integrate with AISBench

The existing performance benchmarking in AISBench mainly targets conventional LLM request workloads. It cannot adequately emulate the complex load characteristics of real agent applications — multi-turn interaction, long context, request intervals, and concurrent sessions — and therefore cannot fully reflect the inference service performance under agentic scenarios.

To complete the agentic inference performance benchmarking capability of AISBench, the open-source AgentX benchmarking framework and its public evaluation datasets need to be integrated, establishing standardized capabilities for agentic workload execution, performance metric collection, and result analysis.

| Goal                             | What AISBench provides                                                                                                                                                             |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unified entry point              | A single `ais_bench --mode agentx` completes environment probing, command composition, execution, and aggregation, with no need to hand-craft AgentX arguments                          |
| Verifiable environment           | Validates the `agentx-harness` version and Python version, detects mixed installation of the upstream `aiperf`, and self-checks the entry point before launch                          |
| Traceable datasets               | The two official corpora `full` / `256k` are registered with HF repo, revision, and sha256; local corpora are verified byte by byte; the manifest records the corpus source            |
| Determinable comparison scope    | Results are classified as `official_agentx` / `internal_only` / `smoke` / `invalid`, with invalid reasons provided                                                                  |
| Engineering capabilities         | Serial execution across concurrency levels, cooldown between levels, fail-fast, `--reuse` to rerun in place, `--dry-run` to emit the plan only                                       |
| Auditable and redacted           | The final argv, ignored extra args, durations, and error codes are all written to disk; logs and config snapshots are redacted automatically                                         |

### 1.3 Two modes

| Mode                | Use case                                                    | Required arguments                                             |
| ------------------- | ----------------------------------------------------------- | -------------------------------------------------------------- |
| `--mode agentx`     | Standard usage, with the command composed by AISBench       | `--url`, `--model-name`, `--path`, `--agentx-dataset-variant`   |
| `--mode agentx_raw` | Debugging or using new arguments not yet wrapped by AISBench | Only `--agentx-extra-arg`; the command body is described by it  |

`agentx_raw` accepts only the four subprocess arguments `--agentx-python/--agentx-bin/--agentx-env/--agentx-extra-arg`, plus run-control arguments such as `--work-dir/--reuse/--dry-run`. Any other argument fails during parsing.

## 2. Quick Start

### 2.1 Prerequisites

1. Prepare a **dedicated** AgentX environment (Python 3.11\~3.13; it must not share an environment with the AISBench base environment).

   1) Install AgentX according to the repository requirements (AISBench base environment already installed)

   Conda (recommended):
   ```bash
   conda create -n agentx113 python=3.11 -y
   conda run -n agentx113 python -m pip install -r requirements/agentx.txt
   # Query the interpreter path for --agentx-python
   conda run -n agentx113 which python
   ```
   venv:
   ```bash
   python3.11 -m venv /opt/venvs/agentx113
   /opt/venvs/agentx113/bin/pip install -r requirements/agentx.txt
   ```
   `requirements/agentx.txt` pins `agentx-harness==1.0.13`. This environment cannot be shared with AISBench: AISBench depends on `Pillow==11.2.1`, while AgentX requires `pillow~=12.3.0`, so installing them in the same environment fails at dependency resolution.

   2) (Optional) Deploy the AgentX environment with Docker

   Refer to the "Quick Start → Run an existing image" section of the [Image Overview](https://github.com/AISBench/benchmark/blob/master/docker/OVERVIEW.en.md) to start the AgentX container (the AgentX image includes both AISBench and the AgentX environment).
2. Prepare the model service: an OpenAI-compatible Chat Completions endpoint. Adjust the model context window based on the test corpus; it must be at least 256K tokens. When deploying with `vllm`, set `--max-model-len`>=262144, and it is recommended to explicitly enable `--enable-prefix-caching` and `--enable-prompt-tokens-details`.
3. Prepare the tokenizer: the model weight directory must exist; it is passed to AgentX as `--tokenizer`.
4. Prepare the corpus: if `--agentx-dataset-path` is not specified, AgentX downloads the public corpus automatically; in offline environments, pass the directory containing the local `traces.jsonl`.

### 2.2 Launching via CLI

#### 2.2.1 Smoke test

A smoke run validates the pipeline; it is not a formal result:

```bash
ais_bench --mode agentx \
  --url http://127.0.0.1:8000 \
  --model-name qwen3.8 \
  --path /mnt/weight/Qwen3.8-27B-w8a8 \
  --batch-size 8 \
  --agentx-dataset-variant 256k \
  --agentx-run-mode smoke \
  --agentx-python /opt/venvs/agentx113/bin/python \
  --work-dir ./agentx_smoke
```

In `smoke` mode, if `--agentx-duration` / `--agentx-trace-limit` are not specified explicitly, it runs for 60 seconds with 4 traces by default. Smoke results are always marked `comparison_scope=smoke`.

#### 2.2.2 Formal run

Formal comparison requires `benchmark` mode with a duration of at least 900 seconds; if the duration is not specified, the AgentX preset of 3600 seconds is used. Multiple concurrency levels are listed in one `--batch-size` and run serially in order:

```bash
ais_bench --mode agentx \
  --url http://127.0.0.1:8000 \
  --model-name qwen3.8 \
  --path /mnt/weight/Qwen3.8-27B-w8a8 \
  --model-abbr qwen3.8 \
  --dataset-abbr weka_trace \
  --batch-size 1 4 \
  --agentx-dataset-variant 256k \
  --agentx-run-mode benchmark \
  --agentx-duration 900 \
  --agentx-python /opt/venvs/agentx113/bin/python \
  --work-dir ./agentx_bench
```

The example above uses the official public corpus (downloaded by AgentX). For offline or custom corpora, append `--agentx-dataset-path`; in that case the comparison scope of valid results is `internal_only`.

After the run, the terminal prints the execution conclusion for each level and a summary table:

```text
AgentX run_id=agentx_256k_batchsize_1 concurrency=1 status=completed parse=success valid=True scope=official_agentx manifest=...
AgentX run_id=agentx_256k_batchsize_4 concurrency=4 status=completed parse=success valid=True scope=official_agentx manifest=...
```

```text
+---------+----------------------+---------------+-----------+-----------------+-----------------+
| model   | dataset              |   concurrency | status    | benchmark_valid | comparison_scope|
+---------+----------------------+---------------+-----------+-----------------+-----------------+
| qwen3.8 | ..._256k              |             1 | completed | True            | official_agentx |
| qwen3.8 | ..._256k              |             4 | completed | True            | official_agentx |
+---------+----------------------+---------------+-----------+-----------------+-----------------+
```

#### 2.2.3 Advanced configuration

**Local corpus and extra environment variables**

```bash
ais_bench --mode agentx \
  --url http://127.0.0.1:8000 --model-name qwen3.8 \
  --path /mnt/weight/Qwen3.8-27B-w8a8 \
  --batch-size 4 --agentx-dataset-variant 256k \
  --agentx-run-mode benchmark --agentx-duration 900 \
  --agentx-python /opt/venvs/agentx113/bin/python \
  --agentx-dataset-path /mnt/data/cc-traces-weka-062126-256K \
  --agentx-env HF_HUB_OFFLINE=1 --agentx-env NO_PROXY='*' \
  --agentx-env AGENTX_PROCESS_TIMEOUT=7200 \
  --cooldown 10 \
  --work-dir ./agentx_bench
```

`--agentx-env` can be repeated in `KEY=VALUE` format and only affects the AgentX subprocess.

- `--cooldown` controls the wait between levels: 5 seconds by default for multiple levels, while for a single level it waits only when explicitly specified.
- The local corpus specified by `--agentx-dataset-path` has its `traces.jsonl` sha256 checked against the officially registered value in non-smoke mode; a mismatch fails immediately. Smoke runs continue and record the mismatch.

**Appending raw AgentX arguments**


```bash
ais_bench --mode agentx ... \
  --agentx-extra-arg '{ --max-context-length 204800 --random-seed 0 --scenario agentx }'
```

Arguments already composed by AISBench take precedence (`--url`, `--model`, `--concurrency`, `--output-artifact-dir`, corpus source, etc.); same-name fragments in extra args (including their values) are ignored and recorded in `agentx_command.ignored_extra_args`. `--scenario` is determined by extra args, and when absent it is recorded as the entry-point default `agentx`.

**Emit the execution plan only**

```bash
ais_bench --mode agentx ... --dry-run
```

`--dry-run` still performs argument validation, corpus hash verification, and AgentX environment probing, but does not start AgentX or access the model service; it only writes the redacted `execution_plan.redacted.json`.

**Rerun in the same directory**

```bash
ais_bench --mode agentx ... --reuse --work-dir ./agentx_bench
```

`--reuse` reuses the existing directory and only cleans up `manifest.json` and `profile_export*` that will be regenerated in this run; other files are kept. Without `--reuse`, a non-empty artifact directory fails immediately, preventing the previous run's results from being mistaken for this run's artifacts.

**Fully custom command (agentx\_raw)**

```bash
ais_bench --mode agentx_raw \
  --work-dir ./agentx_raw_run \
  --agentx-python /opt/venvs/agentx113/bin/python \
  --agentx-extra-arg '{ --scenario agentx --url http://127.0.0.1:8000 --endpoint /v1/chat/completions --model qwen3.8 --concurrency 8 --public-dataset semianalysis_cc_traces_weka_062126_256k --benchmark-duration 900 }'
```

Extra args only need to contain the options after `profile`; `agentx profile` is prepended by AISBench automatically.

### 2.3 Launching via a configuration file

You can write the model, corpus, and run parameters into a Python configuration file and pass it to `ais_bench` as a positional argument (`--mode agentx` still needs to be specified explicitly):

```bash
ais_bench ais_bench/configs/agentx_example/agentx_weka_trace.py --mode agentx
```

Minimal configuration example (see [agentx_weka_trace.py](https://github.com/AISBench/benchmark/blob/master/ais_bench/configs/agentx_example/agentx_weka_trace.py) in the repository for the complete version):

```python
from ais_bench.benchmark.summarizers.agentx import AgentXSummarizer

models = [
    dict(
        abbr="qwen3.8",
        model="qwen3.8",
        url="http://127.0.0.1:8000",
        endpoint="/v1/chat/completions",
        path="/mnt/weight/Qwen3.8-27B-w8a8",
        trust_remote_code=True,
    )
]

datasets = [
    dict(
        abbr="weka-trace-062126",
        args=dict(
            concurrency=8,
            run_mode="benchmark",
            duration=900,
            trace_limit=None,
            variant="256k",
            dataset_path="/mnt/data/cc-traces-weka-062126-256K",
            agentx_python="~/miniconda3/envs/agentx113/bin/python",
            agentx_bin=None,
            extra_args=None,
            env={},
        ),
    )
]

work_dir = "outputs/agentx/"
summarizer = dict(attr="agentx", type=AgentXSummarizer)
```

When both the configuration file and CLI arguments are present, the CLI fields with the same name take precedence over the configuration file, and run-control arguments such as `--work-dir`, `--reuse`, and `--dry-run` also take effect as usual. Arguments not passed on the command line retain the values from the configuration file.

### 2.4 Execution results

Artifact layout (`<run_id>` looks like `agentx_256k_batchsize_4`, or `agentx_raw_batchsize_<N>` in raw mode):

```text
<work_dir>/
├── configs/<cfg_time>_<pid>.py
├── logs/<run_id>/
│   ├── agentx_stdout.log              # Raw AgentX output (redacted)
│   ├── agentx_adapter.log             # AISBench adapter log
│   └── agentx_summary.log             # Summary table and per-task details
└── performances/<run_id>/
    ├── agentx_raw/                    # Raw AgentX artifacts (profile_export_aiperf.json/csv, console, etc.)
    ├── agentx_summary.json            # Performance results (JSON)
    ├── agentx_summary.csv             # Performance results (CSV)
    ├── execution_plan.redacted.json   # Redacted execution plan
    └── manifest.json                  # Run facts, corpus provenance, verdicts, and metrics
```

Key fields of `manifest.json`:

| Field                                                | Meaning                                                                                                                                                            |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `run_params` / `test_params`                         | Effective values of run mode, duration, trace limit, concurrency level, etc.                                                                                        |
| `agentx_command.parameters`                          | Complete parameters parsed from the final command                                                                                                                   |
| `agentx_command.extra_args` / `ignored_extra_args`   | Pass-through arguments, and fragments ignored due to conflicts                                                                                                      |
| `dataset`                                            | Corpus source, variant, alias, repo/revision, sha256 verification result, `registered`                                                                               |
| `result.status` / `result.parse_status`              | Process status and result parsing status                                                                                                                            |
| `result.benchmark_valid` / `result.comparison_scope` | Whether this is a valid formal result, and its comparison scope                                                                                                     |
| `result.invalid_reasons`                             | First root cause of invalidity; see `all_invalid_reasons` for the full list                                                                                          |
| `result.errors`                                      | Error codes and messages from the process, probing, and result parsing stages                                                                                        |
| `result.summary_json` / `result.summary_csv`         | Point to `agentx_summary.json` and `agentx_summary.csv`; the metrics themselves (request count, throughput, TTFT/ITL/latency p90, etc.) live in those files and the summary table |

`comparison_scope` rules:

| Value             | Meaning                                                                                                                |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `official_agentx` | benchmark mode + official public corpus + valid upstream submission                                                     |
| `internal_only`   | Uses a local corpus; only valid for internal horizontal comparison                                                      |
| `smoke`           | smoke mode, for pipeline validation only                                                                                |
| `invalid`         | Conditions above are not met, e.g. unregistered corpus, hash mismatch, process failure, or insufficient metric coverage  |

## 3. Notes

### 3.1 Version and compatibility

- AISBench pins `agentx-harness==1.0.13` and supports Python 3.11\~3.13.
- The AgentX environment must be isolated from the AISBench environment (`Pillow` versions conflict), and the upstream `aiperf` from PyPI must not be installed at the same time.

### 3.2 Parameter and scope constraints

- `--agentx-dataset-variant` must be specified explicitly as `full` or `256k`.
- `--agentx-run-mode benchmark` requires `--agentx-duration >= 900`; `--agentx-trace-limit` is only available in smoke mode.
- When extra args conflict with arguments composed by AISBench, AISBench takes precedence; `--public-dataset` and `--hf-weka-dataset` are mutually exclusive, and switching the corpus via cross-option fragments is ignored.
