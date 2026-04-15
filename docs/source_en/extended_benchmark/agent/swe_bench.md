# SWEbench User Guide

SWE-bench is a benchmark for evaluating how well large language models solve real-world software issues collected from GitHub. Given a repository and an issue, the model is expected to generate a patch that fixes the described problem.

## 1. Feature Overview

`ais_bench` currently supports the following SWEbench capabilities:

- Datasets: `full`, `verified`, `lite`, `multilingual`
- Tasks:
  - `infer`: call `mini-swe-agent` to generate patches (`model_patch`)
  - `eval`: call the SWE-bench harness to run evaluation and count resolved instances
- Result summary: output key metrics such as `accuracy`, `submitted_accuracy`, and `resolved_instances`

Directory `ais_bench/configs/swe_bench_examples/` already provides 4 example configs:

- `mini_swe_agent_swe_bench_lite.py`
- `mini_swe_agent_swe_bench_verified.py`
- `mini_swe_agent_swe_bench_full.py`
- `mini_swe_agent_swe_bench_multilingual.py`

## 2. Prerequisites

Before running, make sure the following dependencies are available:

1) Install `mini-swe-agent` (required for infer)

```bash
pip install mini-swe-agent
```

2) Install the SWE-bench harness (required for eval)

```bash
git clone https://github.com/SWE-bench/SWE-bench.git
cd SWE-bench
pip install -e .
cd -
```

3) Docker is available (both infer and eval depend on containerized environments)

```bash
docker --version
docker ps
```

## 3. Minimal Configuration (Run First, Tune Later)

It is recommended to start from `mini_swe_agent_swe_bench_lite.py` and only modify the three fields in `models[0]`:

- `model`: model name (required)
- `url`: model service endpoint (OpenAI-compatible API)
- `api_key`: service key (use `EMPTY` for local services)

Example (local vLLM setup):

```python
models = [
    dict(
        attr="local",
        abbr="swebench",
        type="LiteLLMChat",
        model="qwen3",
        api_key="EMPTY",
        url="http://127.0.0.1:8000/v1",
        batch_size=1,
        generation_kwargs=dict(),
    )
]
```

### Dataset Path Notes

In the example configs, `path=""` by default, which means online loading from Hugging Face is preferred.

- You can keep `path=""` to fetch data online directly
- For offline usage, change `path` to a local parquet file or directory (supports `data/<split>-*.parquet`)

### First-Run Recommendations

- Start with the `lite` dataset
- Use `batch_size=1`
- Keep `step_limit=200` (default in examples; do not change initially)

## 4. Run Commands

Run the following in the repository root (`config` is the config file path):

```bash
ais_bench ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_lite.py
```

The command above runs the full pipeline (`all`). You can also run it step by step:

```bash
# Inference only, generate predictions
ais_bench ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_lite.py -m infer

# Evaluate based on existing predictions
ais_bench ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_lite.py -m eval
```

### Resume from Checkpoint

Use `--reuse` to skip completed instances, which is useful after interruptions:

```bash
ais_bench ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_lite.py -m infer --reuse
```

## 5. How to Read Outputs

The default output directory is `outputs/default/<timestamp>/`. Focus on:

- Inference outputs:
  - `predictions/swebench/swebench_*.json`
  - Each `instance_id` contains `model_patch`
- Evaluation outputs:
  - `results/swebench/swebench_*.json`
  - Key fields:
    - `accuracy`: `resolved_instances / total_instances`
    - `submitted_accuracy`: `resolved_instances / submitted_instances`
    - `resolved_instances` / `unresolved_instances` / `error_instances`
    - `harness_exit_code`: harness exit code

## 6. Common Issues and Troubleshooting (SWEB Error Codes)

The following error codes come from `SWEB_CODES`. You can also refer to the full FAQ:

- Chinese FAQ: `docs/source_zh_cn/faqs/error_codes.md`

### 1) `SWEB-DEPENDENCY-001`: Missing mini-swe-agent

- Symptom: infer fails to start with dependency import errors
- Cause: `mini-swe-agent` is not installed
- Fix: run `pip install mini-swe-agent`

### 2) `SWEB-DEPENDENCY-002`: Missing SWE-bench harness

- Symptom: harness import error during eval
- Cause: SWE-bench is not installed, or not visible in the current environment
- Fix: install SWE-bench as described in "Prerequisites", and make sure you are using the same Python environment

### 3) `SWEB-PARAM-001`: Empty model configuration

- Symptom: prompt indicates model is not configured
- Cause: `models[0]['model']` is empty or only whitespace
- Fix: configure `model/url/api_key`, and ensure `model` is non-empty at minimum

### 4) `SWEB-DATA-002` / `SWEB-FILE-003`: Dataset loading failure

- Symptom: online loading fails, or local parquet files cannot be found
- Cause:
  - Online mode: network or Hugging Face access issues
  - Local mode: `path` does not exist, or directory layout does not match split parquet rules
- Fix:
  - Switch to local parquet if online loading fails
  - Ensure local path follows: `<root>/data/test-*.parquet` or `<root>/test-*.parquet`

### 5) `SWEB-FILE-001`: Predictions file not found

- Symptom: `-m eval` reports missing predictions
- Cause: infer was not run first, or work_dir/reuse points to a different location
- Fix: run `-m infer` first, and ensure eval and infer use the same config/output directory

### 6) `SWEB-RUNTIME-001` / `SWEB-RUNTIME-002`: Container or harness runtime failure

- Symptom: Docker image pull failure, or evaluation runtime errors
- Cause: unavailable images, network issues, or insufficient container runtime environment
- Fix:
  - Check `docker ps` first
  - Verify required images can be pulled
  - Retry with `--reuse` to avoid recomputing completed instances

## 7. Advanced Tips (Optional)

- For initial debugging, use `lite` first, then switch to `verified/full` after the pipeline is stable
- To reduce empty patches, prioritize improving model capability and agent prompt templates
- During evaluation, focus on `empty_patch_instances` and `error_instances`; they are often more actionable than `accuracy` in early iterations

