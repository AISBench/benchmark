# DeepSWE User Guide

DeepSWE is a benchmark for evaluating coding agents on original, long-horizon software engineering tasks, built by Datacurve. It consists of **113** original tasks spanning **91** active open-source repositories and **5** programming languages (TypeScript, Go, Python, JavaScript, Rust). Unlike SWE-bench-style benchmarks that mine merged fixes from public GitHub repositories, every DeepSWE task is written from scratch and never merged upstream, so its reference solution stays out of pretraining corpora; in addition, each task is graded by a hand-written verifier that checks observable software behavior, which mitigates both data contamination and verification distortion. Official page: `https://deepswe.datacurve.ai/`.

> **Note**: since the official task images are all x86, DeepSWE currently supports evaluation on x86 environments only; ARM is not supported yet.

## 1. Feature Overview

`ais_bench` currently supports the following DeepSWE capabilities:

- Datasets: `full` (full set, v1.1, 113 tasks) and `mini` (small-scale sample)
- Tasks: the general Harbor-based agent evaluation chain (`--mode agent`); the agent autonomously locates code, edits across files, and iterates inside a container environment, and the verifier scores the result automatically
- Result summary: output key metrics such as `avg_score`, `reward_distribution`, `exception_distribution`, and `pass@k`; verifier artifacts additionally contain auxiliary metrics such as `f2p_total` / `p2p_passed`

> 💡 DeepSWE reuses AISBench's general Harbor-based agent evaluation chain. For environment preparation (agent runtime container, offline agent dependency packages), parameter references, and other general topics, see [Agent Evaluation (Harbor)](../../base_tutorials/scenes_intro/agent_benchmark.md).

## 2. Prerequisites

Before running, make sure the following dependencies are available:

1) Docker is available (version >= 20.10.0, docker compose version >= 2.0.0)

```bash
docker --version
docker ps
```

2) Install AISBench and the standalone agent dependency set (including Harbor) in a Python 3.12 environment

```bash
pip install -r requirements/agent.txt
```

3) The model under test: a service that follows the OpenAI chat/completions API and supports tool calls.

## 3. Dataset and Image Preparation

### 3.1 Datasets

| Dataset | Link | Notes |
| ------- | ---- | ----- |
| DeepSWE (full) | https://github.com/AISBench/deep-swe | Harbor-format full dataset (v1.1) provided by AISBench |
| DeepSWE mini | https://modelers.cn/datasets/AISBench/DeepSWE-mini | Small-scale sampled dataset, suitable for a quick first run |

### 3.2 Images

Every case has a corresponding Docker image whose name is defined in the dataset. It is recommended to load the packaged images in advance (base OS debian:12, x86_64 only):

| Packaged image resource | Link | CPU arch | Base OS |
| ----------------------- | ---- | -------- | ------- |
| `deep-swe-v1.1-task-images.tar.gz` | https://aisbench.obs.cn-north-4.myhuaweicloud.com/deepswe/deep-swe-v1.1-task-images.tar.gz | x86_64 | debian:12 |

```bash
docker load -i deep-swe-v1.1-task-images.tar.gz
```

> ⚠️ Where to load the images: with source installation, run `docker load` on the host; with mode A (real docker-in-docker), run it inside the AISBench container; with mode B (socket proxy), run it on the host. See [Harbor Terminal-Bench](harbor_bench.md) and [Agent Evaluation (Harbor)](../../base_tutorials/scenes_intro/agent_benchmark.md) for details.

## 4. Minimal Configuration (Run First, Tune Later)

It is recommended to start from `ais_bench/configs/agent_example/harbor_agent_task.py` and change only two places:

- `models[0]`: agent and model service parameters (`agent_name`, `model_names`, `api_base`, `api_key`)
- `datasets[0]`: the `path` in the task args, changed to the local path of the DeepSWE dataset

Example (terminus-2 agent + local vLLM setup):

```python
models = [
    dict(
        abbr="terminus-2",
        agent_name="terminus-2",            # -a/--agent: harbor AgentName or module.path:ClassName
        model_names=["hosted_vllm/qwen3"],  # --model: model name
        api_base="http://0.0.0.0:8080/v1",  # --api-base: model service base url (unified semantics)
        api_key="EMPTY",                    # --agent-api-key: service key (use EMPTY for local services)
        llm_kwargs={"max_tokens": 4096},    # LLM call parameters, merged into agent kwargs
        model_info={                        # model token limits and cost information
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        # deps_path="/path/to/terminus-2-offline-pack/",  # (optional, recommended) --agent-deps: offline agent dependency path
    )
]

datasets = [
    dict(
        abbr="harbor_deep-swe",
        args=dict(
            path="/path/to/deep-swe/",  # -p/--agent-dataset-path: local dataset path
            n_attempts=1,               # -k/--n-attempts: attempts per trial
            n_concurrent_trials=5,      # -n/--n-concurrent: number of concurrent trials
            environment_type="docker",  # -e/--environment: environment type
            n_tasks=None,               # --n-tasks: max number of tasks; None means all
        ),
    )
]
```

> 💡 The example configuration file above is a concrete application of the [custom configuration file approach](../../advanced_tutorials/run_custom_config.md). The configuration file is essentially a Python script that supports all Python syntax including loops, conditional statements, and list comprehensions. See [Running AISBench with Custom Configuration Files](../../advanced_tutorials/run_custom_config.md) for details.

### First-Run Recommendations

- Start with the `mini` dataset
- Use `n_attempts=1` and `n_concurrent_trials=5`
- DeepSWE tasks are long-horizon and each case takes a while; for a quick pipeline validation, limit the number of tasks via `n_tasks` (e.g. `n_tasks=5`)

## 5. Run Commands

Run the following in the AISBench root directory:

```bash
ais_bench ais_bench/configs/agent_example/harbor_agent_task.py --mode agent
```

You can also keep the config file unchanged and override key parameters on the command line (explicitly specified command-line parameters take precedence over the config file):

```bash
ais_bench ais_bench/configs/agent_example/harbor_agent_task.py --mode agent \
    -a terminus-2 \
    --model hosted_vllm/qwen3 \
    --api-base http://0.0.0.0:8080/v1 \
    --agent-api-key EMPTY \
    -p /path/to/deep-swe \
    -n 5 \
    -k 1
```

### Resume from Checkpoint

After an interruption (e.g. pressing `Ctrl+C`), run the same command again to resume automatically (optionally with `--reuse <timestamp>`, where `<timestamp>` is the output directory name of the previous run):

```bash
ais_bench ais_bench/configs/agent_example/harbor_agent_task.py --mode agent --reuse 20260530_012601
```

Harbor automatically detects whether `details/config.json` exists and skips completed trials.

### Multiple Attempts per Case (pass@k)

Change the `n_attempts` parameter to run the same case multiple times; the `pass@k` metric will be reported afterwards:

```python
datasets = [
    dict(
        abbr="harbor_deep-swe",
        args=dict(
            path="/path/to/deep-swe/",
            n_attempts=4,           # 4 attempts per trial
            n_concurrent_trials=5,
        ),
    )
]
```

## 6. How to Read Outputs

The default output directory is `outputs/default/<timestamp>/`. Focus on:

- Summary table (`summary/summary_*.csv|md|txt`): one row per (model × dataset) task, with columns such as `avg_score`, `correct`, `wrong`, and `exception`
- Task-level results: `results/{model}/{dataset}/{dataset}.json`
- Harbor result details: `results/{model}/{dataset}/details/`
  - `result.json`: task-level summary (`n_total_trials`, `trial_results`, `exception_stats`, etc.)
  - `trial_*/verifier/reward.json`: per-case score; DeepSWE verifier artifacts also contain auxiliary metrics such as `f2p_total` / `p2p_passed` (not counted into `avg_score`, for analysis only)
  - `trial_*/verifier/ctrf.json`: per-test-case pass/fail/skip with failure reasons, useful for locating exactly where a case failed
  - `trial_*/agent/trajectory.json`: agent execution trajectory

Key metrics:

- `avg_score`: average score (reward) over all tasks (`1.0` means passed)
- `reward_distribution`: reward distribution
- `exception_distribution`: exception type distribution (e.g. `AgentTimeoutError`, `ContextWindowExceededError`)
- `pass@k`: probability of at least one success in k attempts (requires `n_attempts>1`)

## 7. Common Issues and Troubleshooting

### 1) Cannot run on ARM

- Symptom: image architecture mismatch errors during the environment build stage
- Cause: DeepSWE task images are provided for x86_64 only
- Fix: run the evaluation on an x86_64 server

### 2) Image pull/build failure

- Symptom: errors during the environment build stage
- Cause: packaged images were not loaded in advance, and the image registry is unreachable
- Fix: run `docker load` in advance to load the packaged images, and make sure the loading location (host or inside the container) matches how the AISBench container was started

### 3) Agent dependency installation failure

- Symptom: errors or timeouts during the agent setup stage
- Cause: the DeepSWE image base OS is debian:12; the offline dependency package is wrong for the OS or the network is unavailable
- Fix: choose the agent offline dependency package for debian:12 (see "Agent dependency preparation" in [Agent Evaluation (Harbor)](../../base_tutorials/scenes_intro/agent_benchmark.md)) and pass it via `deps_path` / `--agent-deps`

### 4) `datasets` library version conflict

- Symptom: `datasets` dependency conflicts after installing Harbor
- Cause: installing harbor upgrades the `datasets` library to 4.0.0 or above
- Fix: this does not affect running the DeepSWE dataset; downgrade the `datasets` library if you need to run other datasets

### 5) Frequent agent timeouts

- Symptom: `AgentTimeoutError` dominates `exception_distribution`
- Cause: DeepSWE tasks are long-horizon, the default timeout is insufficient, or the model service throughput is low
- Fix: increase the timeout multipliers via `timeout_multiplier` / `agent_timeout_multiplier`, or reduce concurrency (`n_concurrent_trials`)

## 8. Advanced Tips (Optional)

- For initial debugging, use the `mini` dataset first, then switch to the full set (113 tasks) once the pipeline is stable
- For long-horizon tasks, check the number of steps and token consumption in `agent/trajectory.json` to estimate context and cost of the model service
- Set `n_attempts=k` when you need the pass@k metric; note the evaluation cost grows roughly k times
- Combine auxiliary metrics such as `f2p_total` / `p2p_passed` to analyze the verification details of failed cases, which is more informative than `avg_score` alone
