# Terminal-Bench 2.1 User Guide

Terminal-Bench 2.1 is a bugfix release of Terminal-Bench 2.0 that evaluates how well AI agents complete multi-step tasks in a real terminal environment. Given an isolated container environment and a task description, the agent must autonomously use the terminal to perform a series of operations such as installation, configuration, coding, and debugging; a verifier then automatically decides whether the task is completed.

> 👉 Terminal-Bench 2.1 has exactly the same number and names of cases as 2.0; only the dataset content and image content of some cases differ. It can be regarded as a case-and-image bugfix release of 2.0.

## 1. Feature Overview

`ais_bench` currently supports the following Terminal-Bench 2.1 capabilities:

- Datasets: `full` (official full set) and `mini` (small-scale sample)
- Tasks: the general Harbor-based agent evaluation chain (`--mode agent`); the agent executes terminal tasks autonomously inside a container environment and the verifier scores the result automatically
- Result summary: output key metrics such as `avg_score`, `reward_distribution`, `exception_distribution`, and `pass@k`

> 💡 Terminal-Bench 2.1 reuses AISBench's general Harbor-based agent evaluation chain. For environment preparation (agent runtime container, offline agent dependency packages), parameter references, and other general topics, see [Agent Evaluation (Harbor)](../../base_tutorials/scenes_intro/agent_benchmark.md).

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

4) ⚠️ The agent needs to access the internet during execution; configure proxy environment variables if the environment cannot reach the internet.

## 3. Dataset and Image Preparation

### 3.1 Datasets

| Dataset | Link | Notes |
| ------- | ---- | ----- |
| Terminal-Bench 2.1 (full) | https://github.com/AISBench/terminal-bench-2-1 | AISBench-modified version: the content of the official original images is unchanged; only the image tag names are modified to distinguish them from 2.0, and the image names in the dataset `task.toml` are updated accordingly |
| Terminal-Bench 2.1 mini | https://modelers.cn/datasets/AISBench/terminal-bench-2-1-mini | Small-scale sampled dataset based on 2.1, suitable for a quick first run |

### 3.2 Images

Every case has a corresponding Docker image whose name is defined in the dataset. On an x86_64 server with good network access, images are automatically pulled and built during execution, but this is usually slow; it is recommended to load the packaged images in advance:

| Packaged image resource | Link | CPU arch | Base OS |
| ----------------------- | ---- | -------- | ------- |
| `terminal-bench-2.1-images-x86_64.tar` | https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2.1-images-x86_64.tar | x86_64 | ubuntu:24.04, debian:12, debian:13 |
| `terminal-bench-2.1-images-aarch64.tar` | https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2.1-images-aarch64.tar | aarch64 | ubuntu:24.04, debian:12, debian:13 |

```bash
docker load -i terminal-bench-2.1-images-x86_64.tar
```

> ⚠️ Where to load the images: with source installation, run `docker load` on the host; with mode A (real docker-in-docker), run it inside the AISBench container; with mode B (socket proxy), run it on the host. See [Harbor Terminal-Bench](harbor_bench.md) for details.

## 4. Minimal Configuration (Run First, Tune Later)

It is recommended to start from `ais_bench/configs/agent_example/harbor_agent_task.py` and change only two places:

- `models[0]`: agent and model service parameters (`agent_name`, `model_names`, `api_base`, `api_key`)
- `datasets[0]`: the `path` in the task args, changed to the local path of the Terminal-Bench 2.1 dataset

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
        abbr="harbor_terminal-bench-2.1",
        args=dict(
            path="/path/to/terminal-bench-2.1/",  # -p/--agent-dataset-path: local dataset path
            n_attempts=1,                         # -k/--n-attempts: attempts per trial
            n_concurrent_trials=5,                # -n/--n-concurrent: number of concurrent trials
            environment_type="docker",            # -e/--environment: environment type
            n_tasks=None,                         # --n-tasks: max number of tasks; None means all
        ),
    )
]
```

> 💡 The example configuration file above is a concrete application of the [custom configuration file approach](../../advanced_tutorials/run_custom_config.md). The configuration file is essentially a Python script that supports all Python syntax including loops, conditional statements, and list comprehensions. See [Running AISBench with Custom Configuration Files](../../advanced_tutorials/run_custom_config.md) for details.

### First-Run Recommendations

- Start with the `mini` dataset
- Use `n_attempts=1` and `n_concurrent_trials=5`
- Limit the number of tasks via `n_tasks` (e.g. `n_tasks=10`) for a quick pipeline validation

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
    -p /path/to/terminal-bench-2.1 \
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
        abbr="harbor_terminal-bench-2.1",
        args=dict(
            path="/path/to/terminal-bench-2.1/",
            n_attempts=5,           # 5 attempts per trial
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
  - `trial_*/verifier/reward.json`: per-case score
  - `trial_*/verifier/ctrf.json`: per-test-case pass/fail/skip with failure reasons, useful for locating exactly where a case failed
  - `trial_*/agent/trajectory.json`: agent execution trajectory

Key metrics:

- `avg_score`: average score (reward) over all tasks (`1.0` means passed)
- `reward_distribution`: reward distribution
- `exception_distribution`: exception type distribution (e.g. `AgentTimeoutError`, `AgentSetupTimeoutError`)
- `pass@k`: probability of at least one success in k attempts (requires `n_attempts>1`)

## 7. Common Issues and Troubleshooting

### 1) The agent cannot access the internet

- Symptom: cases fail or agent dependency installation fails
- Cause: the agent needs to access the internet during Terminal-Bench 2.1 execution
- Fix: configure proxy environment variables (e.g. `--ae HTTPS_PROXY=http://proxy:port`), or refer to the offline solution in [Agent Evaluation (Harbor)](../../base_tutorials/scenes_intro/agent_benchmark.md)

### 2) Image pull/build failure

- Symptom: errors during the environment build stage
- Cause: packaged images were not loaded in advance, and the image registry is unreachable
- Fix: run `docker load` in advance to load the packaged images, and make sure the loading location (host or inside the container) matches how the AISBench container was started

### 3) `datasets` library version conflict

- Symptom: `datasets` dependency conflicts after installing Harbor
- Cause: installing harbor upgrades the `datasets` library to 4.0.0 or above
- Fix: this does not affect running terminal-bench datasets; downgrade the `datasets` library if you need to run other datasets

### 4) Frequent agent timeouts

- Symptom: `AgentTimeoutError` / `AgentSetupTimeoutError` dominate `exception_distribution`
- Cause: the task timeout is too short, or the model service throughput is insufficient
- Fix: increase the timeout multiplier via `timeout_multiplier`, or reduce concurrency (`n_concurrent_trials`)

## 8. Advanced Tips (Optional)

- For initial debugging, use the `mini` dataset first, then switch to the full set once the pipeline is stable
- Focus on `exception_distribution` and `ctrf.json`; exception cases usually pinpoint environment problems better than low-score cases
- Set `n_attempts=k` when you need the pass@k metric; note the evaluation cost grows roughly k times
- Do not compare 2.0 and 2.1 scores directly: some cases differ in dataset and image content, so use the same version for horizontal comparison
