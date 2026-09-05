# Harbor Terminal-Bench

## Harbor Introduction

**Harbor** is a framework for evaluating AI agents, supporting various benchmark tasks including Terminal-Bench-2.

Official repository: [https://github.com/harbor-framework/harbor](https://github.com/harbor-framework/harbor)

### 1. Core Positioning and Background

- **Core Function**: Supports evaluation of various Agents (Terminus-2, Claude Code, OpenHands, etc.)
- **Core Innovation**:
  - Multiple environment support (Docker, Daytona, E2B, Modal)
  - Parallel execution and resume capability
  - Automatic evaluation and result analysis
- **Core Objective**: Evaluate agents' comprehensive capabilities in **task completion, tool usage, and policy compliance**

### 2. Supported Features

1. **Multi-Agent Support**
   - Built-in Agents: terminus-2, claude-code, openhands, aider, codex, etc.
   - Custom Agents: via `--agent-import-path`

2. **Multi-Environment Support**
   - Docker (local)
   - Daytona (cloud)
   - E2B (sandbox)
   - Modal (cloud)

3. **Dataset Support**
   - Local path: `-p /path/to/dataset`
   - Remote dataset: `-d dataset-name@version`

### 3. Core Evaluation Mechanism

- **Automatic verification**: Evaluate results via verifier
- **Parallel execution**: Control concurrency via `-n/--n-concurrent`
- **Resume capability**: Detect existing results, skip completed tasks
- **Trace export**: Export traces via `--export-traces`

## Quick Start with Harbor Terminal-Bench 2.0 in AISBench

### 1. Prepare Inference Services

Ensure deployment of tested inference services following OpenAI chat/completions API specification with tool call support.

### 2. Prepare AISBench-modified Terminal-Bench-2 Dataset and Images

AISBench modified dataset repository: [https://github.com/AISBench/terminal-bench-2](https://github.com/AISBench/terminal-bench-2)

> Note: AISBench only centralized all environment preparation into the Dockerfile without changing the case content, avoiding repeated environment building and dependency installation.

Terminal-Bench-2 pre-packaged images:
| Image Name | Download Link | CPU Architecture | Compressed Size |
| --------- | ------------ | ---------------- | -------------- |
| `terminal-bench-2-prepared-images_aarch64.tar` | [Link](https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-prepared-images_aarch64.tar) | aarch64 | 48.50 GB |
| `terminal-bench-2-prepared-images_x86_64.tar` | [Link](https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-prepared-images_x86_64.tar) | x86_64 | 71.43 GB |

> Tip: If you don't want to prepare images for all cases, you can get the terminal-bench-2-offline-mini sampled dataset from [terminal-bench-2-offline-mini](https://modelers.cn/datasets/AISBench/terminal-bench-2-offline-mini).

### 3. Install AISBench Evaluation Tool & Harbor Dependencies

#### 3.1 Install from Source
> ⚠️ Environment requirements: Ensure Docker version >= 20.10.0 and Docker Compose version >= 2.0.0 (docker compose may need to be installed separately). Also prepare a Python 3.12 runtime environment.
1. In the Python 3.12 environment, refer to [AISBench Installation Documentation](../../get_started/install.md) to install AISBench evaluation tool.
2. In the Python 3.12 environment, install Harbor:
   ```bash
   pip install harbor==0.20.0
   ```
> ⚠️ Note: Installing Harbor will upgrade the datasets library to version 4.0.0 or higher, which will cause dependency conflicts for the datasets library after installation. This does not affect tests for Terminal-Bench datasets using Harbor. However, if you need to test other datasets, you will need to downgrade the datasets library.

> ⚠️ Note: When installing from source, the case image tar downloaded in the [2. Prepare Dataset and Images](#2-prepare-aisbench-modified-terminal-bench-2-dataset-and-images) section must be loaded into the local docker daemon on the **host machine** by running `docker load -i xxxxxxx.tar` before running the evaluation.

#### 3.2 One-Click Preparation (Recommended)
If you don't want to prepare the environment manually, it is recommended to use the **AISBench Agent Runtime one-click preparation solution**. The same script covers both **Quick Start (online)** and **Offline (intranet/isolated environment)** scenarios, and can be freely combined via `--runtime-tar` / `--case-tar` / `--datasets` without switching between different flows.

```bash
# 1. Start the runtime container on the host with one click (automatically select DinD/Socket mode, auto-mount datasets, and auto copy case image tar into the container for docker load)
#    Online scenario: omit --runtime-tar, runtime image will be automatically pulled from ghcr.io
#    Offline scenario: skip external network pull via --runtime-tar, which can be obtained from the latest release information in advance
curl -fsSL https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/ais_bench_agent_bootstrap.sh \
    | bash -s -- \
        --datasets /path/to/terminal-bench-2-offline-mini/terminal-bench-2-offline-selected_0.10/ \
        --runtime-tar /path/to/agent_runtime_image_v3.1-20260701-master-ubuntu24.04-py312-<arch>.tar.gz \
        --case-tar /path/to/terminal-bench-2-offline-prepared-images-selected-0.10.tar \
        --host-path /path/to/test_wkp/ \
        --container-name test_agent_run
# --datasets must point to a directory structure consistent with the terminal-bench-2-offline-selected_0.10/ subdirectory of the terminal-bench-2-offline-mini repo
# --runtime-tar (optional) pre-downloaded runtime image; if omitted, the latest is pulled automatically
# --case-tar must point to a tar structure consistent with the case image tar described in the corresponding agent evaluation document (can be passed multiple times, or as a directory)
# --host-path must be an empty directory; a same-named directory will be created inside the container to mount datasets and case images
# --container-name must be unique; otherwise the old container will be overwritten

# 2. Enter the container (case images are already loaded inside and ready to use)
docker exec -it test_agent_run bash

# 3. (No need to change path) The native config path is read automatically from AISBENCH_AGENT_DATASET_PATH
#    Only vim model_names / api_base
vim ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py

# 4. Verify runtime is ready
ais_bench_agent_doctor.sh harbor

# 5. Enter the evaluation environment
agent_env harbor
```

To switch to another dataset (mini-0.14 / mini-0.20 / full): destroy the old container → restart bootstrap with `bash ... --datasets <new_path> --case-tar <new_tar>`.

`--runtime-tar` / `--case-tar` / `--datasets` are fully independent and can be combined freely. None of them trigger any `docker pull` or `curl` to external networks; in the Quick Start (online) scenario, omit `--runtime-tar` and the script will pull the runtime image from the network automatically.

`--case-tar` works in both A/B modes: the script `docker cp`s the tar into the runtime container, then runs `docker load` inside the container to load it into that container's docker daemon.

This solution addresses the following pain points:
- **Dependency conflicts**: harbor==0.20.0 forces datasets to be upgraded to 4.0+, which would pollute the main environment; the runtime image uses an isolated venv
- **Error-prone container configuration**: DinD mode A/B, `--cgroupns=host`, `daemon.json`, and seccomp are handled automatically
- **Frequent dataset version changes**: datasets and case images are not baked into the runtime image; users prepare them on the host and mount via `--datasets` / load via `--case-tar`, avoiding frequent image expiration
- **Case image management**: `--case-tar` loads case images into the container in one shot during bootstrap; no manual `docker pull` / `docker load` inside the container
- **No environment validation**: `doctor.sh` validates runtime readiness before running evaluations and gives precise fix guidance on failure
- **Offline deployment**: `--runtime-tar <PATH>` skips network fetching of the runtime image; `--case-tar <PATH>` loads case images into the container (can be used multiple times, or with a directory). Intranet-isolated environments can run with zero external network requests throughout

For the solution principles and script implementation, see [`docker/agent_runtime/`](https://github.com/AISBench/benchmark/tree/master/docker/agent_runtime/README.md).

### 4. Configure Custom Configuration File for Harbor Tasks

Modify `ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py` under AISBench tool root directory:

```python
models = [
    dict(
        abbr="terminus-2",
        agent_name="terminus-2",  # -a/--agent: Agent name (terminus-2, claude-code, openhands, etc.)
        model_names=["hosted_vllm/qwen3"],  # -m/--model: Model name, hosted_vllm/{model_name}
        agent_kwargs={  # --ak/--agent-kwarg: Agent extra parameters
            "api_base": "http://0.0.0.0:8080/v1",  # terminus-2 requires api_base to connect to inference service, e.g. "http://0.0.0.0:8080/v1" will access "http://0.0.0.0:8080/v1/chat/completions"
            "model_info": {  # Model token limits and cost information
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
            "llm_call_kwargs": { # LLM call parameters
                "max_tokens": 4096, # Maximum output token number
                # "temperature": 0.7,
                # "top_p": 0.9,
                # "top_k": 50,
            },
        },
        agent_env=None,  # --ae/--agent-env: Environment variables passed to agent
    )
]
# ......
datasets = []
datasets.append(
    dict(
        abbr=f'harbor_terminal-bench-2',
        args=dict(
            n_attempts=1,  # -k/--n-attempts: Number of attempts per trial
            timeout_multiplier=1.0,  # --timeout-multiplier: Timeout multiplier
            # ......
            n_concurrent_trials=5,  # -n/--n-concurrent: Number of concurrent trials
            # ......
            path="/path/to/terminal-bench-2/",  # -p/--path: Local dataset path
            # ......
            n_tasks=None,  # --n-tasks: Maximum number of tasks, None runs all, try setting a few for quick testing
            # ......
        ),
    )
)
# ......
```

### 5. Execute Harbor Tasks

1. Execute the following command in AISBench tool root directory:
   ```bash
   ais_bench ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py --debug
   ```

> Note: Adding `--debug` is recommended because Harbor's native dashboard during execution is clearer and more detailed, allowing real-time score updates. However, in non-debug mode, the dashboard content cannot be logged to disk and can only be seen in the terminal, so it's recommended to run in debug mode.

2. Execution process dashboard example

```
Base path of result&log : outputs/default/20260530_012601
Task Progress Table (Updated at: 2026-05-30 01:30:00)
Press Up/Down arrow to page, 'P' to PAUSE/RESUME screen refresh, 'Ctrl + C' to exit

+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+
| Task Name                         |   Process | Progress                                                   | Time Cost   | Status   | Log Path                                        | Extend Parameters   |
+===================================+===========+============================================================+=============+==========+=================================================+=====================+
| terminus-2/harbor_terminal-bench-2 |   1234567 | [######                        ] 10/21 Running Harbor | 0:07:13     | running  | logs/eval/terminus-2/harbor_terminal-bench-2.out | None                |
+-----------------------------------+-----------+------------------------------------------------------------+-------------+----------+-------------------------------------------------+---------------------+
```

3. After task execution is complete, the following accuracy results will be printed:

```
============================================================
Dataset: harbor_terminal-bench-2
Model: terminus-2
============================================================
Total Count: 74
Errors: 54
Avg Score: 0.045

Reward Distribution:
+--------+-------+
|  Score | Count |
+========+=======+
|    0.0 |    70 |
+--------+-------+
|    1.0 |     4 |
+--------+-------+

Exception Distribution:
+----------------------------+-------+
| Exception                  | Count |
+============================+=======+
| AgentTimeoutError          |    39 |
+----------------------------+-------+
| AgentSetupTimeoutError     |    13 |
+----------------------------+-------+
| InternalServerError        |     2 |
+----------------------------+-------+

Pass@k:
+----+-----------+
| k  | Pass Rate |
+====+===========+
|  1 |    0.0541 |
+----+-----------+
|  2 |    0.0811 |
+----+-----------+

+--------------------+-----------+----------------+--------+---------------+--------------+
| dataset                 | version   | metric         | mode   |   total_count |   terminus-2 |
+========================+===========+================+========+===============+==============+
| harbor_terminal-bench-2 | a39421    | avg_score      | gen    |            74 |        0.045 |
+--------------------+-----------+----------------+--------+---------------+--------------+
| harbor_terminal-bench-2 | a39421    | n_errors       | gen    |            74 |           54 |
+--------------------+-----------+----------------+--------+---------------+--------------+
| harbor_terminal-bench-2 | a39421    | n_total_trials | gen    |            74 |           74 |
+--------------------+-----------+----------------+--------+---------------+--------------+
```

- `Avg Score`: Average score across all tasks
- `n_errors`: Number of exceptions during execution
- `reward_distribution`: Reward distribution
- `exception_distribution`: Exception type distribution
- `pass@k`: Success rate for k executions

4. The structure of result files in the final `outputs/default/{timestamp}` directory is as follows:

```shell
outputs/default/20260530_012601
├── configs
│   └── 20260530_012601.py
├── logs
│   └── eval
│       └── terminus-2
│           └── harbor_terminal-bench-2.out
├── results
│   └── terminus-2
│       └── harbor_terminal-bench-2
│           ├── details
│           │   ├── config.json
│           │   ├── result.json
│           │   └── trial_*/
│           └── harbor_terminal-bench-2.json
└── summary
    ├── summary_20260530_012601.csv
    ├── summary_20260530_012601.md
    └── summary_20260530_012601.txt
```

## Continue Evaluation After Interruption

After interrupting task execution (e.g., pressing `Ctrl+C`), execute the same command again with `--reuse`:

```bash
ais_bench ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py --debug --reuse 20260530_012601
```

Where `20260530_012601` is the timestamp of the previous failed task execution. Replace with your actual timestamp.

Harbor will automatically detect if `details/config.json` exists and skip completed trials.

## Multiple Executions of a Single Case (pass@k)

Modify the `n_attempts` parameter to execute the same case multiple times:

```python
datasets.append(
    dict(
        abbr='harbor_terminal-bench-2',
        args=dict(
            path="/path/to/terminal-bench-2/",
            n_attempts=5,  # Execute each trial 5 times
            n_concurrent_trials=5,
        ),
    )
)
```

After execution, `pass@k` metrics will be displayed, indicating the probability of at least one success in k executions.

## Task Configuration (in datasets) - Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | - | Local dataset path (-p/--path) |
| `n_attempts` | int | 1 | Number of attempts per trial (-k/--n-attempts) |
| `n_concurrent_trials` | int | 5 | Number of concurrent trials (-n/--n-concurrent) |
| `environment_type` | str | docker | Environment type (-e/--env) |
| `environment_force_build` | bool | False | Whether to force rebuild environment |
| `environment_delete` | bool | True | Whether to delete environment after completion |
| `timeout_multiplier` | float | 1.0 | Timeout multiplier |
| `max_retries` | int | 0 | Maximum number of retries |
| `task_names` | list[str] | None | Task names to include (--include-task-name) |
| `exclude_task_names` | list[str] | None | Task names to exclude (--exclude-task-name) |
| `n_tasks` | int | None | Maximum number of tasks (--n-tasks) |

## Agent Configuration (in models) - Related Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `abbr` | str | - | Model abbreviation |
| `agent_name` | str | oracle | Agent name (-a/--agent) |
| `model_names` | list[str] | None | Model name (-m/--model) |
| `agent_kwargs` | dict | {} | Agent extra parameters (--ak/--agent-kwarg) |
| `agent_env` | dict | {} | Agent environment variables (--ae/--agent-env) |