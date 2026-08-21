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

### 2. Install AISBench Evaluation Tool & Harbor Dependencies

#### 2.1 Install from Source
> ⚠️ Environment requirements: Ensure Docker version >= 20.10.0 and Docker Compose version >= 2.0.0 (docker compose may need to be installed separately). Also prepare a Python 3.12 runtime environment.
1. In the Python 3.12 environment, refer to [AISBench Installation Documentation](../../get_started/install.md) to install AISBench evaluation tool.
2. In the Python 3.12 environment, install Harbor:
   ```bash
   pip install harbor==0.6.1
   ```
3. Edit the Harbor docker compose configuration file.
   First, run:
   ```bash
   pip3 show harbor | grep Loca
   ```
   Find the site-packages path and locate `harbor/environments/docker/docker-compose-base.yaml` under that path, for example:
   ```
   /usr/local/lib/python3.12/dist-packages/harbor/environments/docker/docker-compose-base.yaml
   ```
   Configure the file as follows:
   ```yaml
   services:
     main:
       network_mode: host # Share host network, required
       environment: # Environment variables applied globally in the container
         http_proxy: XXXXXXXX # Configure proxy environment variables if the execution environment has no internet access
         https_proxy: XXXXXXXX # Configure proxy environment variables if the execution environment has no internet access
         no_proxy: XXXXXXXX
       volumes:
         - type: bind
           source: ${HOST_VERIFIER_LOGS_PATH}
           target: ${ENV_VERIFIER_LOGS_PATH}
         - type: bind
           source: ${HOST_AGENT_LOGS_PATH}
           target: ${ENV_AGENT_LOGS_PATH}
         - type: bind
           source: ${HOST_ARTIFACTS_PATH}
           target: ${ENV_ARTIFACTS_PATH}
       deploy:
         resources:
           limits:
             cpus: ${CPUS}
             memory: ${MEMORY}
   ```
> ⚠️ Note: Installing Harbor will upgrade the datasets library to version 4.0.0 or higher, which will cause dependency conflicts for the datasets library after installation. This does not affect tests for Terminal-Bench datasets using Harbor. However, if you need to test other datasets, you will need to downgrade the datasets library.

#### 2.2 Install Inside a Docker Container
1. Refer to the "Running Agent / Sandbox Benchmarks (Docker Inside the Container)" section in the [Image Overview](https://github.com/AISBench/benchmark/blob/master/docker/OVERVIEW.en.md) to start a container based on a **Python 3.12 or above image (only images published after 2026.7.1 are supported)**.
2. Inside the container, run the following command to install Harbor:
   ```bash
   pip install harbor==0.6.1 --break-system-packages
   ```
3. Edit Harbor's docker compose configuration file `/usr/local/lib/python3.12/dist-packages/harbor/environments/docker/docker-compose-base.yaml`:
```yaml
services:
  main:
    network_mode: host # Share host network, required
    security_opt: # Required when starting the container with Mode B (Socket Passthrough)
      - seccomp=unconfined
    environment: # Environment variables applied globally in the container
      http_proxy: XXXXXXXX # Configure proxy environment variables if the execution environment has no internet access
      https_proxy: XXXXXXXX # Configure proxy environment variables if the execution environment has no internet access
      no_proxy: XXXXXXXX
    volumes:
      - type: bind
        source: ${HOST_VERIFIER_LOGS_PATH}
        target: ${ENV_VERIFIER_LOGS_PATH}
      - type: bind
        source: ${HOST_AGENT_LOGS_PATH}
        target: ${ENV_AGENT_LOGS_PATH}
      - type: bind
        source: ${HOST_ARTIFACTS_PATH}
        target: ${ENV_ARTIFACTS_PATH}
    deploy:
      resources:
        limits:
          cpus: ${CPUS}
          memory: ${MEMORY}
```
> ⚠️ Note: Installing Harbor will upgrade the datasets library to version 4.0.0 or higher, which will cause dependency conflicts for the datasets library after installation. This does not affect tests for Terminal-Bench datasets using Harbor. However, if you need to test other datasets, you will need to downgrade the datasets library.


### 3. Prepare AISBench-modified Terminal-Bench Dataset and Images

#### 3.1 terminal-bench 2

AISBench modified dataset repository: [https://github.com/AISBench/terminal-bench-2](https://github.com/AISBench/terminal-bench-2)

> Note: AISBench did not change the case content. It only centralized all environment preparation (including all dependencies of the terminus-2 agent and verification resources) into the Dockerfile, avoiding repeated environment building and dependency installation on every run.

Terminal-Bench-2 pre-packaged images:
| Image Name | Download Link | CPU Architecture | Compressed Size |
| --------- | ------------ | ---------------- | -------------- |
| `terminal-bench-2-prepared-images_aarch64.tar` | [Link](https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-prepared-images_aarch64.tar) | aarch64 | 48.50 GB |
| `terminal-bench-2-prepared-images_x86_64.tar` | [Link](https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-prepared-images_x86_64.tar) | x86_64 | 71.43 GB |

> Tip: If you don't want to prepare images for all cases, you can get the terminal-bench-2-offline-mini sampled dataset from [terminal-bench-2-offline-mini](https://modelers.cn/datasets/AISBench/terminal-bench-2-offline-mini).

#### 3.2 terminal-bench 2.1

> terminal-bench 2.1 is essentially a bug-fix release of terminal-bench-2.0. The number of cases and case names are completely identical; only the dataset content and image content of certain cases differ.

AISBench modified dataset repository: [https://github.com/AISBench/terminal-bench-2.1](https://github.com/AISBench/terminal-bench-2.1)

> Note: AISBench did not modify the content of the official original images of terminal-bench-2.1; only the image tag names have been changed to distinguish them from terminal-bench-2.0. The image names in task.toml within the dataset are also updated accordingly.

| Image Name | Download Link | CPU Architecture | Compressed Size |
| --------- | ------------ | ---------------- | -------------- |
| `terminal-bench-2.1-images-aarch64.tar` | Not Supported | aarch64 | NA |
| `terminal-bench-2.1-images-x86_64.tar` | [Link](https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2.1-images-x86_64.tar) | x86_64 | 38.62 GB |

> ⚠️ Note:
> If you installed AISBench & Harbor dependencies from source, deploy the Terminal-Bench-2/2.1 images on the **host machine** by running `docker load -i xxxxxxx.tar`.
> If you started the AISBench container using Mode A (true Docker-in-Docker), deploy the Terminal-Bench-2/2.1 images **inside the container** by running `docker load -i xxxxxxx.tar`.
> If you started the AISBench container using Mode B (Socket Passthrough), deploy the Terminal-Bench-2/2.1 images on the **host machine** by running `docker load -i xxxxxxx.tar`.

### 4. Configure Custom Configuration File for Harbor Tasks

Modify `ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py` under AISBench tool root directory:

> 💡 The above `harbor_terminal_bench_2_task.py` is a concrete application of the [custom configuration file approach](../../advanced_tutorials/run_custom_config.md). The configuration file is essentially a Python script that supports all Python syntax including loops, conditional statements, list comprehensions, etc. You can refer to this example file to write a configuration file that meets specific needs. See [Running AISBench with Custom Configuration Files](../../advanced_tutorials/run_custom_config.md) for details.

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
            path="/path/to/terminal-bench-2/",  # -p/--path: Local dataset path, the path to either the terminal-bench-2 or terminal-bench 2.1 dataset
            # ......
            n_tasks=None,  # --n-tasks: Maximum number of tasks, None runs all, try setting a few for quick testing
            # ......
        ),
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