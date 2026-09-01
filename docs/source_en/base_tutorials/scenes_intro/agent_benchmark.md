# Agent Evaluation (Harbor)

AISBench natively integrates [Harbor](https://github.com/harbor-framework/harbor) as the agent evaluation engine. With a single `--mode agent` command you can launch a Harbor evaluation, watch the status of each case in real time, and get a single-table + CSV summary. This path is fully independent of AISBench's native inference/accuracy chain and **does not change any other AISBench functionality**.

Unlike the existing [Harbor Terminal-Bench](../../extended_benchmark/agent/harbor_bench.md) integration, this is a general-purpose agent evaluation with the following capabilities:

- **All Harbor-defined Agents**: built-in `AgentName` agents (terminus-2, claude-code, openhands, aider, codex, etc.) plus custom `module.path:ClassName` agents.
- **Unified parameter adaptation**: different agents pass the same semantic parameters (model service base url / API key / LLM call parameters / model info) through different channels (kwargs or environment variables). `AgentParamAdapter` translates them automatically, so you only use one set of unified semantic parameters.
- **All Harbor dataset sources**: local path (including a single task directory), registry `name@version`, and package `org/name@ref`.
- **Real-time monitoring HTTP service**: stdlib-only, no extra dependency; external clients can fetch live execution info of all Harbor tasks.
- **Standalone dependency set**: agent evaluation only requires installing `requirements/agent.txt` and does not depend on AISBench's heavy native dependencies.

## Installation

### Prerequisites

- A model inference service that follows the **OpenAI chat/completions API** and supports **tool call** (local or cloud).
- A Python 3.12 environment; Docker / execution environment prepared as required by Harbor.

### Install the Agent dependency set

```bash
pip install -r requirements/agent.txt
```

> ⚠️ **Non-critical warnings during installation**: while installing the agent dependencies (especially installing Harbor and its transitive dependencies from source in editable mode), `pip` may print some warnings/errors that do **not affect agent evaluation**, including:
> - dependency version-conflict warnings (e.g. Harbor upgrades the `datasets` library to 4.0.0+, which may produce version-conflict warnings with other dependencies);
> - warnings/issues for third-party builds, or `yanked` / `deprecated` hints during resolution.
> As long as these do **not cause the install to fail** (pip reports an `error` and stops), you can ignore them and continue using Harbor agent evaluation normally. To confirm a successful install, run `pip show harbor`.

## Dataset Support

AISBench supports **all datasets resolvable by Harbor** through a unified entry, with three sources, all selectable via `-d/--dataset` or the dataset `args`:

| Source | CLI / config | Description |
| --- | --- | --- |
| Local dataset directory | `-p/--agent-dataset-path` (`path`) | Points to a Harbor dataset directory containing multiple task (`task.toml`, etc.); can be narrowed with `--include/exclude-task-name` and `--n-tasks` |
| Single task directory | `-p/--agent-dataset-path` (`path`) | Points directly to one task directory (Harbor checks `is_valid_dir`); runs only that task |
| Registry dataset | `-d name@version` (`dataset_name_version`) | Pulls a dataset from the Harbor dataset registry by name and version; a `name` without `@version` uses the registry dataset |
| Package dataset | `-d org/name@ref` (`dataset_name_version`) | References a dataset package on the Harbor hub / a Git repo via `org/name` + `@ref` |

The set of supported datasets depends on the adapters on the Harbor side; AISBench does not restrict a specific list. Below are some benchmark examples that ship with a built-in adapter in Harbor's `adapters/` directory (see that directory for the full list):

| Category | Example datasets |
| --- | --- |
| Terminal / CLI | terminal-bench (2 / 2.1), swe-bench, swe-bench-pro, swebench-multilingual, swt-bench, swe-smith, swe-lancer, swe-gym |
| Code / software engineering | bix-bench, code-pde, ds-1000, quix-bugs, sci-code, devops-gym, research-code-bench, feat-bench |
| Others | strong-reject, simple-qa, text-arena, usaco, replication-bench, reasoning-gym, webgen-bench, etc. |

> 💡 For terminal-bench 2 / 2.1, AISBench provides offline datasets and pre-built images; see 📚 [Harbor Terminal-Bench](../../extended_benchmark/agent/harbor_bench.md) for resources and preparation.

## Agent Support

AISBench supports all agents defined by Harbor (pass the name directly via `-a/--agent`), as well as custom `module.path:ClassName` agents via `--agent-import-path`. The full list of built-in `AgentName` values:

| Agent (`AgentName`) | Agent (`AgentName`) |
| --- | --- |
| `oracle`, `nop`, `acp` | `claude-code`, `cline-cli`, `cortex-code` |
| `terminus`, `terminus-1`, `terminus-2` | `aider`, `codex`, `cursor-cli` |
| `gemini-cli`, `antigravity-cli`, `antigravity-sdk` | `rovodev-cli`, `goose`, `grok-build` |
| `hermes`, `mini-swe-agent`, `nemo-agent` | `swe-agent`, `opencode`, `openclaw` |
| `openhands`, `openhands-sdk`, `kimi-code` | `kimi-cli`, `langgraph`, `deerflow` |
| `mimo`, `pi`, `qwen-coder` | `copilot-cli`, `devin`, `trae-agent` |
| `computer-1`, `eve`, `fx` | `dsh`, `dspy-rlm`, `vibe` |

> 💡 Different agents pass the same semantic parameters (model service base url / API key, etc.) through different channels. AISBench's `AgentParamAdapter` adapts them automatically: agents that use environment variables (e.g. `claude-code`→`ANTHROPIC_*`, `dsh`→`DSH_*`, `openhands`→`LLM_*`/`OPENAI_*`) and agents that use constructor kwargs (e.g. `terminus-2`→`api_base`) all work with the same unified semantic parameters (`--api-base` / `--agent-api-key`), with no need to distinguish.

## Resource Preparation

### Harbor-format dataset resources

## Quick Start (either of the two ways)

| ⭐ Recommended: Use command-line parameters | Alternative: Use a custom config file |
| :--- | :--- |
| Configure the agent, model service, dataset and runtime parameters all in one command, no new file needed | Centralize all parameters and reuse them across runs |
| Unified semantic parameters are adapted automatically, ready to use | Supports all Python syntax for flexible extension |

::::{tab-set}
:::{tab-item} ⭐ Recommended: Use command-line parameters

The command-line way configures every parameter in one command. Besides the regular model-service parameters, the parameters newly introduced for agent evaluation (`--mode agent`, `-a/--agent`, `--api-base`, `--agent-api-key`, `-p/--agent-dataset-path`, `-d/--dataset`, `-n/--n-concurrent`, `-k/--n-attempts`, `-e/--environment`, `--monitor-port`, etc.) work together with `--mode agent` / `--mode agent_viz`. See 📚 [User Configuration Parameters - Agent Evaluation Parameters](../all_params/cli_args.md) for the full list.

Take a local terminal-bench-2 dataset with the terminus-2 agent as an example:

```bash
ais_bench --mode agent \
    -a terminus-2 \                        # Agent name (or a custom import path)
    --model hosted_vllm/qwen3 \            # Model name (repeatable)
    --api-base http://0.0.0.0:8080/v1 \    # Model service base url (unified semantic)
    --agent-api-key sk-xxx \               # Model service API key (unified semantic)
    -p /path/to/terminal-bench-2 \         # Local dataset path
    -n 5 \                                 # Concurrent trial count
    -k 1 \                                 # Attempts per trial
    -e docker \                            # Environment type
    --monitor-port 8788                    # Real-time monitoring HTTP service port
```

To append raw agent kwargs or environment variables (highest priority), use `--ak key=value` / `--ae KEY=VALUE`:

```bash
ais_bench --mode agent -a terminus-2 --model hosted_vllm/qwen3 \
    --api-base http://0.0.0.0:8080/v1 \
    -p /path/to/terminal-bench-2 \
    --ak max_tokens=4096 \
    --ae HTTPS_PROXY=http://proxy:port
```

If the local dataset is a **single task directory** (rather than a dataset directory), use `-p` the same way. For remote datasets use `-d name@version` / `-d org/name@ref`.

:::
:::{tab-item} Alternative: Use a custom config file

A custom config file puts "model service and agent parameters" under `models` and "agent evaluation task parameters" under `datasets`, so it can be written once and reused. See `ais_bench/configs/agent_example/harbor_agent_task.py` / `harbor_terminal_bench_2_task.py` for reference. A complete example:

```python
from mmengine.config import read_base
from ais_bench.benchmark.tasks.custom_tasks.harbor_agent_task import HarborAgentTask
from ais_bench.benchmark.runners.harbor_runner import HarborRunner
from ais_bench.benchmark.tasks.base import EmptyTask
from ais_bench.benchmark.summarizers.harbor import HarborSummarizer

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer

# models: model service and agent parameters; unified semantic parameters are
# translated automatically by AgentParamAdapter
models = [
    dict(
        abbr="terminus-2",
        agent_name="terminus-2",            # -a/--agent: a harbor AgentName or module.path:ClassName
        model_names=["hosted_vllm/qwen3"],  # --model: model name (multi-valued)
        api_base="http://0.0.0.0:8080/v1",  # --api-base: model service base url (unified semantic)
        agent_api_key="sk-xxx",             # --agent-api-key: model service API key (unified semantic)
        llm_kwargs={"max_tokens": 4096},    # LLM call parameters, merged into agent kwargs
        model_info={                        # model token limits and cost info
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
        },
        # deps_path=None,                   # --agent-deps: offline agent deps bundle path
    )
]

# datasets: agent evaluation task parameters
datasets = [
    dict(
        abbr="harbor_terminal-bench-2",
        args=dict(
            path="/path/to/terminal-bench-2/",  # -p/--agent-dataset-path: local dataset path
            # dataset_name_version=None,        # -d/--dataset: remote dataset name@version / org/name@ref
            n_concurrent_trials=5,              # -n/--n-concurrent: concurrent trial count
            n_attempts=1,                       # -k/--n-attempts: attempts per trial
            environment_type="docker",          # -e/--environment: environment type
            max_retries=0,                      # --max-retries: max retry count
            yes=True,                           # -y/--yes: auto-confirm environment variable prompts
        ),
    )
]

# agent mode has no native inference stage
infer = dict(runner=dict(task=dict(type=EmptyTask)))
eval = dict(
    runner=dict(
        type=HarborRunner,
        monitor_port=0,       # --monitor-port: monitoring HTTP service port (0 = off)
        task=dict(type=HarborAgentTask),
    ),
)
summarizer = dict(attr="accuracy", type=HarborSummarizer)
```

After editing the config file, run:

```bash
ais_bench ais_bench/configs/agent_example/harbor_agent_task.py --mode agent
```

> 💡 Parameters explicitly passed on the command line override the corresponding fields in the custom config file. Meanings and acceptable values of each parameter are listed in 📚 [Appendix: Custom Config File Parameters](#appendix-custom-config-file-parameters) at the end.

:::
::::

### Real-time monitoring service

Turn on the read-only HTTP service (stdlib-only, default port `0` = off) via `--monitor-port <port>`:

```bash
ais_bench ... --mode agent --monitor-port 8788
```

Common endpoints (`{model}`, `{dataset}`, `{case}` refer to the actual tasks):

| Endpoint | Description |
| --- | --- |
| `GET /api/health` | Liveness probe |
| `GET /api/tasks` | All task-level snapshots |
| `GET /api/tasks/{model}/{dataset}/` | Task overview = raw `result.json` |
| `GET /api/tasks/{model}/{dataset}/{case}` | Raw `result.json` of one case (`trial_00000` / index / harbor task name) |
| `GET /api/tasks/{model}/{dataset}/cases` | Per-case derived status / pass-fail detail |
| `GET /api/jobs` | Progress of each job (aggregated counts + case status) |

### Resume & automatic retry of exception cases

Harbor auto-detects whether `details/config.json` exists and skips completed trials; re-running the same command (optionally with `--reuse <timestamp>`) resumes:

```bash
ais_bench ais_bench/configs/agent_example/harbor_agent_task.py --mode agent --reuse 20260530_012601
```

The `--purge-exception-cases` switch **takes effect only when `--reuse` is set**: before execution it extracts exception case names from each task's `result.json` `exception_stats` and deletes the corresponding case directories, so Harbor re-creating the job automatically re-runs those exception-finished cases:

```bash
# first run produces some exception cases, then re-run
ais_bench ... --mode agent --reuse 20260530_012601 --purge-exception-cases
```

With existing results, use `--mode agent_viz` to only summarize without starting any Harbor job:

```bash
ais_bench ... --mode agent_viz --reuse 20260530_012601
```

## Execution Results and On-Disk Files

### Meaning of the printed results

After the run, `HarborSummarizer` prints one table and writes one CSV, with one row per (model × dataset) task. The columns map to the following metrics:

| Column | Meaning |
| --- | --- |
| `agent` | The agent name used |
| `model_name` | Model name (`model_names`, `,`-separated when multiple) |
| `dataset` | Dataset task abbreviation (`dataset.abbr`) |
| `avg_score` | Average score of the task (read from the aggregated `result.json` `avg_score`) |
| `correct` | Number of completed trials with score `>= 1.0` (from `reward_distribution`) |
| `wrong` | Number of completed trials with score `0 ~ 1.0` (excluding 1.0) (from `reward_distribution`) |
| `exception` | Number of trials that ended with an exception (corresponds to `n_errors`) |

> 💡 More raw metrics (`total_count` / `n_errors` / `reward_distribution` / `exception_distribution` / `pass@k`, etc.) stay in the per-task `results/{model}/{dataset}.json` and `details/result.json`; inspect them via the `--monitor-port` endpoints or by reading the on-disk files directly.

### On-disk file structure and meaning

Results are stored under `outputs/default/{timestamp}/` (the working directory can be changed with `--work-dir`). The outer layout matches other AISBench scenarios; the core is the **Harbor on-disk result** under `results/{model}/{dataset}/details/`, which mirrors the Harbor job result layout and records the execution and verification detail of each case:

```bash
outputs/default/20260530_012601/
├── configs
│   └── 20260530_012601.py            # synthesized complete config (all CLI/config parameters)
├── logs/eval/{model}/{dataset}.out    # execution log
├── results/{model}/{dataset}/         # evaluation results (where Harbor lands its output)
│   ├── details/                      # ▽ the result directory of this task's Harbor job (core) ▽
│   │   ├── config.json               # task job config (resume basis: resume when present)
│   │   ├── result.json               # task-level summary: n_total_trials/stats/trial_results/exception_stats
│   │   └── trial_00000/              # one directory per case (scheduled order)
│   │       ├── result.json           # this case result (status/reward/exception_info/timings/agent_info)
│   │       ├── exception.txt         # exception message text of this case (when it errored out)
│   │       ├── trial.log             # trial run log of this case
│   │       ├── config.json           # task config of this case
│   │       ├── agent/                # agent run artifacts
│   │       │   └── trajectory.json   # agent trajectory (its presence indicates a trajectory)
│   │       └── verifier/             # verifier artifacts
│   │           ├── reward.json       # score of this case
│   │           ├── ctrf.json         # per test-case pass/fail/skip + failure reason
│   │           ├── test-stdout.txt   # tail of the verifier stdout
│   │           └── test-stderr.txt   # tail of the verifier stderr
│   └── {dataset}.json                # aggregated result of this task (for the summarizer)
└── summary/
    ├── summary_20260530_012601.csv   # summary table (csv format)
    ├── summary_20260530_012601.md    # summary table (markdown format)
    └── summary_20260530_012601.txt   # summary table (text format)
```

Meaning of each Harbor on-disk file:

| File | Meaning |
| --- | --- |
| `details/config.json` | Raw job config; **its presence determines whether to resume** (if present, `_resume_job` skips completed trials) |
| `details/result.json` | Task-level summary; `stats.evals[*].exception_stats` records exception case names and is the **data source** for `--purge-exception-cases` auto-retry; it can also be read via `GET /api/tasks/{model}/{dataset}/` |
| `trial_*/result.json` | Result of a single case (status / reward / exception_info / timings / agent info) |
| `trial_*/exception.txt` | Exception message text of an errored case, for troubleshooting |
| `trial_*/trial.log` | Run log of a single case |
| `trial_*/agent/trajectory.json` | Agent trajectory; its presence means the case has a reproducible trajectory |
| `trial_*/verifier/reward.json` | Score of this case |
| `trial_*/verifier/ctrf.json` | Per test-case pass / fail / skip and failure info, for locating the exact failure point |
| `trial_*/verifier/test-stdout.txt` / `test-stderr.txt` | Verifier stdout / stderr tail |

> 💡 The real-time monitoring service (`--monitor-port`) reads the above files (`result.json` / `ctrf.json`, etc.) and uses mtime-based incremental caching to keep high-frequency refresh cheap; these files are also used to compute `correct / wrong / exception / avg_score` in real time for the board and the `/api/*` endpoints.

## Appendix: Custom Config File Parameters

### `models` (model service and agent parameters)

| Parameter | Corresponding CLI | Meaning |
| --- | --- | --- |
| `abbr` | - | Model abbreviation (unique identifier, used in result dir names) |
| `agent_name` | `-a/--agent` | Agent name (a Harbor AgentName) or a custom import path |
| `agent_import_path` | `--agent-import-path` | Import path of a custom agent (`module.path:ClassName`) |
| `model_names` | `--model` | Model name list, multi-valued |
| `api_base` | `--api-base` | Model service base url (unified semantic, auto-translated) |
| `agent_api_key` | `--agent-api-key` | Model service API key (unified semantic, auto-translated) |
| `llm_kwargs` | - | LLM call parameters (e.g. `max_tokens`), merged into agent kwargs |
| `model_info` | - | Model token limits and cost info (`max_input_tokens` / `max_output_tokens` / `input_cost_per_token` / `output_cost_per_token`) |
| `agent_kwargs` | `--ak/--agent-kwarg` | Extra raw agent kwargs (highest priority) |
| `agent_env` | `--ae/--agent-env` | Extra agent environment variables (highest priority) |
| `deps_path` | `--agent-deps` | Offline agent deps bundle path |
| `n_concurrent` | - | Concurrency upper bound per agent |
| `skills` / `mcp_servers` | - | Agent skill directories / MCP server configs |
| `resume_trajectory` / `load_trajectory` | - | Resume an agent session across steps / preload trajectory files |
| `extra_allowed_hosts` | - | Extra allowed hosts/IPs |
| `include_logs` / `exclude_logs` | - | Agent log globs to keep / exclude |
| `override_timeout_sec` / `override_setup_timeout_sec` / `max_timeout_sec` | - | Overrides and upper bound for agent run / environment setup timeouts |

### `datasets[].args` (agent evaluation task parameters)

| Parameter | Corresponding CLI | Meaning |
| --- | --- | --- |
| `path` | `-p/--agent-dataset-path` | Local dataset path (or a single task directory) |
| `dataset_name_version` | `-d/--dataset` | Remote dataset `name@version` / `org/name@ref` |
| `registry_url` / `registry_path` | - | Dataset registry url / path |
| `n_concurrent_trials` | `-n/--n-concurrent` | Concurrent trial count |
| `n_attempts` | `-k/--n-attempts` | Attempts per trial |
| `debug` | `--debug` | Enable debug logs |
| `quiet` | `-q/--quiet` | Suppress per-trial progress output |
| `timeout_multiplier` | `--timeout-multiplier` | Task timeout multiplier |
| `agent_timeout_multiplier` / `verifier_timeout_multiplier` / `agent_setup_timeout_multiplier` / `environment_build_timeout_multiplier` | - | Per-stage timeout multipliers |
| `max_retries` | `--max-retries` | Max retry count |
| `retry_include_exceptions` / `retry_exclude_exceptions` | - | Exception types to include / exclude for retry |
| `environment_type` | `-e/--environment` | Environment type (`docker`, `daytona`, `e2b`, `modal`, etc.) |
| `environment_force_build` | `--force-build/--no-force-build` | Whether to force rebuild the environment |
| `environment_delete` | `--delete/--no-delete` | Whether to delete the environment afterwards |
| `environment_kwargs` | `--host-network`, etc. | Extra environment kwargs (`--host-network` writes `{"host_network": True}`) |
| `environment_env` | - | Environment variables |
| `disable_verification` | `--disable-verification` | Disable the verifier |
| `verifier_env` / `verifier_import_path` / `verifier_kwargs` | - | Verifier environment variables / custom import path / extra kwargs |
| `task_names` | `--include-task-name` | Task names to include (supports glob) |
| `exclude_task_names` | `--exclude-task-name` | Task names to exclude (supports glob) |
| `n_tasks` | `--n-tasks` | Maximum number of tasks to take from the dataset |
| `yes` | `-y/--yes` | Auto-confirm environment variable prompts |
| `env_file` | `--env-file` | Path to a `.env` file |

### Others: `eval.runner`

| Parameter | Corresponding CLI | Meaning |
| --- | --- | --- |
| `monitor_port` | `--monitor-port` | Harbor monitoring HTTP service port (0 = off, default 0) |

> 📚 For more on Harbor environment preparation and terminal-bench 2 / 2.1 datasets and images, see [Harbor Terminal-Bench](../../extended_benchmark/agent/harbor_bench.md). For the full CLI parameter list, see 📚 [User Configuration Parameters - Agent Evaluation Parameters](../all_params/cli_args.md).