# SWE-Bench Pro User Guide

SWE-Bench Pro is a challenging benchmark evaluating LLMs/Agents on long-horizon software engineering tasks. Given a repository and an issue, the model is expected to generate a patch that fixes the described problem.

> **Note**: Since the official Docker images are all x86 architecture, SWE-bench Pro currently only supports evaluation on x86 environments. ARM environments are not supported.

## 1. Feature Overview

`ais_bench` currently supports the following SWE-Bench Pro capabilities:

- Datasets: `full`, `mini`
- Tasks:
  - `infer`: call `mini-swe-agent` to generate patches (`model_patch`)
  - `eval`: call the SWE-bench Pro harness to run evaluation and count resolved instances
- Result summary: output key metrics such as `accuracy` and `eval_resolved_instances_num`

Directory `ais_bench/configs/swe_bench_pro_examples/` provides the following example configs:

- `mini_swe_agent_swe_bench_pro_mini.py`: SWE-bench Pro Mini — commonly used for quick iterations.
- `mini_swe_agent_swe_bench_pro_full.py`: SWE-bench Pro Full — the full test set.

## 2. Prerequisites

Before running, make sure the following dependencies are available:

1) Install `mini-swe-agent` (required for infer)

> **Note**: SWE-Bench Pro official organization scaleapi has adapted mini-swe-agent. You need to download the adapted version from scaleapi's repository.

```bash
# Clone mini-swe-agent repository
git clone https://github.com/scaleapi/mini-swe-agent.git

# Enter the project directory
cd mini-swe-agent/

# Install dependencies
pip install -e .

# Return to parent directory
cd -
```

2) Install `SWE-Bench_Pro` (required for infer and eval)

```bash
# Clone SWE-Bench_Pro repository
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git

# Enter the project directory
cd SWE-bench_Pro-os/

# Install dependencies
pip install -r requirements.txt

# Return to parent directory
cd -
```

3) Docker is available (both infer and eval depend on containerized environments)

```bash
docker --version
docker ps
```

## 3. Minimal Configuration (Run First, Tune Later)

It is recommended to start from `mini_swe_agent_swe_bench_pro_mini.py` and only modify the three fields in `models[0]`:

- `model`: model name (required)
- `url`: model service endpoint (OpenAI-compatible API)
- `api_key`: service key (use `EMPTY` for local services)

Example (local vLLM setup):

```python
models = [
    dict(
        attr="local",
        abbr="swebench_pro_mini_module",
        type="LiteLLMChat",
        model="qwen3",
        api_key="EMPTY",
        url="http://127.0.0.1:8000/v1",
        batch_size=4,
        generation_kwargs=dict(),
    )
]
```

### Dataset Path Notes

Different datasets have different loading methods:

- **full dataset**: Supports both online loading from Hugging Face and local loading
  - Online loading: keep `path=""`
  - Local loading: set `path` to local parquet file or directory

- **mini dataset**: **Must be prepared locally in advance**, cannot be fetched online
  - Download URL: `https://modelers.cn/datasets/AISBench/SWE-Bench_Pro_mini`
  - Recommended format: parquet
  - Set `path` to the locally downloaded parquet file or directory

### SWEBP Scripts and Docker Directory Configuration

SWE-Bench Pro evaluation **must** specify the following two paths. There is no default behavior:

- `swebp_scripts_dir`: Absolute path to the `run_scripts` directory of the SWE-bench Pro official repository
- `swebp_docker_dir`: Absolute path to the `dockerfiles` directory of the SWE-bench Pro official repository

```python
SWEBP_SCRIPT_PATH_ABS = "{your_work_dir}/SWE-bench_Pro-os/run_scripts"  # Must be specified
SWEBP_DOCKER_PATH_ABS = "{your_work_dir}/SWE-bench_Pro-os/dockerfiles"  # Must be specified
```

> **Note**: You need to clone the SWE-bench Pro official repository first: `git clone https://github.com/scaleapi/SWE-bench_Pro-os.git`

### First-Run Recommendations

- Start with the `mini` dataset
- Use `batch_size=4` (each instance creates a container; large batch_size may cause host OOM)
- Keep `step_limit=250` (default in examples; do not change initially)

## 4. Run Commands

Run the following in the repository root (`config` is the config file path):

```bash
ais_bench ais_bench/configs/swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_mini.py
```

The command above runs the full pipeline (`all`). You can also run it step by step:

```bash
# Inference only, generate predictions
ais_bench ais_bench/configs/swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_mini.py -m infer

# Evaluate based on existing predictions
ais_bench ais_bench/configs/swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_mini.py -m eval
```

### Resume from Checkpoint

Use `--reuse` to skip completed instances, which is useful after interruptions:

```bash
ais_bench ais_bench/configs/swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_mini.py -m infer --reuse
```

## 5. How to Read Outputs

The default output directory is `outputs/default/<timestamp>/`. The directory structure is as follows:

### Inference Outputs (predictions)

```
├── predictions
│   └── swebench_pro_mini_model
│       ├── swebench_pro_mini_data
│       │   ├── exit_statuses.yaml     # Exit status statistics for all instances
│       │   │
│       │   ├── instance_gravitational__teleport-xxx    # Directory for instance xxx
│       │   │   ├── instance_gravitational__teleport-xxx.config.yaml
│       │   │   ├── instance_gravitational__teleport-xxx.debug.log
│       │   │   ├── instance_gravitational__teleport-xxx.info.log
│       │   │   ├── instance_gravitational__teleport-xxx.pred
│       │   │   └── instance_gravitational__teleport-xxx.traj.json
│       │   │
│       │   └── instance_qutebrowser__qutebrowser-yyy   # Directory for instance yyy
│       │       ├── instance_qutebrowser__qutebrowser-yyy.config.yaml
│       │       ├── instance_qutebrowser__qutebrowser-yyy.debug.log
│       │       ├── instance_qutebrowser__qutebrowser-yyy.info.log
│       │       ├── instance_qutebrowser__qutebrowser-yyy.pred
│       │       └── instance_qutebrowser__qutebrowser-yyy.traj.json
│       │
│       └── swebench_pro_mini_data.json    # Final inference results
```

### Evaluation Outputs (results)

```
├── results
│   ├── swebench_pro_mini_model
│   │   ├── instance_gravitational__teleport-xxx      # Directory for instance xxx
│   │   │   ├── swebench_pro_mini_data_entryscript.sh
│   │   │   ├── swebench_pro_mini_data_output.json
│   │   │   ├── swebench_pro_mini_data_patch.diff
│   │   │   ├── swebench_pro_mini_data_stderr.log
│   │   │   ├── swebench_pro_mini_data_stdout.log
│   │   │   └── workspace
│   │   │
│   │   └── instance_qutebrowser__qutebrowser-yyy      # Directory for instance yyy
│   │       ├── swebench_pro_mini_data_entryscript.sh
│   │       ├── swebench_pro_mini_data_output.json
│   │       ├── swebench_pro_mini_data_patch.diff
│   │       ├── swebench_pro_mini_data_stderr.log
│   │       ├── swebench_pro_mini_data_stdout.log
│   │       └── workspace
│   │
│   └── swebench_pro_mini_model_swebench_pro_mini_data_report.json   # Final evaluation report
```

### Key Fields in Evaluation Report

```json
{
  "total_instances_num": 2,   // Number of dataset instances
  "total_prediction_num": 2,  // Number of inference results
  "build_patch_instances_num": 2,   // Number of instances that successfully generated patches within the step limit
  "empty_patch_instances_num": 0,   // Number of instances with empty patches (failed to complete within step limit)
  "eval_resolved_instances_num": 1,  // Number of instances evaluated as "resolved"
  "eval_unresolved_instances_num": 1,  // Number of instances evaluated as "unresolved"
  "empty_patch_instances_ids": [],   // Instance IDs that failed to complete within step limit
  "unresolved_instances_ids": [
    "instance_gravitational__teleport-xxx"   // Instance IDs evaluated as "unresolved"
  ],
  "accuracy": 50.0   // Final evaluation accuracy
}
```

## 6. Common Issues and Troubleshooting (SWEBP Error Codes)

The following error codes come from `SWEBP_CODES`. You can also refer to the full FAQ:

- FAQ: `docs/source_en/faqs/error_codes.md`

### 1) `SWEBP-DEPENDENCY-001`: Missing mini-swe-agent

- Symptom: infer fails to start with dependency import errors
- Cause: `mini-swe-agent` is not installed
- Fix: run `pip install mini-swe-agent`

### 2) `SWEBP-PARAM-001`: Empty model configuration

- Symptom: prompt indicates model is not configured
- Cause: `models[0]['model']` is empty or only whitespace
- Fix: configure `model/url/api_key`, and ensure `model` is non-empty at minimum

### 3) `SWEBP-PARAM-002`: Invalid dataset name

- Symptom: prompt indicates dataset name is not supported
- Cause: `name` is not in the supported set `full`, `mini`
- Fix: set dataset `name` to one of the supported values

### 4) `SWEBP-DATA-001`: Dataset loading failure

- Symptom: online loading fails, or local files cannot be found
- Cause:
  - Online mode: network or Hugging Face access issues (full dataset only)
  - Local mode: `path` does not exist, or file format is not supported
- Fix:
  - For full dataset: switch to local parquet file if online loading fails
  - For mini dataset: ensure it has been downloaded from `https://modelers.cn/datasets/AISBench/SWE-Bench_Pro_mini`

### 5) `SWEBP-FILE-001`: Predictions file not found

- Symptom: `-m eval` reports missing predictions
- Cause: infer was not run first, or work_dir/reuse points to a different location
- Fix: run `-m infer` first, and ensure eval and infer use the same config/output directory

### 6) `SWEBP-RUNTIME-001` / `SWEBP-RUNTIME-002`: Container or harness runtime failure

- Symptom: Docker image pull failure, or evaluation runtime errors
- Cause: unavailable images, network issues, insufficient container runtime environment, or host OOM due to insufficient memory
- Fix:
  - Check `docker ps` first
  - Verify images can be pulled from Docker Hub
  - If host memory is insufficient, reduce `batch_size` (recommended <= 4)
  - Retry with `--reuse` to avoid recomputing completed instances

## 7. Advanced Tips (Optional)

- For initial debugging, use `mini` first, then switch to `full` after the pipeline is stable
- To reduce empty patches, prioritize improving model capability and agent prompt templates
- During evaluation, focus on `empty_patch_instances_ids` and `unresolved_instances_ids`; they are often more actionable than `accuracy` in early iterations
- SWE-Bench Pro uses Docker images for evaluation; ensure stable network for faster image pulling
- Control `batch_size` carefully to avoid container exit due to host memory exhaustion
