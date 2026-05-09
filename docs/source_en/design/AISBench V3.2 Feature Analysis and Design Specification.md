---
**Status:** Draft / Reviewing / Approved / Rejected / Superseded
**Authors:** @Your_Community
**Created:** 2026-05-09
**Updated:** 2026-05-09
**Related Issue/PR:** #285,#286,#279

---

# 1. Overview

## 1.1 Introduction
Unify and integrate SWE-Bench, BFCLV4, Tau2-Bench, and Terminal-Bench 2 into AISBench. Build a standardized evaluation pipeline covering **data reading → inference execution → result judgment → metric aggregation → sample auditing**, forming a configurable, reproducible, and traceable evaluation system.

## 1.2 Motivation
- SWE-Bench focuses on real-world software engineering issue resolution.
- BFCLV4 focuses on function/tool calling accuracy.
- Tau2-Bench evaluates tool usage and state interaction for conversational Agents.
- Terminal-Bench 2 (terminal-bench-2) assesses Agent performance on real-world workflows (security remediation, data processing, model training, etc.) inside containerized terminal environments.

All the above are mainstream public benchmarks for Agent capability evaluation. AISBench currently lacks unified support for these benchmarks, leading to scattered evaluation criteria, high delivery costs, and difficult horizontal comparison.

## 1.3 Objectives
Support datasets including SWE-Bench-Verified, SWE-Bench-Pro, Terminal-Bench, Tau2-Bench, and BFCL-V4 to cover Agent scenarios such as coding, function call, and tool call.

# 2. Use Case Analysis
Support datasets including SWE-Bench-Verified, SWE-Bench-Pro, Terminal-Bench, Tau2-Bench, and BFCL-V4 to cover Agent scenarios such as coding, function call, and tool call.

# 3. Solution Design

## 3.1 Overall Solution
Given the significant differences across the three types of benchmarks (code repair vs function calling vs conversational Agent), AISBench adopts a layered design:
**Unified Shell (registration, scheduling, data persistence, reporting) + Dataset-Specific Adapters (reader, executor, evaluator)**.
The three submodules can be delivered independently; overall objectives are achieved once each passes acceptance.

## 3.2 Technical Selection
Implement based on open-source datasets and adapt existing tool capabilities.

## 3.3 Functional & Performance Design

### 3.3.1 SWE-Bench Dataset Integration & Evaluation Capability
**Background**: Samples are real GitHub issues, relying on code repository context, patch generation, and test pass judgment.

#### Key Solution Points
- **Data**: Map issue descriptions, repository metadata, task identifiers to unified AISBench task input; standardize readers and fields.
- **Execution**: Provide infer/eval configuration templates; integrate patch artifacts and harness (e.g., mini-swe-agent); collect test judgment results.
- **Evaluation**: Aggregate pass-related core metrics; support sample-level judgment and failure auditing.

#### Acceptance Criteria
- **Model**: Qwen3-Coder-32B; **Platform**: NPU/GPU
- **Datasets**: swebench-lite, swebench-verify, swebench-multi
- **Metrics**: Use mini-swe-agent with post-processing `temperature=0`. Random fixed 10% subset per dataset; calculate pass@5. Accuracy deviation vs official SWE-Bench test method **< 1%**.

```sequenceDiagram
  participant Bench as AISBench Scheduler
  participant Data as SWE-Bench Data
  participant Agent as Inference & Patch Generation
  participant Harness as Test Harness
  participant Eval as Evaluation & Auditing
  Bench->>Data: Load issue samples
  Bench->>Agent: Infer and generate patch
  Agent->>Harness: Submit patch for testing
  Harness->>Eval: Pass/Fail & logs
  Eval->>Bench: Persist metrics & sample audit records
```


#### Usage Instructions
1. SWE-Bench: Configure dataset subset (lite/verified/multi), mini-swe-agent parameters, temperature=0, and fixed 10% sampling strategy (solidify seed and sample list path in config or docs).
2。 General Constraints: NPU/GPU platform; artifact directory complies with AISBench specifications; version info (dataset commit, toolchain version) written into runtime metadata.

#### Test Design
##### Unit Test
- Dataset readers: key field completeness, abnormal sample reporting.
- SWE-Bench: patch path and sample ID mapping.
- BFCLV4: function call parsing.
- Tau2-Bench: multi-turn message sequence integrity.

##### Integration Test
- End-to-end infer + eval pass for each submodule; complete log and result persistence; clear error messages for failure scenarios.

##### End-to-End / Official Alignment
- SWE-Bench：pass@5 与官方方法差异 < 1%（给定验收模型与子集）。

##### Acceptance Checkpoints (Verified via Testing)
- Cover data standardization, pipeline integration, evaluator and audit output for each submodule respectively (see internal requirement breakdown).

### 3.3.2：SWE-Bench-Pro Dataset Design
The key difference between Pro and standard SWE-Bench lies in evaluation: it is mandatory to pull images, mount workspaces, execute per-sample scripts, and parse test results via dedicated parsers.
AISBench encapsulates data standardization + patch protocol + execution backend adaptation + result schema.
Scope: SWE-bench Pro Dataset Integration & Evaluation Capability
Same as above: pull images, mount workspace, run per-case scripts, parse results. AISBench provides unified encapsulation for data, patch protocol, backend execution, and result schema.

sequenceDiagram
  participant AIS as AISBench
  participant DS as 数据集CSV或JSONL
  participant PJ as patch JSON
  participant Dock as Docker或Modal
  participant TS as run_script与parser
  AIS->>DS: 加载样本元数据
  AIS->>PJ: 读取预测patch
  AIS->>Dock: 启动容器dockerhub_tag
  Dock->>TS: 执行run_script
  TS->>AIS: 测试结果与日志
  AIS->>AIS: FAIL或PASS覆盖判定
  AIS->>AIS: eval_results与审计落盘

##### Data Reading & Field Standardization

- Support HF ScaleAI/SWE-bench_Pro or local exported CSV/JSONL.
- Mandatory fields: instance_id, dockerhub_tag (construct dockerhub_username/sweap-images:{dockerhub_tag}), before_repo_set_cmd, selected_test_files_to_run, fail_to_pass / pass_to_pass (or FAIL_TO_PASS/PASS_TO_PASS).
- Optional: base_commit, repo, base_dockerfile, instance_dockerfile.
- Explicit field standardization to avoid implicit column name guessing in evaluation scripts.

##### Patch Artifact Protocol

Compatible with swe_bench_pro_eval.py: list format [{instance_id, patch, prefix}], supports alias model_patch.
Define empty patch policy (count as submitted / direct failure / audit method); clarify prefix semantics for multi-run / multi-model comparison and output path indexing.
Align conventions with aggregation tools such as gather_patches.py.
##### Evaluation Execution Pipeline

Support at least one backend: local Docker or Modal (Modal recommended).
Align parameters: dockerhub_username, scripts_dir (default SWE-bench_Pro-os/run_scripts, configurable), num_workers, block_network, timeout, failure retry/skip logic.
Stable execution with fixed subset under Linux + Docker; retain stdout/stderr, patch snapshot, entry script snapshot for failures.

##### Metrics & Auditing

- Aggregation: resolved accuracy (optional submitted accuracy).
- Sample-level: instance_id → resolved(bool) + audit path (stdout/stderr, output.json, patch.diff, entryscript.sh, etc.).
- Optional failure classification: image pull failure, missing script, patch apply failure, test timeout, parser failure. 

##### Minimum Acceptance & Reproducibility
Fixed subset strategy (top-N by sorting or fixed seed sampling); lock Docker/Modal/dataset version; standardize input/output directory structure; maintain gold patch validation set.

#### Usage Instructions
- Prerequisites: Linux, Docker (optional Modal account & config); accessible DockerHub images; sufficient disk and network for image pulling.
- Configurable items: data path, patch JSON path, execution backend, dockerhub username, scripts directory, concurrency, timeout, network isolation.
- Input: protocol-compliant patch JSON; sample table and HF snapshot version recorded in metadata.
- Constraints: high resource consumption; slow cold start; clear error reporting and documentation for permission insufficiency.

#### Test Design
##### Unit Test
- CSV/JSONL/HF field parsing and missing field detection.
- Patch list schema validation and empty patch branch coverage.

##### Integration Test
- Minimum subset + gold patches: full eval pipeline pass with compliant eval_results.json structure.
- Simulate failures: missing image / missing script; verify failure classification and audit fields.

##### End-to-End / Alignment Acceptance
- Consistent resolved judgment with reference swe_bench_pro_eval.py under identical patch input.
- Deterministic judgment on repeated runs with same input and config.


### 3.3.3 BFCLV4 Dataset Integration & Evaluation Capability
Background: Berkeley Function Calling Leaderboard V4. Evaluates function selection, parameter construction, format compliance, and tool calling in multi-turn/Agent tasks.
#### Key Solution Points

- Data: Standardize mapping of natural language requests, function schemas, calling constraints, and - multi-category subset fields.
- Execution: Function calling inference entry; output parsing and multi-turn trajectory collection.
- Evaluation: Dedicated BFCLV4 evaluator and aggregated metrics at subtask dimension.
#### Acceptance Criteria

- Model: Qwen3-235B-A22B-Instruct-2507; Platform: NPU/GPU
- temperature=0; single-turn accuracy deviation vs official Gorilla README test method < 1%; multi-turn accuracy deviation vs official leaderboard < 1%.

#### Usage Instructions
1. BFCLV4: Separate single-turn / multi-turn evaluation config; specify model and temperature; integrate official scoring script or equivalent implementation.
2. General Constraints: NPU/GPU platform; artifact directory complies with AISBench specifications; version info recorded in runtime metadata.


#### Test Design
##### Unit Test
- Dataset readers: key field completeness, abnormal sample reporting.

##### Integration Test
- End-to-end infer + eval pass; complete log and result persistence; clear error messages for failure scenarios.

##### End-to-End / Official Alignment
- BFCLV4: single-turn and multi-turn deviation < 1% respectively.

##### Acceptance Checkpoints
- Cover data standardization, pipeline integration, evaluator and audit output for each submodule.


### 3.3.4 Tau2-Bench Dataset Integration & Evaluation Capability
Background: Conversational Agent benchmark focusing on multi-turn decision-making and execution reliability in tool-enabled environments.

#### Key Solution Points

- Data: Parse and register multi-turn context, tool definitions, and environment state fields.
- Execution: Integrate multi-turn Agent execution; structurally record interaction trajectories and calling results.
- Evaluation: Task completion metrics; sample-level execution auditing.

#### Acceptance Criteria

- Model: Qwen3-Coder-32B; Platform: NPU/GPU
- Datasets: airline, retail, telecom
- temperature=0; select fixed top 20% cases per subcategory; pass@5 accuracy deviation vs official test method < 1%.
(Refer to official Tau2-Bench process and pass@5 definition during implementation.)

#### Scope Impact
Add three new dataset task packages, configuration templates and evaluators; may extend general function call/Agent execution abstraction. Align with existing accuracy report fields to avoid breaking legacy QA tasks.

#### Usage Instructions
1. Tau2-Bench: specify domain (airline/retail/telecom), 20% subset rule, pass@5 calculation method and referenced version.
2. General Constraints: NPU/GPU platform; artifact directory complies with AISBench specifications; version info recorded.

#### Test Design
Same unit / integration / end-to-end test specifications as above; pass@5 alignment deviation < 1%.

### 3.3.5 Terminal-Bench Dataset Integration & Evaluation Capability

#### Task Discovery & Field Standardization (Task Loading)

- task_id / task_path; instruction (instruction.md)
- Environment: docker_image or environment/Dockerfile; cpus, memory, storage, build_timeout_sec
- Timeout: agent_timeout_sec, verifier_timeout_sec
- Verification: verifier_script (default tests/test.sh); artifacts /logs/verifier/reward.txt, /logs/verifier/ctrf.json
- Metadata: difficulty, category, tags, author info (parsed from task.toml sections: [metadata], [agent], [verifier], [environment])

#### Container Execution & Orchestration

- Startup: pull docker_image or build from Dockerfile.
- Mount: task working directory, result path, /logs collection; ensure verifier write permission.
- Agent Output Protocol: document and validate critical required artifact paths (e.g., /app/results.txt).
- Resource & Isolation: enforce task.toml resource limits and concurrency isolation to avoid cross-task contamination and log overwriting.

#### Evaluation Judgment (Reward / CTRF)

- Final pass/fail determined solely by numeric value (0/1) in /logs/verifier/reward.txt.
- Keep ctrf.json for test-level auditing (failed cases, stack trace, latency).
- Failure classification: image pull failure, container startup failure, dependency installation failure, with observable error logs.

#### Metrics & Auditing (Aggregation & Persistence)

- Aggregation: pass rate (optional weighted/layered statistics).
- Grouping: by category, difficulty, tags.
- Sample-level: task_id, judgment result, reward/CTRF path, key log path, failure type (timeout / env_error / verifier_error, etc.).

#### Minimum Validation Set & Reproducibility

- Fixed subset strategy (top-N / fixed seed); lock image tag and repo commit; fix resource quota and timeout; standardize concurrency and isolation policy.
- Validation set covers multiple categories (security, model-training, data-processing).
- External Reference: Official TB2 Harbor CLI (harbor run --dataset terminal-bench@2.0 ...). AISBench must maintain consistent behavior, judgment criteria and audit artifacts.

#### Usage Instructions
1. Prerequisites: Docker available; disk/memory meet peak task requirements; Agent executor supports non-interactive / pseudo-terminal execution inside containers.
2. Configuration: task subset list or filter rules, image pull policy, resource ceiling, timeout, concurrency, host log root path.
3. Judgment: Forbid custom pass/fail logic bypassing test.sh, except explicitly marked debug mode (excluded from official evaluation).
4. Reproducibility: Record task repo commit, image digest, AISBench version on each run.


#### Test Design
##### Unit Test
- task.toml parsing and default value injection; error on missing mandatory fields.
- reward parsing: mock reward.txt 0/1 and non-numeric exception handling.

##### Integration Test
- Fixed minimal subset: full infer (or mock Agent) + verifier pipeline; collect reward and ctrf outputs.
- Fault injection: missing image / timeout; verify failure classification and audit fields.

##### End-to-End / Official Alignment
- Deterministic judgment on repeated runs with same input; discrepancies explainable via logs.
- Verify troubleshooting guidelines in user docs for missing dependencies.

##### Acceptance Checkpoints
- Cover task loading, orchestration compliance, reward/CTRF judgment, aggregation & auditing, minimal validation set.


## 3.4 Security, Privacy & DFX Design

This release covers the following security aspects:
1. **API Key Security Management**: Delivered via environment variables or config files; never printed in logs.
2. **Network Transmission Security**： Support HTTPS to protect data transmission.
3. **File Access Security**: Validate file paths to prevent path traversal attacks.
4. **Server-Side Security**: Validate server URL to prevent SSRF attacks.

### 3.4.1 API Key Security Management

**Security Measures**：

- Never print API Key in logs to prevent leakage.
- Pass API Key via environment variables; avoid exposure in CLI arguments.
- Support HTTPS to secure API Key transmission.
- Return clear error prompts on key errors without exposing key content.

**Risk Analysis**：

- Risk Level: Medium
- Impact: Leakage may cause unauthorized access.
- Mitigation: Environment variable delivery, HTTPS enablement, log desensitization.

### 3.4.2 Network Transmission Security

**Security Measures**：

- Encrypt transmission via HTTPS.
- Support SSL certificate verification to prevent man-in-the-middle attacks.
- Support connection timeout control to avoid persistent idle connections.

**Risk Analysis**：

- Risk Level: Medium
- Impact: Eavesdropping may lead to data leakage.
- Mitigation: HTTPS enforcement, SSL validation.

### 3.4.3 File Access Security

**Security measures**:

- Validate file paths to prevent path traversal attacks
- Restrict file access scope to only user-specified files
- Validate file formats to prevent malicious files

**Risk analysis**:

- Risk level: Low
- Impact: Path traversal attacks may lead to unauthorized file access
- Mitigation measures: Path validation, access scope restriction

### 3.4.4 Privacy Protection

**Privacy data**:

- This tool does not collect personal data
- Evaluation results contain only evaluation metrics, not raw data
- Logs contain no sensitive information

**Privacy protection measures**:

- Do not collect or store personal data
- Evaluation result files contain only evaluation metrics
- Logs are desensitized, sensitive information not printed

### 3.4.5 Resilience Design

**Fault tolerance mechanisms**:

- Automatic retry on network request failure
- Failure of a single data inference does not affect other data
- Failure of partial functions does not affect core functionality

**Recovery mechanisms**:

- Support recovery from checkpoints
- Support backup and restoration of result files
- Support backup and restoration of configuration files

### 3.4.5 Reliability & Availability Design

#### 3.4.5.1 Redundancy Design

**Configuration parameter backup**:

- Key configuration parameters (model configuration, dataset configuration, etc.) support configuration file backup
- Support version management of configuration files for easy rollback

**Data backup**:

- Evaluation results are automatically saved; support backup and restoration of result files
- Support version management of evaluation results for comparative analysis

**Recovery strategies**:

- If a configuration file is corrupted, use default configuration or prompt the user to fix it
- If evaluation results are lost, support re-executing the evaluation task
- Support checkpoint recovery to avoid redundant computation

#### 3.4.5.2 Fault Management

**Fault detection**:

- Network connection failure: Automatically detect vLLM service connection status; retry automatically on connection failure
- Dataset loading failure: Detect whether dataset files exist and have correct format
- Evaluation computation failure: Detect anomalies during evaluation metric calculation and log error information

**Fault isolation**:

- Failure of a single data inference does not affect processing of other data
- Failure to load a single dataset does not affect processing of other datasets
- Failure of a single evaluator computation does not affect execution of other evaluators

**Fault localization**:

- Detailed error logs including error type, location, and cause
- Support debug mode to output more detailed debugging information
- Support error stack traces to facilitate problem localization

**Fault recovery**:

- Network request failure: Automatic retry (2 retries by default); if retries fail, log the error
- Dataset loading failure: Prompt the user to check the dataset file; do not continue execution
- Evaluation computation failure: Log the failed data and continue processing other data

**Alerting design**:

- Critical errors (e.g., service connection failure) log at ERROR level
- General errors (e.g., single data inference failure) log at WARNING level
- Support configurable log level to control log output

#### 3.4.5.3 Overload Control Design

**Traffic detection**:

- Monitor the number of concurrent requests to avoid exceeding system capacity
- Monitor memory usage to avoid out-of-memory errors
- Monitor CPU usage to avoid CPU overload

**Rate limiting mechanisms**:

- Support limiting the number of parallel tasks via the `--max-num-workers` parameter
- Support controlling the request sending rate via the `request_rate` parameter
- Support controlling batch size via the `batch_size` parameter

**Overload handling**:

- When concurrency is too high, automatically queue requests
- When memory is insufficient, prompt the user to reduce concurrency or dataset size
- When CPU is overloaded, automatically reduce concurrency

**Graceful degradation**:

- Non-critical features (e.g., detailed logging) can be degraded
- Core features (e.g., inference and evaluation) are prioritized
- Support partial result output; even if some data fails, already processed results can still be output

#### 3.4.5.4 Upgrade Without Service Interruption

**Version compatibility**:

- New versions support configuration files and command-line arguments from old versions
- New versions support dataset formats from old versions
- New versions support evaluation result formats from old versions

**Upgrade strategy**:

- Support smooth upgrades without stopping existing tasks
- Support configuration migration, automatically converting old version configurations to new version format
- Support rollback; if upgrade fails, rollback to the old version

**Data compatibility**:

- Evaluation result format is forward-compatible; new versions can read old version results
- Configuration file format is forward-compatible; new versions can read old version configurations
- Dataset format is forward-compatible; new versions can read old version datasets

#### 3.4.5.5 Human Error Prevention Design

**Configuration error protection**:

- Parameter validation: All command-line and configuration parameters are validated
- Default values: Provide reasonable default values to reduce configuration errors
- Error hints: Provide clear error hints and remediation suggestions for configuration errors

**Operation error protection**:

- Dangerous operation prompts: Provide confirmation prompts for operations that may affect data
- Operation logging: All operations are logged for traceability
- Quick rollback: Support operation rollback; configuration errors can be quickly recovered

**Data protection**:

- Evaluation results are automatically saved to avoid data loss
- Support result file backup to prevent accidental deletion
- Support result file version management for comparative analysis

#### 3.4.5.6 Fault Prediction and Prevention Design

**Resource monitoring**:

- Monitor memory usage to provide early warning of insufficient memory
- Monitor disk space to provide early warning of insufficient disk space
- Monitor network connections to provide early warning of network issues

**Health checks**:

- Periodically check the health status of the vLLM service
- Periodically check the integrity of dataset files
- Periodically check the validity of configuration files

**Preventive measures**:

- Support resource usage limits to prevent resource exhaustion
- Support automatic temporary file cleanup to prevent disk space exhaustion
- Support connection pool management to prevent connection leaks

## 3.5 Programming and API Design

### 3.5.1 Interface Definition and Design

In this proposal, the main changes relate to the definitions of external interfaces and internal input/output files that need to be adapted during agent evaluation.

1. **External interfaces**:
   - HuggingFace datasets: `load_dataset(hf_id, split=...)` – online dataset pulling
   - mini-swe-agent: `process_instance(...)` – generate instance patches
   - SWE-bench harness: `run_instance(...)` – execute instance evaluation
   - Docker: `image inspect/pull` and container cleanup interfaces

2. **Internal interfaces**:
   - Inference output: `predictions/<model>/<dataset>.json` (instance_id to predicted object mapping)
   - Evaluation input: prediction file; Evaluation output: aggregated report `results/<model>/<dataset>.json`
   - Summarizer input: aggregated evaluation results; output: unified accuracy view and statistical fields


---

## Appendix

* **Reference links.**
1. [SWE-Bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
2. [Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces](https://arxiv.org/abs/2601.11868)
3. [Humanity's Last Exam](https://arxiv.org/abs/2501.14249)
4. [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
5. [terminal-bench](https://github.com/harbor-framework/terminal-bench)

* **Glossary.**
1. HLE: Humanity's Last Exam

* **Document update plan**
1. June 30: Update detailed design for multimodal generation dataset integration; solution for improving SWE-Bench evaluation efficiency.



  
