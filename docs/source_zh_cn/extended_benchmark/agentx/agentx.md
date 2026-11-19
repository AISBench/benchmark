# AgentX Weka Trace数据集性能测评

本文介绍如何在 AISBench 中评测 AgentX Weka Trace数据集，覆盖简介、快速入门与注意事项。参数全量说明见 [命令行参数](../../base_tutorials/all_params/cli_args.md)。

## 1. 简介

### 1.1 AgentX 是什么

AgentX是面向真实智能体负载的性能压测工具。它以 Weka 采集的真实 coding-agent 会话轨迹（`traces.jsonl`）为单位回放多轮会话 DAG（含工具调用、上下文增长等特征），在独立进程中完成预热与 profiling，输出 TTFT、ITL、吞吐、有效并发等指标。

### 1.2 为什么要接入 AISBench 体系

现有 AISBench 的大模型性能测评主要围绕传统 LLM 请求负载展开，难以充分模拟真实 Agent 应用中的多轮交互、长上下文、请求间隔及并发会话等复杂负载特征，无法全面反映推理服务在 Agentic 场景下的性能表现。

为完善 AISBench 的 Agentic 推理性能测评能力，需要集成 AgentX 开源测评框架及其公开评测集，建立标准化的 Agentic 负载执行、性能指标采集和结果分析能力。

| 目标      | AISBench 提供的保障                                                             |
| ------- | -------------------------------------------------------------------------- |
| 统一入口    | 一条 `ais_bench --mode agentx` 完成环境探测、命令合成、执行与汇总，不需要手工拼 AgentX 参数           |
| 环境可验证   | 启动前校验 `agentx-harness` 版本、Python 版本、是否混装 upstream `aiperf`，并做入口自检          |
| 数据可溯源   | `full` / `256k` 两个官方语料登记 HF repo、revision、sha256；本地语料逐字节校验；manifest 记录语料来源 |
| 口径可判定   | 结果区分 `official_agentx` / `internal_only` / `smoke` / `invalid`，并给出无效原因     |
| 工程化能力   | 多并发档串行、档间冷却、失败即停、`--reuse` 覆盖重跑、`--dry-run` 只出计划                           |
| 可审计、可脱敏 | 最终 argv、被忽略的 extra args、耗时、错误码全部落盘；日志与配置快照自动脱敏                             |

### 1.3 两种模式

| 模式                  | 适用场景                    | 必填参数                                                       |
| ------------------- | ----------------------- | ---------------------------------------------------------- |
| `--mode agentx`     | 常规使用，由 AISBench 合成命令    | `--url`、`--model-name`、`--path`、`--agentx-dataset-variant` |
| `--mode agentx_raw` | 调试或使用 AISBench 尚未封装的新参数 | 仅 `--agentx-extra-arg`，命令主体由该参数描述                          |

`agentx_raw` 只接受 `--agentx-python/--agentx-bin/--agentx-env/--agentx-extra-arg` 四个子进程参数及 `--work-dir/--reuse/--dry-run` 等运行控制参数，其余参数在解析阶段直接报错。

## 2. 快速入门

### 2.1 前置准备

1. 准备**独立**的 AgentX 环境（Python 3.11\~3.13，不能和 AISBench 基础环境共部署）。

   1）按仓库依赖清单安装 AgentX（已安装 AISBench 基础环境）

   Conda 方式（推荐）：
   ```bash
   conda create -n agentx113 python=3.11 -y
   conda run -n agentx113 python -m pip install -r requirements/agentx.txt
   # 查询解释器路径，供 --agentx-python 使用
   conda run -n agentx113 which python
   ```
   venv 方式：
   ```bash
   python3.11 -m venv /opt/venvs/agentx113
   /opt/venvs/agentx113/bin/pip install -r requirements/agentx.txt
   ```
   `requirements/agentx.txt` 固定 `agentx-harness==1.0.13`。该环境不能与 AISBench 共用：AISBench 依赖 `Pillow==11.2.1`，AgentX 要求 `pillow~=12.3.0`，同环境安装会直接解析失败。

   2）（可选）使用 Docker 部署 AgentX 环境

   参考[镜像概览](https://github.com/AISBench/benchmark/blob/master/docker/OVERVIEW.zh.md)的“快速开始 → 运行已有镜像”章节启动 AgentX 容器（AgentX 镜像已内置 AISBench 和 AgentX 环境）。
2. 准备模型服务：OpenAI 兼容的 Chat Completions 接口。推理模型上下文窗口根据测试语料调整，至少为256K tokens。使用`vllm`部署模型时，需要设置`--max-model-len`>=262144，建议显示开启--enable-prefix-caching、--enable-prompt-tokens-details。
3. 准备 tokenizer：模型权重目录必须存在，会作为 `--tokenizer` 传给 AgentX。
4. 准备语料：不传 `--agentx-dataset-path` 时由 AgentX 自动下载公共语料；离线环境请传本地 `traces.jsonl` 所在目录。

### 2.2 CLI 命令启动

#### 2.2.1 冒烟测试

冒烟用于验证链路，不是正式结果：

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

`smoke` 模式下未显式指定 `--agentx-duration` / `--agentx-trace-limit` 时，默认运行 60 秒、发 4 条 trace。冒烟结果统一标记 `comparison_scope=smoke`。

#### 2.2.2 正式运行

正式口径要求 `benchmark` 模式且时长不小于 900 秒；未指定时长时沿用 AgentX 预设 3600 秒。多个并发档写在同一个 `--batch-size` 里，按顺序串行执行：

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

上例使用官方公共语料（由 AgentX 下载）；如需离线或自定义语料，追加 `--agentx-dataset-path`，此时有效结果的口径为 `internal_only`。

运行结束后终端会打印每个档位的执行结论与汇总表：

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

#### 2.2.3 高级配置

**本地语料与额外环境变量**

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

`--agentx-env` 可重复，`KEY=VALUE` 格式，只作用于 AgentX 子进程。

- `--cooldown` 控制档间等待：多档默认 5 秒，单档只有显式传入才等待。
- `--agentx-dataset-path` 指定的本地语料会在非 smoke 模式下校验 `traces.jsonl` 的 sha256 与官方登记值，不一致直接失败；smoke 允许继续并记录不一致。

**追加 AgentX 原始参数**


```bash
ais_bench --mode agentx ... \
  --agentx-extra-arg '{ --max-context-length 204800 --random-seed 0 --scenario agentx }'
```

AISBench 已生成的参数优先级更高（`--url`、`--model`、`--concurrency`、`--output-artifact-dir`、语料来源等），extra args 里的同名片段（含取值）会被忽略并记录到 `agentx_command.ignored_extra_args`。`--scenario` 由 extra args 决定，未传入时按入口默认值 `agentx` 记录。

**只出执行计划**

```bash
ais_bench --mode agentx ... --dry-run
```

`--dry-run` 仍执行参数校验、语料哈希校验与 AgentX 环境探测，但不启动 AgentX、不访问模型服务，只写脱敏后的 `execution_plan.redacted.json`。

**复跑同一目录**

```bash
ais_bench --mode agentx ... --reuse --work-dir ./agentx_bench
```

`--reuse` 复用已有目录，仅清理本轮会重新生成的 `manifest.json` 与 `profile_export*`，其余文件保留；不加 `--reuse` 且产物目录非空时直接报错，避免把上一轮结果当成本轮产物。

**完全自定义命令（agentx\_raw）**

```bash
ais_bench --mode agentx_raw \
  --work-dir ./agentx_raw_run \
  --agentx-python /opt/venvs/agentx113/bin/python \
  --agentx-extra-arg '{ --scenario agentx --url http://127.0.0.1:8000 --endpoint /v1/chat/completions --model qwen3.8 --concurrency 8 --public-dataset semianalysis_cc_traces_weka_062126_256k --benchmark-duration 900 }'
```

extra args 只需写 `profile` 之后的选项，`agentx profile` 由 AISBench 自动拼接。

### 2.3 配置文件启动

支持把模型、语料和运行参数写入 Python 配置文件，再作为位置参数传给 `ais_bench`（`--mode agentx` 仍需显式指定）：

```bash
ais_bench ais_bench/configs/agentx_example/agentx_weka_trace.py --mode agentx
```

最小配置示例（完整版见仓库 [agentx_weka_trace.py](https://github.com/AISBench/benchmark/blob/master/ais_bench/configs/agentx_example/agentx_weka_trace.py)）：

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

当配置文件和CLI参数同时存在时，传入的CLI同名字段优先级高于配置文件，`--work-dir`、`--reuse`、`--dry-run` 等运行控制参数也照常生效。未在命令行传入的参数保持配置文件中的值。

### 2.4 执行结果

产物布局（`<run_id>` 形如 `agentx_256k_batchsize_4`，raw 模式为 `agentx_raw_batchsize_<N>`）：

```text
<work_dir>/
├── configs/<cfg_time>_<pid>.py
├── logs/<run_id>/
│   ├── agentx_stdout.log              # AgentX 原始输出（已脱敏）
│   ├── agentx_adapter.log             # AISBench 适配层日志
│   └── agentx_summary.log             # 汇总表与逐任务明细
└── performances/<run_id>/
    ├── agentx_raw/                    # AgentX 原始产物（profile_export_aiperf.json/csv、console 等）
    ├── agentx_summary.json            # 性能结果json
    ├── agentx_summary.csv             # 性能结果csv
    ├── execution_plan.redacted.json   # 脱敏后的执行计划
    └── manifest.json                  # 运行事实、语料溯源、判定与指标
```

`manifest.json` 的关键字段：

| 字段                                                   | 含义                                                                                           |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `run_params` / `test_params`                         | 运行模式、时长、trace 上限、并发档等实际生效值                                                                   |
| `agentx_command.parameters`                          | 最终命令解析出的完整参数                                                                                 |
| `agentx_command.extra_args` / `ignored_extra_args`   | 透传参数，以及因冲突被忽略的片段                                                                             |
| `dataset`                                            | 语料来源、variant、alias、repo/revision、sha256 校验结果、`registered`                                    |
| `result.status` / `result.parse_status`              | 进程状态与结果解析状态                                                                                  |
| `result.benchmark_valid` / `result.comparison_scope` | 是否为有效正式结果与可比口径                                                                               |
| `result.invalid_reasons`                             | 判定无效的首个根因，完整列表见 `all_invalid_reasons`                                                        |
| `result.errors`                                      | 进程、探测、结果解析等阶段的错误码与信息                                                                         |
| `result.summary_json` / `result.summary_csv`         | 指向 `agentx_summary.json` 与 `agentx_summary.csv`，指标本体（请求数、吞吐、TTFT/ITL/latency p90 等）在该文件与汇总表中 |

`comparison_scope` 判定规则：

| 取值                | 含义                                |
| ----------------- | --------------------------------- |
| `official_agentx` | benchmark 模式 + 官方公共语料 + 上游提交有效    |
| `internal_only`   | 使用本地语料，只能内部横向对比                   |
| `smoke`           | smoke 模式，仅用于链路验证                  |
| `invalid`         | 以上条件不满足，例如语料未注册、哈希不一致、进程失败、指标覆盖不足 |

## 3. 注意事项

### 3.1 版本与兼容性

- AISBench固定使用 `agentx-harness==1.0.13`，Python 3.11\~3.13；
- AgentX 环境必须与 AISBench 环境隔离（`Pillow` 版本互斥），且不能同时安装 PyPI 上的 upstream `aiperf`。

### 3.2 参数与口径约束

- `--agentx-dataset-variant` 必须显式指定 `full` 或 `256k`。
- `--agentx-run-mode benchmark` 要求 `--agentx-duration >= 900`；`--agentx-trace-limit` 仅 smoke 可用。
- extra args 与 AISBench 生成参数冲突时以 AISBench 为准；`--public-dataset` 与 `--hf-weka-dataset` 互斥，跨选项切换语料会被忽略。