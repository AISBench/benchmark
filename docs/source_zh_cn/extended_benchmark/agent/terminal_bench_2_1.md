# Terminal-Bench 2.1 使用指南

Terminal-Bench 2.1 是 Terminal-Bench 2.0 的 bugfix 版本，用于评测 AI Agent 在真实终端环境中完成多步骤任务的能力。给定一个隔离的容器环境与任务描述，Agent 需要自主使用终端完成安装、配置、编码、调试等一系列操作，最终由验证器（verifier）自动判定任务是否完成。

> 👉 Terminal-Bench 2.1 与 2.0 的用例数量和用例名称完全一致，仅部分 case 的数据集内容与镜像内容有所差别，可以理解为 2.0 的用例与镜像修复版。

## 1. 功能概览

当前在 `ais_bench` 已接入以下 Terminal-Bench 2.1 能力：

- 数据集：`full`（官方全量）、`mini`（小规模采样）
- 任务：基于 Harbor 框架的通用 Agent 测评链路（`--mode agent`），Agent 在容器环境内自主执行终端任务，由 verifier 自动判定得分
- 结果汇总：输出 `avg_score`、`reward_distribution`、`exception_distribution`、`pass@k` 等关键指标

> 💡 Terminal-Bench 2.1 复用 AISBench 基于 Harbor 框架的通用 Agent 测评链路。环境准备（agent runtime 容器、Agent 离线依赖包）、参数说明等通用内容见 [基于Harbor框架的Agent 测评](../../base_tutorials/scenes_intro/agent_benchmark.md)。

## 2. 前置依赖

运行前请确保以下依赖可用：

1) Docker 可用（版本 >= 20.10.0，docker compose 版本 >= 2.0.0）

```bash
docker --version
docker ps
```

2) 在 Python 3.12 环境中安装 AISBench 与 Agent 独立依赖集（含 Harbor）

```bash
pip install -r requirements/agent.txt
```

3) 被测模型服务：遵循 OpenAI chat/completions API 规范且支持 tool call。

4) ⚠️ 执行过程中 Agent 需要访问外网，无法访问外网的环境需要配置代理环境变量。

## 3. 数据集与镜像准备

### 3.1 数据集

| 数据集 | 获取链接 | 说明 |
| ------ | -------- | ---- |
| Terminal-Bench 2.1（全量） | https://github.com/AISBench/terminal-bench-2-1 | AISBench 修改版：未修改官方原始镜像内容，仅修改镜像 tag 名称便于和 2.0 区分，数据集 `task.toml` 中镜像名称已同步修改 |
| Terminal-Bench 2.1 mini | https://modelers.cn/datasets/AISBench/terminal-bench-2-1-mini | 基于 2.1 的小规模采样数据集，适合先跑通流程/快速迭代 |

### 3.2 镜像

每个 case 均有对应的 Docker 镜像，镜像名称在数据集中定义。x86_64 服务器网络条件良好时执行过程中会自动拉取并构建，但过程往往比较漫长，建议提前加载打包镜像：

| 镜像打包资源 | 获取链接 | cpu架构 | 基础os |
| -------- | -------- | ------- | ------ |
| `terminal-bench-2.1-images-x86_64.tar` | https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2.1-images-x86_64.tar | x86_64 | ubuntu:24.04, debian:12, debian:13 |
| `terminal-bench-2.1-images-aarch64.tar` | https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2.1-images-aarch64.tar | aarch64 | ubuntu:24.04, debian:12, debian:13 |

```bash
docker load -i terminal-bench-2.1-images-x86_64.tar
```

> ⚠️ 镜像加载位置：源码安装方式在物理机上执行 `docker load`；模式 A（真 docker in docker）在 AISBench 容器内执行；模式 B（Socket 代理）在物理机上执行。详见 [Harbor Terminal-Bench](harbor_bench.md)。

## 4. 最小配置（先跑通再调优）

推荐从 `ais_bench/configs/agent_example/harbor_agent_task.py` 开始，仅改两处：

- `models[0]`：Agent 与模型服务相关参数（`agent_name`、`model_names`、`api_base`、`api_key`）
- `datasets[0]`：任务参数中的 `path`，改为 Terminal-Bench 2.1 数据集的本地路径

示例（terminus-2 Agent + 本地 vLLM 场景）：

```python
models = [
    dict(
        abbr="terminus-2",
        agent_name="terminus-2",            # -a/--agent: harbor AgentName 或 module.path:ClassName
        model_names=["hosted_vllm/qwen3"],  # --model: 模型名称
        api_base="http://0.0.0.0:8080/v1",  # --api-base: 模型服务 base url（统一语义）
        api_key="EMPTY",                    # --agent-api-key: 服务密钥（本地服务可用 EMPTY）
        llm_kwargs={"max_tokens": 4096},    # LLM 调用参数，合并进 agent kwargs
        model_info={                        # 模型 token 限制与成本信息
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        # deps_path="/path/to/terminus-2-offline-pack/",  # （可选，推荐）--agent-deps: 离线 agent 依赖包路径
    )
]

datasets = [
    dict(
        abbr="harbor_terminal-bench-2.1",
        args=dict(
            path="/path/to/terminal-bench-2.1/",  # -p/--agent-dataset-path: 本地数据集路径
            n_attempts=1,                         # -k/--n-attempts: 每个 trial 的尝试次数
            n_concurrent_trials=5,                # -n/--n-concurrent: 并发运行的 trial 数
            environment_type="docker",            # -e/--environment: 环境类型
            n_tasks=None,                         # --n-tasks: 最大任务数量，None 默认跑全部
        ),
    )
]
```

> 💡 上述示例配置文件即为 [自定义配置文件方式](../../advanced_tutorials/run_custom_config.md) 的具体应用。配置文件本质上是 Python 脚本，支持循环、条件判断、列表推导等所有 Python 语法。你可以参考这些示例文件自行编写满足特定需求的配置文件。详见 [自定义配置文件运行AISBench](../../advanced_tutorials/run_custom_config.md)。

### 首跑建议

- 数据集先用 `mini`
- `n_attempts=1`，`n_concurrent_trials=5`
- 通过 `n_tasks` 限制任务数量（如 `n_tasks=10`），快速跑通流程

## 5. 运行命令

在 AISBench 工具根目录执行：

```bash
ais_bench ais_bench/configs/agent_example/harbor_agent_task.py --mode agent
```

也可以不改配置文件，直接用命令行覆盖关键参数（命令行显式指定的参数优先级高于配置文件）：

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

### 断点续跑

中断后（如按下 `Ctrl+C`）再次执行相同命令即可自动续测（可加 `--reuse <时间戳>`，`<时间戳>` 为上次执行任务的输出目录名）：

```bash
ais_bench ais_bench/configs/agent_example/harbor_agent_task.py --mode agent --reuse 20260530_012601
```

Harbor 会自动检测 `details/config.json` 是否存在，并跳过已完成的 trial。

### 单条 case 多次执行（pass@k）

修改 `n_attempts` 参数可以多次执行同一 case，执行后将显示 `pass@k` 指标：

```python
datasets = [
    dict(
        abbr="harbor_terminal-bench-2.1",
        args=dict(
            path="/path/to/terminal-bench-2.1/",
            n_attempts=5,           # 每个 trial 尝试 5 次
            n_concurrent_trials=5,
        ),
    )
]
```

## 6. 输出结果怎么看

默认输出目录为 `outputs/default/<时间戳>/`，重点关注：

- 汇总表（`summary/summary_*.csv|md|txt`）：每行对应一个（模型 × 数据集）任务，包含 `avg_score`、`correct`、`wrong`、`exception` 等列
- 任务级结果：`results/{模型}/{数据集}/{数据集}.json`
- Harbor 落盘明细：`results/{模型}/{数据集}/details/`
  - `result.json`：任务级汇总（`n_total_trials`、`trial_results`、`exception_stats` 等）
  - `trial_*/verifier/reward.json`：单 case 得分
  - `trial_*/verifier/ctrf.json`：逐测试用例 pass/fail/skip 与失败原因，用于定位 case 具体失败点
  - `trial_*/agent/trajectory.json`：Agent 运行轨迹

关键指标：

- `avg_score`：所有任务的平均得分（reward，`1.0` 为通过）
- `reward_distribution`：奖励分布
- `exception_distribution`：异常类型分布（如 `AgentTimeoutError`、`AgentSetupTimeoutError`）
- `pass@k`：k 次执行中至少成功一次的概率（需 `n_attempts>1`）

## 7. 常见问题与排障

### 1) Agent 无法访问外网

- 现象：case 执行失败或 agent 依赖安装失败
- 原因：Terminal-Bench 2.1 执行过程中 Agent 需要访问外网
- 处理：配置代理环境变量（如 `--ae HTTPS_PROXY=http://proxy:port`），或参考 [基于Harbor框架的Agent 测评](../../base_tutorials/scenes_intro/agent_benchmark.md) 中的离线化方案

### 2) 镜像拉取/构建失败

- 现象：环境构建阶段报错
- 原因：未提前加载打包镜像，且网络无法访问镜像仓库
- 处理：提前执行 `docker load` 加载打包镜像，并确认加载位置（物理机/容器内）与 AISBench 容器启动方式一致

### 3) `datasets` 库版本冲突

- 现象：安装 Harbor 后报 `datasets` 依赖冲突
- 原因：安装 harbor 会将 `datasets` 库升级到 4.0.0 以上版本
- 处理：对执行 terminal-bench 相关数据集没有影响；如需测试其他数据集，需要降低 `datasets` 库版本

### 4) Agent 超时较多

- 现象：`exception_distribution` 中 `AgentTimeoutError` / `AgentSetupTimeoutError` 占比高
- 原因：任务超时时间不足，或模型服务吞吐不够
- 处理：通过 `timeout_multiplier` 调大超时倍数，或降低并发数（`n_concurrent_trials`）

## 8. 进阶建议（可选）

- 初次调试优先 `mini` 数据集，确认流程稳定后再切全量
- 关注 `exception_distribution` 与 `ctrf.json`，异常 case 通常比低分 case 更能定位环境问题
- 需要 pass@k 指标时设置 `n_attempts=k`，注意评测成本会相应增加约 k 倍
- 2.0 与 2.1 得分不宜直接对比：两者部分 case 的数据集内容与镜像内容有差别，横向对比时请使用同一版本
