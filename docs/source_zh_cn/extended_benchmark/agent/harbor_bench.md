# Harbor Terminal-Bench

## Harbor 简介

**Harbor** 是一个用于评估 AI Agent 的框架，支持运行多种 benchmark 任务，包括 Terminal-Bench-2 等。

基准官方仓库：[https://github.com/harbor-framework/harbor](https://github.com/harbor-framework/harbor)

### 一、核心定位与背景

- **核心功能**：支持多种 Agent（Terminus-2、Claude Code、OpenHands 等）的评测
- **核心创新**：
  - 支持多种环境（Docker、Daytona、E2B、Modal 等）
  - 支持并行执行和断点续测
  - 自动评估和结果分析
- **核心目标**：评测 Agent 的**任务完成、工具使用、策略遵守**综合能力

### 二、支持的功能

1. **多 Agent 支持**
   - 内置 Agent：terminus-2, claude-code, openhands, aider, codex 等
   - 自定义 Agent：通过 `--agent-import-path` 指定

2. **多环境支持**
   - Docker（本地）
   - Daytona（云端）
   - E2B（沙箱）
   - Modal（云端）

3. **数据集支持**
   - 本地路径：`-p /path/to/dataset`
   - 远程数据集：`-d dataset-name@version`

### 三、核心评测机制

- **自动化验证**：通过 verifier 自动评估结果
- **并行执行**：通过 `-n/--n-concurrent` 控制并发数
- **断点续测**：检测已有结果，自动跳过已完成任务
- **轨迹导出**：通过 `--export-traces` 导出轨迹

## AISBench 中快速上手基于Harbor的Terminal-Bench 2.0 测评

### 1. 准备推理服务

确保本地或云端部署了遵循 OpenAI chat/completions API 规范且支持 tool call 的被测推理服务。

### 2. 准备AISBench修改过的Terminal-Bench-2数据集和对应镜像
AISBench修改的数据集获取链接：https://github.com/AISBench/terminal-bench-2
> 👉注意: AISBench没有改用例内容，只是将所有环境的准备全部集中到Dockerfile中，避免反复执行还需要反复构建环境和安装依赖

Terminal-Bench-2 预制打包镜像信息：
| 镜像名称 | 获取链接 |cpu架构| 打包压缩包大小 |
| -------- | -------- | ------- |-------- |
|`terminal-bench-2-prepared-images_aarch64.tar`| https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-prepared-images_aarch64.tar | aarch64 | 48.50 GB |
|`terminal-bench-2-prepared-images_x86_64.tar`| https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-prepared-images_x86_64.tar | x86_64 | 71.43GB |

> 🌟提示：如果不想准备所有case的镜像，可以从[terminal-bench-2-offline-mini](https://modelers.cn/datasets/AISBench/terminal-bench-2-offline-mini)获取基于terminal-bench-2.0小规模采样的数据集及对应打包镜像

### 3. 安装 AISBench 测评工具 & Harbor 依赖
#### 3.1 源码安装
> ⚠️环境限制： 确保环境docker 版本 >= 20.10.0，docker compose 版本 >= 2.0.0（docker compose可能需要额外安装）。同时需要准备一个python 3.12的运行环境
1. 在python 3.12的运行环境内，参考 [AISBench 安装文档](../../get_started/install.md) 安装 AISBench 测评工具。
2. python 3.12的运行环境内安装 Harbor：
   ```bash
   pip install harbor==0.6.1
   ```
> ⚠️注意：安装harbor会将datasets库的版本升级到4.0.0以上的版本，这会导致安装后报datasets库的依赖冲突，对于执行harbor测试terminal-bench相关数据集没有影响，但是如果你需要测试其他数据集，需要降低datasets库的版本。

> ⚠️注意：源码安装方式下，[2. 准备数据集和对应镜像](##-2-准备aisbench修改过的terminal-bench-2数据集和对应镜像) 章节下载的 case 镜像 tar 需要在**物理机**上执行 `docker load -i xxxxxxx.tar` 加载到本地 docker daemon 后再跑测评。

#### 3.2 一键准备方案（推荐）

如果不想手动处理依赖冲突 / DinD 配置，推荐使用 **AISBench Agent Runtime 一键准备方案**。同一脚本同时覆盖**快速入门（在线）**与**离线场景（内网/隔离环境）**，通过 `--runtime-tar` / `--case-tar` / `--datasets` 自由组合，无需切换不同流程。

```bash
# 1. 物理机上一键起 runtime 容器（自动选 DinD/Socket 模式，自动挂载数据集，自动把 case 镜像 tar 拷进容器内部 docker load 完）
#    在线场景：省略 --runtime-tar，runtime 镜像自动从 ghcr.io 拉取
#    离线场景：通过 --runtime-tar 跳过外网拉取
curl -fsSL https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/ais_bench_agent_bootstrap.sh \
    | bash -s -- \
        --datasets /path/to/terminal-bench-2-offline-mini/terminal-bench-2-offline-selected_0.10/ \
        --runtime-tar /path/to/agent_runtime_image_v3.1-20260701-master-ubuntu24.04-py312-<arch>.tar.gz \
        --case-tar /path/to/terminal-bench-2-offline-prepared-images-selected-0.10.tar \
        --host-path /path/to/test_wkp/ \
        --container-name test_agent_run
# --datasets 指向的目录结构需与 terminal-bench-2-offline-mini 仓库的 terminal-bench-2-offline-selected_0.10/ 子目录结构一致
# --runtime-tar （可选）提前准备的测评镜像，不传则自动拉取最新
# --case-tar 指向的 tar 结构需与对应 agent 测评文档的 case 镜像 tar 结构一致（可多次传，也可传目录）
# --host-path 指向的目录需为空目录，容器内会自动创建同名目录挂载数据集和 case 镜像
# --container-name 指向的容器名需唯一，否则会覆盖旧容器

# 2. 进入容器（case 镜像已在内部，直接可用）
docker exec -it test_agent_run bash

# 3. （无需改 path）原生配置 path 自动从 AISBENCH_AGENT_DATASET_PATH 读
#    仅需 vim 改 model_names / api_base
vim ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py

# 4. 验证 runtime 就绪
ais_bench_agent_doctor.sh harbor

# 5. 跑测评
agent_env harbor
ais_bench ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py --debug
```

切到其它数据集（mini-0.14 / mini-0.20 / full）：销毁旧容器 → 重新 `bash ... --datasets <新路径> --case-tar <新tar>` 起容器。

`--runtime-tar` / `--case-tar` / `--datasets` 三者完全独立，可任意组合。三个都不会触发任何 `docker pull` 或 `curl` 到外网的操作；在快速入门（在线）场景中省略 `--runtime-tar`，脚本会自动从网络拉取 runtime 镜像。

`--case-tar` 在 A/B 两种模式下都生效：脚本会 `docker cp` 把 tar 拷进 runtime 容器，再在容器内 `docker load` 加载到该容器的 docker daemon。

该方案解决了以下痛点：
- **依赖冲突**：harbor==0.6.1 强制升级 datasets 到 4.0+ 会污染主环境，runtime 镜像用独立 venv 隔离
- **容器配置易错**：DinD 模式 A/B、`--cgroupns=host`、`daemon.json`、seccomp 自动处理
- **数据集版本频繁**：数据集与 case 镜像均不烤入 runtime 镜像，由用户在物理机准备后通过 `--datasets` 挂载 / `--case-tar` 加载，避免镜像频繁过期
- **case 镜像管理**：通过 `--case-tar` 在 bootstrap 时一次性加载到容器内，容器内无需手动 `docker pull` / `docker load`
- **环境无验证**：`doctor.sh` 在跑测评前验证 runtime 就绪，失败时给精确修复指引
- **离线部署**：支持 `--runtime-tar <PATH>` 跳过 runtime 镜像的网络获取；支持 `--case-tar <PATH>` 加载 case 镜像到容器内（可多次，可传目录）。内网隔离环境可全程零外网请求

方案原理与脚本实现见 [`docker/agent_runtime/`](https://github.com/AISBench/benchmark/tree/master/docker/agent_runtime/README.md)。

### 4. 配置 Harbor 任务的自定义配置文件

在 AISBench 工具根目录下修改 `ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py`：

```python
models = [
    dict(
        abbr="terminus-2",
        agent_name="terminus-2",  # -a/--agent: Agent名称 (terminus-2, claude-code, openhands等)
        model_names=["hosted_vllm/qwen3"],  # -m/--model: 模型名称, hosted_vllm/{模型名称}
        agent_kwargs={  # --ak/--agent-kwarg: Agent额外参数
            "api_base": "http://0.0.0.0:8080/v1",  # terminus-2需要api_base连接推理服务，例如填"http://0.0.0.0:8080/v1"会访问"http://0.0.0.0:8080/v1/chat/completions"
            "model_info": {  # 模型token限制和成本信息
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
            "llm_call_kwargs": { # LLM调用参数
                "max_tokens": 4096, # 最大输出token数
                # "temperature": 0.7,
                # "top_p": 0.9,
                # "top_k": 50,
            },
        },
        agent_env=None,  # --ae/--agent-env: 传递给agent的环境变量
    )
]
# ......
datasets = []
for task in sub_tasks:
    datasets.append(
        dict(
            abbr=f'harbor_{task}',
            args=dict(
                n_attempts=1,  # -k/--n-attempts: 每个trial的尝试次数
                timeout_multiplier=1.0,  # --timeout-multiplier: 超时倍数（所有超时乘以此系数）
                # ......
                n_concurrent_trials=5,  # -n/--n-concurrent: 并发运行的trial数量
                # ......
                path="/path/to/terminal-bench-2/",  # -p/--path: 本地数据集路径
                # ......
                n_tasks=None,  # --n-tasks: 最大任务数量, None默认跑全部，快速入门可以尝试设置几条快速跑通流程
                # ......
            ),
        )
    )

# ......
```

### 5. 执行 Harbor 任务

1. 在 AISBench 工具根目录下执行以下命令：
   ```bash
   ais_bench ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py --debug
   ```

> 这里推荐加`--debug`的原因是因为harbor执行过程中原生的日志看板更加清晰详细，可以精确到实时得分，但是这个实时刷新的看板的内容日志在非debug场景后台执行时无法落盘，只能在终端看到，所以推荐在debug场景下执行。

2. 执行过程看板示例

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

3. 任务执行完成后，会打印如下精度结果：

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

- `Avg Score`：所有任务的平均得分
- `n_errors`：执行过程中出现的异常数量
- `reward_distribution`：奖励分布
- `exception_distribution`：异常类型分布
- `pass@k`：k 次执行的成功率

4. 最终 `outputs/default/{时间戳}` 目录下结果文件的结构如下：

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

## 中断后继续执行测评

中断任务执行后（如按下 `Ctrl+C`），再次执行相同命令即可自动续测：

```bash
ais_bench ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py --debug --reuse 20260530_012601
```
其中`20260530_012601`为上次失败任务执行时的时间戳，需要根据实际情况替换。
Harbor 会自动检测 `details/config.json` 是否存在，并跳过已完成的 trial。


## 单条 case 多次执行（pass@k）

修改 `n_attempts` 参数可以多次执行同一 case：

```python
datasets.append(
    dict(
        abbr='harbor_terminal-bench-2',
        args=dict(
            path="/path/to/terminal-bench-2/",
            n_attempts=5,  # 每个trial尝试5次
            n_concurrent_trials=5,
        ),
    )
)
```

执行后将显示 `pass@k` 指标，表示 k 次执行中至少成功一次的概率。

## 任务配置（datasets 中）关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | str | - | 本地数据集路径（-p/--path） |
| `n_attempts` | int | 1 | 每个 trial 的尝试次数（-k/--n-attempts） |
| `n_concurrent_trials` | int | 5 | 并发 trial 数（-n/--n-concurrent） |
| `environment_type` | str | docker | 环境类型（-e/--env） |
| `environment_force_build` | bool | False | 是否强制重建环境 |
| `environment_delete` | bool | True | 完成后是否删除环境 |
| `timeout_multiplier` | float | 1.0 | 超时倍数 |
| `max_retries` | int | 0 | 最大重试次数 |
| `task_names` | list[str] | None | 包含的任务名（--include-task-name） |
| `exclude_task_names` | list[str] | None | 排除的任务名（--exclude-task-name） |
| `n_tasks` | int | None | 最大任务数量（--n-tasks） |

## Agent 配置（models 中）相关参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `abbr` | str | - | 模型简称 |
| `agent_name` | str | oracle | Agent 名称（-a/--agent） |
| `model_names` | list[str] | None | 模型名称（-m/--model） |
| `agent_kwargs` | dict | {} | Agent 额外参数（--ak/--agent-kwarg） |
| `agent_env` | dict | {} | Agent 环境变量（--ae/--agent-env） |