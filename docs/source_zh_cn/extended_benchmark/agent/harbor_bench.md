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

### 2. 安装 AISBench 测评工具 & Harbor 依赖
#### 2.1 源码安装
> ⚠️环境限制： 确保环境docker 版本 >= 20.10.0，docker compose 版本 >= 2.0.0（docker compose可能需要额外安装）。同时需要准备一个python 3.12的运行环境
1. 在python 3.12的运行环境内，参考 [AISBench 安装文档](../../get_started/install.md) 安装 AISBench 测评工具。
2. python 3.12的运行环境内安装 Harbor：
   ```bash
   pip install harbor==0.6.1
   ```
3. 编辑harbor中的docker compose配置文件
首先执行：
```bash
pip3 show harbor | grep Loca
```
找到site-package路径，在这个路径下找到`harbor/environments/docker/docker-compose-base.yaml`, 例如

```
/usr/local/lib/python3.12/dist-packages/harbor/environments/docker/docker-compose-base.yaml
```

在这个文件中配置：

```yaml
services:
  main:
    network_mode: host # 共享主机网络，必须配置
    environment: # 在容器中普遍生效的环境变量
      http_proxy: XXXXXXXX # 如果执行环境无法访问互联网，需要配置代理环境变量
      https_proxy: XXXXXXXX # 如果执行环境无法访问互联网，需要配置代理环境变量
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

> ⚠️注意：安装harbor会将datasets库的版本升级到4.0.0以上的版本，这会导致安装后报datasets库的依赖冲突，对于执行harbor测试terminal-bench相关数据集没有影响，但是如果你需要测试其他数据集，需要降低datasets库的版本。

#### 2.2 在docker容器中安装
1. 参考[镜像概览](https://github.com/AISBench/benchmark/blob/master/docker/OVERVIEW.zh.md)的“运行 Agent / 沙箱类测评（在容器内使用 Docker）”章节启动基于**python3.12及以上版本镜像（2026.7.1之后发布的镜像才支持）**的容器。
2. 在容器内执行以下命令安装 Harbor：
   ```bash
   pip install harbor==0.6.1 --break-system-packages
   ```
3. 编辑harbor中的docker compose配置文件`/usr/local/lib/python3.12/dist-packages/harbor/environments/docker/docker-compose-base.yaml`
```yaml
services:
  main:
    network_mode: host # 共享主机网络，必须配置
    security_opt: # 模式 B 启动的容器需要配置
      - seccomp=unconfined
    environment: # 在容器中普遍生效的环境变量
      http_proxy: XXXXXXXX # 如果执行环境无法访问互联网，需要配置代理环境变量
      https_proxy: XXXXXXXX # 如果执行环境无法访问互联网，需要配置代理环境变量
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
> ⚠️注意：安装harbor会将datasets库的版本升级到4.0.0以上的版本，这会导致安装后报datasets库的依赖冲突，对于执行harbor测试terminal-bench相关数据集没有影响，但是如果你需要测试其他数据集，需要降低datasets库的版本。


### 3. 准备AISBench修改过的Terminal-Bench数据集和对应镜像
#### 3.1 terminal-bench 2
AISBench修改的数据集获取链接：https://github.com/AISBench/terminal-bench-2
> 👉注意: AISBench没有改用例内容，只是将所有环境的准备（包括terminus 2这个agent的所有依赖以及验证资源的）全部集中到Dockerfile中，避免反复执行还需要反复构建环境和安装依赖

Terminal-Bench-2 预制打包镜像信息：
| 镜像名称 | 获取链接 |cpu架构| 打包压缩包大小 |
| -------- | -------- | ------- |-------- |
|`terminal-bench-2-prepared-images_aarch64.tar`| https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-prepared-images_aarch64.tar | aarch64 | 48.50 GB |
|`terminal-bench-2-prepared-images_x86_64.tar`| https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-prepared-images_x86_64.tar | x86_64 | 71.43GB |

> 🌟提示：如果不想准备所有case的镜像，可以从[terminal-bench-2-offline-mini](https://modelers.cn/datasets/AISBench/terminal-bench-2-offline-mini)获取基于terminal-bench-2.0小规模采样的数据集及对应打包镜像

#### 3.2 terminal-bench 2.1
> terminal-bench 2.1简单来说就是terminal-bench-2.0的一个bugfix版本，用例数量和用例名称是完全一致的，只有部分case的数据集内容和镜像内容有所差别。

AISBench修改的数据集获取链接：https://github.com/AISBench/terminal-bench-2.1

> 👉注意：AISBench没有修改terminal-bench-2.1的官方原始镜像的内容，仅修改了镜像的tag名称便于和terminal-bench-2.0进行区分，数据集中的task.toml中镜像名称也做了同步修改。

| 镜像名称 | 获取链接 |cpu架构| 打包压缩包大小 |
| -------- | -------- | ------- |-------- |
|`terminal-bench-2.1-images-aarch64.tar`| 暂不支持 | aarch64 | NA |
|`terminal-bench-2.1-images-x86_64.tar`| https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2.1-images-x86_64.tar | x86_64 | 38.62GB |


> ⚠️注意：
> 如果通过源码安装 AISBench 测评工具 & Harbor 依赖这种方式安装依赖的情况下，部署Terminal-Bench-2/2.1的镜像需要在**物理机**上执行`docker load -i xxxxxxx.tar`
> 如果通过模式 A（真 docker in docker）启动AISBench容器，部署Terminal-Bench-2/2.1的镜像需要在**容器内**上执行`docker load -i xxxxxxx.tar`
> 如果提供给模式 B（Socket 代理）启动AISBench容器，部署Terminal-Bench-2/2.1的镜像需要在**物理机**上执行`docker load -i xxxxxxx.tar`

### 4. 配置 Harbor 任务的自定义配置文件

在 AISBench 工具根目录下修改 `ais_bench/configs/agent_example/harbor_terminal_bench_2_task.py`：

> 💡 上述 `harbor_terminal_bench_2_task.py` 即为 [自定义配置文件方式](../../advanced_tutorials/run_custom_config.md) 的具体应用。配置文件本质上是 Python 脚本，支持循环、条件判断、列表推导等所有 Python 语法。你可以参考此示例文件自行编写满足特定需求的配置文件。详见 [自定义配置文件运行AISBench](../../advanced_tutorials/run_custom_config.md)。

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
                path="/path/to/terminal-bench-2/",  # -p/--path: 本地数据集路径，terminal-bench-2或者terminal-bench 2.1数据集的路径
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