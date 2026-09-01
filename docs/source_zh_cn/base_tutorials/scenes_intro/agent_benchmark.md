# 基于Harbor框架的Agent 测评

AISBench 原生集成了 [Harbor](https://github.com/harbor-framework/harbor) 作为 Agent 测评引擎，通过 `--mode agent` 一条命令即可拉起 Harbor 评测、实时查看每个 case 状态，并输出单表 + CSV 汇总结果。该链路完全独立于 AISBench 原生的推理/精度链路，**不改变 AISBench 其他任何功能**。
> 👉注意：AISBench对harbor做了一些离线化部署的改造，并会持续接入新的agent，实际依赖的harbor为[fork主仓改造后的Harbor](https://github.com/AISBench/harbor)

与既有 [Harbor Terminal-Bench](../../extended_benchmark/agent/harbor_bench.md) 的接入不同，本章为通用化 Agent 测评，具备以下能力：

- **支持 Harbor 定义的全量 Agent**：内置 AgentName（terminus-2、claude-code、openhands、aider、codex 等）以及自定义 `module.path:ClassName` Agent。
- **统一参数适配**：不同 Agent 对同一含义参数（模型服务 base url / API key / LLM 调用参数 / 模型信息）的传入方式不同（有的走 kwargs、有的走环境变量），由 `AgentParamAdapter` 自动转换，用户只使用一套统一语义参数。
- **支持 Harbor 全量数据集来源**：本地路径（含单一 task 目录）、registry `name@version`、package `org/name@ref`。
- **实时监控 HTTP 服务**：标准库实现，无需额外依赖，外部可实时获取各 Harbor 任务的执行信息。
- **独立依赖集**：Agent 测评只需安装 `requirements/agent.txt`，不依赖 AISBench 原生繁重依赖。

## 安装

### 前置约束

- 一个遵循 **OpenAI chat/completions API** 规范、且支持 **tool call** 的被测推理服务（本地或云端均可）。
- Python 3.12 运行环境；执行 Harbor 所需的 Docker / 环境按 Harbor 要求准备。

### 安装 Agent 独立依赖集

```bash
pip install -r requirements/agent.txt
```

> ⚠️ **安装过程中的无关紧要报错说明**：执行 Agent 依赖安装（尤其是从源码以可编辑方式安装 Harbor 及其传递依赖）时，`pip` 可能会输出一些**不影响 Agent 测评使用**的报错或告警，主要包括：
> - 依赖版本冲突告警（例如 Harbor 会把 `datasets` 库升级到 4.0.0+，导致该库与其它依赖出现版本冲突告警）；
> - 个别包编译/构建的 warning，或依赖解析时的 `yanked` / `deprecated` 提示等。
> 这些告警只要**未导致安装失败（pip 报 `error` 并中断）**，即可直接忽略，继续安装后的 Harbor Agent 测评即可正常使用。若确需判断是否安装成功，可在安装后执行 `pip show harbor` 确认 Harbor 已正确就位。


## 资源准备
### harbor格式数据集准备
AISBench 理论上支持全量Harbor适配的数据集，具体支持的数据集参考[Harbor 数据集适配器列表](https://github.com/AISBench/harbor/tree/main/adapters/datasets)
。这些数据集需要参考harbor的文档自行构建。

🔍**AISBench直接提供了如下数据集资源**：

|数据集名称|全量数据集资源|小规模采样数据集资源| 备注 |
| ----- | ------ | ------ | ----- |
| SWEBench Verified |https://aisbench.obs.cn-north-4.myhuaweicloud.com/datasets/harbor_adapt_datasets/swebench-verified-offline.zip | https://modelers.cn/datasets/AISBench/SWE-Bench_Verified_mini | 小规模采样数据集资源中 `harbor_adapt`开头的文件夹是Harbor格式的数据集 |
| SWEBench Multilingual |https://aisbench.obs.cn-north-4.myhuaweicloud.com/datasets/harbor_adapt_datasets/swebench-multilingual-offline.zip | https://modelers.cn/datasets/AISBench/SWE-Bench_Multilingual_mini | 小规模采样数据集资源中 `harbor_adapt`开头的文件夹是Harbor格式的数据集  |
| SWEBench Pro | https://aisbench.obs.cn-north-4.myhuaweicloud.com/datasets/harbor_adapt_datasets/swebench-pro-offline.zip | https://modelers.cn/datasets/AISBench/SWE-Bench_Pro_mini | 小规模采样数据集资源中 `harbor_adapt`开头的文件夹是Harbor格式的数据集 |
| terminal-bench 2.0 | https://github.com/AISBench/terminal-bench-2 | https://modelers.cn/datasets/AISBench/terminal-bench-2-offline-mini | ⚠️执行过程中agent需要访问外网 |
| terminal-bench 2.1 | https://github.com/AISBench/terminal-bench-2-1 | https://modelers.cn/datasets/AISBench/terminal-bench-2-1-mini | ⚠️执行过程中需要agent访问外网 |

### 数据集对应镜像准备
agent数据集的测评每一个case都有对应的镜像，这些镜像名称在数据集中定义，如果在x86_64服务器上，网络条件良好且能够访问外网，执行过程中会自动拉取并构建对应的镜像。但是这个过程往往比较漫长。

🔍**AISBench直接提供了如下镜像打包资源**：

以下打包镜像资源可通过`docker load -i <镜像打包资源>`命令加载到测试环境中。

|数据集名称|全量数据集镜像打包资源|小规模采样数据集镜像打包资源 | 基础os | 备注 |
| ----- | ------ | ------ | ------ | ----- |
| SWEBench Verified | x86_64: https://aisbench.obs.cn-north-4.myhuaweicloud.com/datasets/SWEBenchData/verified.tar | NA | ubuntu:22.04 | 不支持aarch64 |
| SWEBench Multilingual | x86_64: https://aisbench.obs.cn-north-4.myhuaweicloud.com/datasets/SWEBenchData/multilingual.tar | NA | debian:12 | 不支持aarch64 |
| SWEBench Pro | NA | x86_64: https://modelers.cn/datasets/AISBench/SWE-Bench_Pro_mini  | ubuntu:24.04 | 不支持aarch64 |
| terminal-bench 2.0 | x86_64: <br> https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-prepared-images_x86_64.tar <br> aarch64: <br>https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-prepared-images_aarch64.tar | x86_64: <br> https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-offline-prepared-images-selected-0.10_x86_64.tar <br>https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-offline-prepared-images-selected-0.14_x86_64.tar <br>https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-offline-prepared-images-selected-0.20_x86_64.tar <br> aarch64: https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-offline-prepared-images-selected-0.10_aarch64.tar <br>https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-offline-prepared-images-selected-0.14_aarch64.tar <br>https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2-offline-prepared-images-selected-0.20_aarch64.tar | ubuntu:24.04, debian:11, debian:12, debian:13 | NA |
|terminal-bench 2.1| x86_64: <br>https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2.1-images-x86_64.tar <br>aarch64:<br>https://aisbench.obs.cn-north-4.myhuaweicloud.com/terminal-bench-2-images/terminal-bench-2.1-images-aarch64.tar | NA | ubuntu:24.04, debian:11, debian:12, debian:13 |  NA |

### Agent 支持列表

AISBench 支持 Harbor 定义的全量 Agent（`-a/--agent` 直接传名称），也支持通过 `--agent-import-path` 指定自定义 `module.path:ClassName` Agent。以下为 Harbor `AgentName` 内置全部 Agent：

| Agent（AgentName） | Agent（AgentName） |
| --- | --- |
| `oracle`、`nop`、`acp` | `claude-code`、`cline-cli`、`cortex-code` |
| `terminus`、`terminus-1`、`terminus-2` | `aider`、`codex`、`cursor-cli` |
| `gemini-cli`、`antigravity-cli`、`antigravity-sdk` | `rovodev-cli`、`goose`、`grok-build` |
| `hermes`、`mini-swe-agent`、`nemo-agent` | `swe-agent`、`opencode`、`openclaw` |
| `openhands`、`openhands-sdk`、`kimi-code` | `kimi-cli`、`langgraph`、`deerflow` |
| `mimo`、`pi`、`qwen-coder` | `copilot-cli`、`devin`、`trae-agent` |
| `computer-1`、`eve`、`fx` | `dsh`、`dspy-rlm`、`vibe` |

> 💡 不同 Agent 对同一含义参数（模型服务 base url / API key 等）的传入方式不同，AISBench 的 `AgentParamAdapter` 会自动转换：走环境变量的（如 `claude-code`→`ANTHROPIC_*`、`dsh`→`DSH_*`、`openhands`→`LLM_*`/`OPENAI_*`）与走构造函数 kwarg 的（如 `terminus-2`→`api_base`）均使用同一套统一语义参数（`--api-base` / `--agent-api-key`）即可，无需区分。

在测评任务执行过程中，harbor会在每个case的容器中安装对应agent的依赖，包括agent的代码、依赖库、配置文件等，如果你的环境没有网络条件：

🔍**AISBench直接提供了如下agent依赖打包资源**：

> ⚠️注意：依据数据集镜像的基础os，选择对应的agent打包资源。

|Agent| 依赖打包资源| 备注 |
| ----- | ------ | ------ |
|mini-swe-agent|x86_64: <br>ubuntu:22.04 https://aisbench.obs.cn-north-4.myhuaweicloud.com/others/agent_offline_pack/mini-swe-agent/mini-swe-agent-ubuntu-22.04-x86_64.tar.gz<br>ubuntu:24.04 https://aisbench.obs.cn-north-4.myhuaweicloud.com/others/agent_offline_pack/mini-swe-agent/mini-swe-agent-ubuntu-24.04-x86_64.tar.gz<br>debian:12 https://aisbench.obs.cn-north-4.myhuaweicloud.com/others/agent_offline_pack/mini-swe-agent/mini-swe-agent-debian-12-x86_64.tar.gz||
|terminus-2|x86_64: <br>ubuntu22.04 https://aisbench.obs.cn-north-4.myhuaweicloud.com/others/agent_offline_pack/terminus-2/terminus-2-ubuntu-22.04-x86_64.tar.gz<br>ubuntu24.04 https://aisbench.obs.cn-north-4.myhuaweicloud.com/others/agent_offline_pack/terminus-2/terminus-2-ubuntu-24.04-x86_64.tar.gz<br>debian:11 https://aisbench.obs.cn-north-4.myhuaweicloud.com/others/agent_offline_pack/terminus-2/terminus-2-debian-11-x86_64.tar.gz<br>debian:12 https://aisbench.obs.cn-north-4.myhuaweicloud.com/others/agent_offline_pack/terminus-2/terminus-2-debian-12-x86_64.tar.gz<br>debian:13 https://aisbench.obs.cn-north-4.myhuaweicloud.com/others/agent_offline_pack/terminus-2/terminus-2-debian-13-x86_64.tar.gz||
|dsh|x86_64: <br>https://aisbench.obs.cn-north-4.myhuaweicloud.com/others/agent_offline_pack/dsh/dsh-ubuntu-22.04-x86_64.tar.gz||


## 快速入门（两种方式任选其一）

| ⭐ 推荐：使用命令行参数 | 备选：使用自定义配置文件 |
| :--- | :--- |
| 无需新建文件，一条命令把 Agent、模型服务、数据集及运行参数全部配好 | 集中管理所有参数，一次编写多次复用 |
| 依赖统一语义参数由适配器自动转换，开箱即用 | 支持 Python 全部语法，灵活扩展 |

::::{tab-set}
:::{tab-item} ⭐ 推荐：使用命令行参数

命令行方式通过一条命令配置所有参数。除常规模型服务参数外，Agent 测评新引入的参数（`--mode agent`、`-a/--agent`、`--api-base`、`--agent-api-key`、`-p/--agent-dataset-path`、`-d/--dataset`、`-n/--n-concurrent`、`-k/--n-attempts`、`-e/--environment`、`--monitor-port` 等）均需配合 `--mode agent` / `--mode agent_viz` 使用。完整参数见 📚 [用户配置参数 - Agent 测评参数](../all_params/cli_args.md#agent-测评参数)。

以本地 terminal-bench-2 数据集 + terminus-2 Agent 为例：

```bash
ais_bench --mode agent \
    -a terminus-2 \                        # Agent 名称（或自定义 import path）
    --model hosted_vllm/qwen3 \             # 模型名称（可多次）
    --api-base http://0.0.0.0:8080/v1 \     # 模型服务 base url（统一语义）
    --agent-api-key sk-xxx \                # 模型服务 API key（统一语义）
    -p /path/to/terminal-bench-2 \          # 本地数据集路径
    # --agent-deps /path/to/terminus-2-offline-pack/ \ # （可选，推荐）Agent 依赖打包资源路径, 里面放置各种os的tar.gz文件
    -n 5 \                                  # 并发 trial 数
    -k 1 \                                  # 每个 trial 尝试次数
    -e docker \                             # 环境类型
    --monitor-port 8788                     # 实时监控 HTTP 服务端口
```

如需为 Agent 追加原始参数或环境变量（优先级最高），可用 `--ak key=value` / `--ae KEY=VALUE`：

```bash
ais_bench --mode agent -a terminus-2 --model hosted_vllm/qwen3 \
    --api-base http://0.0.0.0:8080/v1 \
    -p /path/to/terminal-bench-2 \
    --ak max_tokens=4096 \
    --ae HTTPS_PROXY=http://proxy:port
```

若本地数据集是**单一 task 目录**（而非数据集目录），同样用 `-p` 指定即可；远程数据集用 `-d name@version` / `-d org/name@ref`。

:::
:::{tab-item} 备选：使用自定义配置文件

自定义配置文件把「模型服务与 Agent 本身参数」放在 `models`、把「Agent 测评任务参数」放在 `datasets`，一次编写多次复用。可参考 `ais_bench/configs/agent_example/harbor_agent_task.py` / `harbor_terminal_bench_2_task.py`，以下为完整示例：

```python
from mmengine.config import read_base
from ais_bench.benchmark.tasks.custom_tasks.harbor_agent_task import HarborAgentTask
from ais_bench.benchmark.runners.harbor_runner import HarborRunner
from ais_bench.benchmark.tasks.base import EmptyTask
from ais_bench.benchmark.summarizers.harbor import HarborSummarizer

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer

# models：模型服务与 agent 本身参数；统一语义参数由 AgentParamAdapter 自动转换
models = [
    dict(
        abbr="terminus-2",
        agent_name="terminus-2",            # -a/--agent: harbor AgentName 或 module.path:ClassName
        model_names=["hosted_vllm/qwen3"],  # --model: 模型名称（可多值）
        api_base="http://0.0.0.0:8080/v1",  # --api-base: 模型服务 base url（统一语义）
        agent_api_key="sk-xxx",             # --agent-api-key: 模型服务 API key（统一语义）
        llm_kwargs={"max_tokens": 4096},    # LLM 调用参数，合并进 agent kwargs
        model_info={                        # 模型 token 限制与成本信息
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
        },
        # deps_path=None,                  # （可选，推荐）--agent-deps: 离线 agent 依赖包路径, 里面放置各种os的tar.gz文件
    )
]

# datasets：Agent 测评任务参数
datasets = [
    dict(
        abbr="harbor_terminal-bench-2",
        args=dict(
            path="/path/to/terminal-bench-2/",  # -p/--agent-dataset-path: 本地数据集路径
            # dataset_name_version=None,        # -d/--dataset: 远程数据集 name@version / org/name@ref
            n_concurrent_trials=5,              # -n/--n-concurrent: 并发 trial 数
            n_attempts=1,                       # -k/--n-attempts: 每个 trial 尝试次数
            environment_type="docker",          # -e/--environment: 环境类型
            max_retries=0,                      # --max-retries: 最大重试次数
            yes=True,                           # -y/--yes: 自动确认环境变量提示
        ),
    )
]

# agent 模式无需原生 inference 阶段
infer = dict(runner=dict(task=dict(type=EmptyTask)))
eval = dict(
    runner=dict(
        type=HarborRunner,
        monitor_port=0,       # --monitor-port: 监控 HTTP 服务端口（0=关闭）
        task=dict(type=HarborAgentTask),
    ),
)
summarizer = dict(attr="accuracy", type=HarborSummarizer)
```

修改好配置文件后，执行命令：

```bash
ais_bench ais_bench/configs/agent_example/harbor_agent_task.py --mode agent
```

> 💡 命令行显式指定的参数会覆盖自定义配置文件中对应字段。各参数的含义与可选值详见文末 📚 [附录：自定义配置文件参数说明](#附录自定义配置文件参数说明)。

:::
::::

### 实时监控服务

通过 `--monitor-port <port>` 开启只读 HTTP 服务（标准库实现，默认 0 = 关闭）：

```bash
ais_bench ... --mode agent --monitor-port 8788
```

常用端点（`{模型}`、`{数据集}`、`{case}` 参考实际任务）：

| 端点 | 说明 |
| --- | --- |
| `GET /api/health` | 存活探测 |
| `GET /api/tasks` | 全部任务级快照 |
| `GET /api/tasks/{模型}/{数据集}/` | 任务总览 = `result.json` 原文 |
| `GET /api/tasks/{模型}/{数据集}/{case}` | 单 case 原始 `result.json`（`trial_00000` / 序号 / harbor 任务名） |
| `GET /api/tasks/{模型}/{数据集}/cases` | 每 case 派生状态/成败原因明细 |
| `GET /api/jobs` | 各 job 进度（聚合计数 + case 状态统计） |

### 中断续测 & 异常用例自动重试

Harbor 会自动检测 `details/config.json` 是否存在并跳过已完成 trial，再次执行相同命令（可加 `--reuse <时间戳>`）即可续测：

```bash
ais_bench ais_bench/configs/agent_example/harbor_agent_task.py --mode agent --reuse 20260530_012601
```

`--purge-exception-cases` 开关**仅在 `--reuse` 生效时启用**：执行前从每个任务 `result.json` 的 `exception_stats` 提取异常 case 名，删除同目录下的 case 目录，使 Harbor 重建 job 时自动重跑这些异常退出的 case：

```bash
# 先正常跑出部分异常 case，再重跑
ais_bench ... --mode agent --reuse 20260530_012601 --purge-exception-cases
```

已有结果后，可用 `--mode agent_viz` 只做汇总，不启动任何 Harbor job：

```bash
ais_bench ... --mode agent_viz --reuse 20260530_012601
```

## 执行结果与落盘文件

### 执行结果含义

任务结束后 `HarborSummarizer` 会打印一张汇总表并落盘一个 CSV，每行对应一个（模型 × 数据集）任务，列即如下指标：

| 列 | 含义 |
| --- | --- |
| `agent` | 本次使用的 Agent 名称 |
| `model_name` | 模型名称（`model_names`，多个以 `,` 分隔） |
| `dataset` | 数据集任务简称（`dataset.abbr`） |
| `avg_score` | 该任务平均得分（读取聚合 `result.json` 的 `avg_score`） |
| `correct` | 得分 `>= 1.0` 的完成 trial 数（由 `reward_distribution` 统计） |
| `wrong` | 得分 `0 ~ 1.0`（不含 1.0）的完成 trial 数（由 `reward_distribution` 统计） |
| `exception` | 执行中出现异常的 trial 数（对应 `n_errors`） |

> 💡 更多原始指标（`total_count` / `n_errors` / `reward_distribution` / `exception_distribution` / `pass@k` 等）保留在逐任务落盘的 `results/{模型}/{数据集}.json` 与 `details/result.json` 中，如需查看可叠加 `--monitor-port` 端点或直接读取落盘文件。

### 落盘文件结构与含义

结果保存在 `outputs/default/{时间戳}/`（工作目录可通过 `--work-dir` 修改），外层目录与 AISBench 其它场景保持一致，核心是 `results/{模型}/{数据集}/details/` 下的 **Harbor 落盘结果**，它镜像了 Harbor job 的结果布局，逐 case 记录了执行与验证明细：

```bash
outputs/default/20260530_012601/
├── configs
│   └── 20260530_012601.py            # 合成后的完整配置（含 CLI/配置文件的全部参数）
├── logs/eval/{模型}/{数据集}.out      # 执行过程日志
├── results/{模型}/{数据集}/           # 测评结果（Harbor 落盘所在目录）
│   ├── details/                      # ▽ 该任务 Harbor job 的结果目录（核心）▽
│   │   ├── config.json               # 该任务的 job 配置原文（断点续测依据：存在即 resume）
│   │   ├── result.json               # 任务级汇总：n_total_trials/stats/trial_results/exception_stats
│   │   └── trial_00000/              # 每个 case 一个目录（按调度序编号）
│   │       ├── result.json           # 该 case 结果（status/reward/exception_info/timings/agent_info）
│   │       ├── exception.txt         # 该 case 的异常消息文本（异常退出时生成）
│   │       ├── trial.log             # 该 case 的 trial 运行日志
│   │       ├── config.json           # 该 case 的任务配置
│   │       ├── agent/                # Agent 执行产物
│   │       │   └── trajectory.json   # Agent 运行轨迹（是否存在表示有无轨迹）
│   │       └── verifier/             # 验证器产物
│   │           ├── reward.json       # 该 case 得分
│   │           ├── ctrf.json         # 逐测试用例 pass/fail/skip + 失败原因
│   │           ├── test-stdout.txt   # 验证执行 stdout 尾部
│   │           └── test-stderr.txt   # 验证执行 stderr 尾部
│   └── {数据集}.json                 # 该任务聚合结果（供 summarizer 汇总）
└── summary/
    ├── summary_20260530_012601.csv   # 汇总表（csv 格式）
    ├── summary_20260530_012601.md    # 汇总表（markdown 格式）
    └── summary_20260530_012601.txt   # 汇总表（文本格式）
```

各 Harbor 落盘文件的作用：

| 文件 | 作用 |
| --- | --- |
| `details/config.json` | job 配置原文；**是否存在决定了是否断点续测**（存在则 `_resume_job`，跳过已完成 trial） |
| `details/result.json` | 任务级汇总；`stats.evals[*].exception_stats` 记录异常 case 名，是 `--purge-exception-cases` 自动重试的**数据来源**，也可经监控服务 `GET /api/tasks/{模型}/{数据集}/` 直接读取 |
| `trial_*/result.json` | 单个 case 的结果（状态 / reward / exception_info / 耗时 / agent 信息） |
| `trial_*/exception.txt` | 异常 case 的异常消息文本，便于排查失败原因 |
| `trial_*/trial.log` | 单个 case 的运行日志 |
| `trial_*/agent/trajectory.json` | Agent 运行轨迹；存在即代表该 case 有轨迹可复现 |
| `trial_*/verifier/reward.json` | 该 case 的得分 |
| `trial_*/verifier/ctrf.json` | 逐测试用例 pass / fail / skip 与失败信息，用于定位 case 具体失败点 |
| `trial_*/verifier/test-stdout.txt` / `test-stderr.txt` | 验证执行的输出 / 错误尾部 |

> 💡 实时监控服务（`--monitor-port`）逐条读取上述文件（`result.json` / `ctrf.json` 等），并以 mtime 增量缓存保证高频刷新下开销低廉；这些文件也被用于在任务执行中实时计算 `correct / wrong / exception / avg_score` 展示在看板与 `/api/*` 端点上。

## 附录：自定义配置文件参数说明

### `models`（模型服务与 Agent 本身参数）

| 参数 | 对应 CLI | 说明 |
| --- | --- | --- |
| `abbr` | - | 模型简称（唯一标识，用于结果目录名） |
| `agent_name` | `-a/--agent` | Agent 名称（Harbor AgentName）或自定义 import path |
| `agent_import_path` | `--agent-import-path` | 自定义 Agent 的导入路径（`module.path:ClassName`） |
| `model_names` | `--model` | 模型名称列表，可多值 |
| `api_base` | `--api-base` | 模型服务 base url（统一语义，自动转换） |
| `agent_api_key` | `--agent-api-key` | 模型服务 API key（统一语义，自动转换） |
| `llm_kwargs` | - | LLM 调用参数（如 `max_tokens`），合并进 agent kwargs |
| `model_info` | - | 模型 token 限制与成本信息（`max_input_tokens` / `max_output_tokens` / `input_cost_per_token` / `output_cost_per_token`） |
| `agent_kwargs` | `--ak/--agent-kwarg` | 追加的 Agent 原始 kwargs（优先级最高） |
| `agent_env` | `--ae/--agent-env` | 追加的 Agent 环境变量（优先级最高） |
| `deps_path` | `--agent-deps` | 离线 Agent 依赖包路径 |
| `n_concurrent` | - | 每个 Agent 的并发上限 |
| `skills` / `mcp_servers` | - | Agent 技能目录 / MCP 服务器配置 |
| `resume_trajectory` / `load_trajectory` | - | 跨步恢复 agent 会话 / 预加载轨迹文件 |
| `extra_allowed_hosts` | - | 额外允许访问的 host/IP |
| `include_logs` / `exclude_logs` | - | 保留 / 排除的 agent 日志 glob |
| `override_timeout_sec` / `override_setup_timeout_sec` / `max_timeout_sec` | - | Agent 执行 / 环境搭建超时覆盖与上限 |

### `datasets[].args`（Agent 测评任务参数）

| 参数 | 对应 CLI | 说明 |
| --- | --- | --- |
| `path` | `-p/--agent-dataset-path` | 本地数据集路径（或单一 task 目录） |
| `dataset_name_version` | `-d/--dataset` | 远程数据集 `name@version` / `org/name@ref` |
| `registry_url` / `registry_path` | - | 数据集 registry 地址 / 路径 |
| `n_concurrent_trials` | `-n/--n-concurrent` | 并发运行 trial 数 |
| `n_attempts` | `-k/--n-attempts` | 每个 trial 尝试次数 |
| `debug` | `--debug` | 启用调试日志 |
| `quiet` | `-q/--quiet` | 抑制单个 trial 进度显示 |
| `timeout_multiplier` | `--timeout-multiplier` | 任务超时倍数 |
| `agent_timeout_multiplier` / `verifier_timeout_multiplier` / `agent_setup_timeout_multiplier` / `environment_build_timeout_multiplier` | - | 各环节超时倍数 |
| `max_retries` | `--max-retries` | 最大重试次数 |
| `retry_include_exceptions` / `retry_exclude_exceptions` | - | 重试包含 / 排除的异常类型集合 |
| `environment_type` | `-e/--environment` | 环境类型（`docker`、`daytona`、`e2b`、`modal` 等） |
| `environment_force_build` | `--force-build/--no-force-build` | 是否强制重建环境 |
| `environment_delete` | `--delete/--no-delete` | 完成后是否删除环境 |
| `environment_kwargs` | `--host-network` 等 | 环境附加参数（`--host-network` 写入 `{"host_network": True}`） |
| `environment_env` | - | 环境变量 |
| `disable_verification` | `--disable-verification` | 禁用 verifier |
| `verifier_env` / `verifier_import_path` / `verifier_kwargs` | - | verifier 环境变量 / 自定义导入路径 / 附加参数 |
| `task_names` | `--include-task-name` | 需要包含的任务名（支持 glob） |
| `exclude_task_names` | `--exclude-task-name` | 需要排除的任务名（支持 glob） |
| `n_tasks` | `--n-tasks` | 从数据集选取的最大任务数量 |
| `yes` | `-y/--yes` | 自动确认环境变量提示 |
| `env_file` | `--env-file` | `.env` 文件路径 |

### 其他：`eval.runner`

| 参数 | 对应 CLI | 说明 |
| --- | --- | --- |
| `monitor_port` | `--monitor-port` | Harbor 监控 HTTP 服务端口（0 = 关闭，默认 0） |

> 📚 更详细的 Harbor 环境准备、terminal-bench 2/2.1 数据集与镜像说明，参见 [Harbor Terminal-Bench](../../extended_benchmark/agent/harbor_bench.md)。全部 CLI 参数见 📚 [用户配置参数 - Agent 测评参数](../all_params/cli_args.md#agent-测评参数)。