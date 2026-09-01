# 评测场景简介
### 精度测评
#### 服务化精度测评
- 功能描述：评估部署为服务形式的模型在特定数据集上的预测准确率，当前支持基于生成式和PPL(Perplexity-based，困惑度)模式精度测评。

- 要求：模型已部署，需测试其实际服务能力

- 此场景支持的模型任务和数据集任务：

    - **模型任务**：📚 [服务化推理后端](../all_params/models.md#服务化推理后端)

    - **数据集任务**：📚 [开源数据集](../../get_started/datasets.md#开源数据集) 与 📚 [自定义数据集](../../get_started/datasets.md#自定义数据集)

- 约束：当前PPL模式精度测评任务只支持`vllm_api_general`和`vllm_api_general_chat`两种模型配置，其他均不支持。

依据使用需求选好**模型任务**和**数据集任务**后，此场景的具体使用方法详见文档：📚 [服务化精度测评指南](accuracy_benchmark.md)

#### 纯模型精度测评
- 功能描述：评估本地加载模型（非服务化）在不同数据集上的准确性

- 要求：离线模型权重和部署环境

- 支持：

    - **模型任务**：📚 [本地模型后端](../all_params/models.md#本地模型后端)

    - **数据集任务**：📚 [开源数据集](../../get_started/datasets.md#开源数据集) 与 📚 [自定义数据集](../../get_started/datasets.md#自定义数据集)

- 约束：不支持PPL模式测评任务

依据使用需求选好**模型任务**和**数据集任务**后，此场景的具体使用方法详见文档：📚 [纯模型精度测评指南](accuracy_benchmark_local.md)

#### 基于Harbor的Agent 测评

- 功能描述：通过 `--mode agent` 拉起 [Harbor](https://github.com/harbor-framework/harbor) 执行 Agent 测评，逐 case 执行并实时监控，输出单表 + CSV 汇总。内置 Harbor 全量 Agent，不同 Agent 对同一含义参数由 `AgentParamAdapter` 自动适配（统一 `--api-base` / `--agent-api-key`），并支持自定义 `module.path:ClassName` Agent。

- 要求：一个遵循 **OpenAI chat/completions API** 规范且支持 **tool call** 的被测推理服务；Python 3.12 环境；按 Harbor 要求准备 Docker / 执行环境；安装独立依赖集 `requirements/agent.txt`（安装过程可能出现不影响使用的版本冲突/编译告警，可忽略）。

- 支持：
    - **Agent 任务**：Harbor `AgentName` 全量内置 Agent 或自定义 `module.path:ClassName`（`-a/--agent`）
    - **数据集任务**：Harbor 能解析的三种来源——本地数据集目录 / 单一 task 目录（`-p/--agent-dataset-path`）、Registry `name@version`、Package `org/name@ref`（`-d/--dataset`）
    - 统一语义参数（`--api-base` / `--agent-api-key` / `--model`）、参数覆盖（`--ak` / `--ae`）、实时监控 HTTP 服务（`--monitor-port`）、断点续测（`--reuse`）与异常用例自动重试（`--reuse` + `--purge-exception-cases`）

- 约束：`--purge-exception-cases` 仅在 `--reuse` 生效时启用；Agent 测评不依赖 AISBench 原生推理/精度链路，为一套独立精简依赖集。

依据使用需求选好 **Agent 任务**和**数据集任务**后，此场景的具体使用方法详见文档：📚 [基于Harbor的Agent测评指南](agent_benchmark.md)

### 性能测评
#### 服务化性能测评
- 功能描述：在真实部署环境中评估服务模型的运行效率（吞吐、延迟）

- 要求：模型推理服务需支持**流式接口**方式访问

- 支持：

    - **模型任务**：📚 [服务化推理后端](../all_params/models.md#服务化推理后端)中的流式接口类型

    - **数据集任务**：📚 [支持数据集类型](../../get_started/datasets.md#支持数据集类型)中的所有数据类型

- 注意：性能测评所占用的缓存大小与请求的上下文长度以及请求的数量成正比，因此通常与测评时长呈正相关增长

- 约束：不支持PPL模式测评任务

依据使用需求选好**模型任务**和**数据集任务**后，此场景的具体使用方法详见文档：📚 [服务化性能测评指南](performance_benchmark.md#服务化性能测评指南)。