# 推理器（Inferencer）概述

推理器（Inferencer）是 AISBench 中负责执行模型推理的核心组件，它连接了数据集、检索器（Retriever）和模型，负责将处理好的提示词（Prompt）发送给模型进行推理，并收集和管理推理结果。

## 核心功能

推理器在 AISBench 的评测流程中承担以下核心职责：

1. **数据准备**：从检索器（Retriever）获取数据列表，包括输入提示词、标准答案等信息
2. **模型调用**：根据模型类型（API 模型或本地模型）采用不同的方式调用模型进行推理
   - **API 模型**：通过异步 HTTP 请求调用服务化推理接口
   - **本地模型**：直接调用本地加载的模型进行批量推理
3. **结果管理**：收集、处理和保存推理结果，包括：
   - 模型生成的文本内容
   - 推理状态（成功/失败）
   - 性能指标（如延迟、吞吐量等，在性能模式下）
   - 错误信息（如果推理失败）
4. **状态跟踪**：在性能测评模式下，跟踪和统计请求状态，包括：
   - 已发送请求数（post）
   - 已接收响应数（rev）
   - 失败请求数（failed）
   - 已完成请求数（finish）

## 架构设计

推理器采用分层设计，包含以下基类：

- **BaseInferencer**：所有推理器的基类，提供模型构建、输出处理等通用功能
- **BaseApiInferencer**：API 模型推理器的基类，提供异步请求处理、状态跟踪等功能
- **BaseLocalInferencer**：本地模型推理器的基类，提供批量推理、数据加载等功能

推理器可以根据需要同时继承 `BaseApiInferencer` 和 `BaseLocalInferencer`，以同时支持 API 模型和本地模型。

## 当前支持的推理器类型

AISBench 目前支持以下推理器类型：

### 1. GenInferencer（生成式推理器）

**功能**：用于生成式任务的推理器，支持文本生成、问答等任务。

**特点**：

- 同时支持 API 模型和本地模型
- 支持流式和非流式推理
- 支持性能测评模式
- 支持自定义停止条件（stopping_criteria）

**适用场景**：

- 文本生成任务
- 问答任务
- 代码生成任务
- 数学推理任务

**实现文件**：[icl_gen_inferencer.py](../../../ais_bench/benchmark/openicl/icl_inferencer/icl_gen_inferencer.py)

### 2. MultiTurnGenInferencer（多轮对话推理器）

**功能**：用于多轮对话任务的推理器，支持多轮交互式对话场景。

**特点**：

- 同时支持 API 模型和本地模型
- 支持多种推理模式：
  - `every`：逐轮推理，将模型上一轮输出作为下一轮输入
  - `last`：仅对最后一轮进行推理
  - `every_with_gt`：逐轮推理，但使用标准答案而非模型输出
- 支持性能测评模式

**适用场景**：

- 多轮对话任务
- 需要上下文交互的任务
- 对话式问答任务

**实现文件**：[icl_multiturn_inferencer.py](../../../ais_bench/benchmark/openicl/icl_inferencer/icl_multiturn_inferencer.py)

### 3. PPLInferencer（困惑度推理器）

**功能**：用于困惑度（Perplexity）评估的推理器，通过计算每个选项的困惑度来选择答案，主要用于多选题（MCQ）任务。

**特点**：

- 仅支持 API 模型（不支持本地模型）
- 不支持流式推理
- 不支持性能测评模式
- 通过计算每个候选答案的困惑度，选择困惑度最低的选项作为预测结果

**适用场景**：

- 多选题（MCQ）任务
- 需要基于困惑度进行选择的分类任务

**实现文件**：[ppl_inferencer.py](../../../ais_bench/benchmark/openicl/icl_inferencer/ppl_inferencer.py)

### 4. BFCLV3FunctionCallInferencer（函数调用推理器）

**功能**：用于函数调用任务的推理器，支持模型调用外部函数或工具的场景。

**特点**：

- 仅支持 API 模型
- 支持多轮函数调用
- 支持 holdout function（保留函数）机制
- 支持函数调用的结果处理和反馈

**适用场景**：

- 函数调用任务
- 工具使用任务
- 需要模型调用外部 API 的任务

**实现文件**：[icl_bfcl_v3_inferencer.py](../../../ais_bench/benchmark/openicl/icl_inferencer/icl_bfcl_v3_inferencer.py)

## 推理器选择指南

根据不同的任务类型和模型类型，选择合适的推理器：

| 任务类型 | 模型类型 | 推荐推理器 |
|---------|---------|-----------|
| 文本生成、问答 | API 模型 | GenInferencer |
| 文本生成、问答 | 本地模型 | GenInferencer |
| 多轮对话 | API 模型 | MultiTurnGenInferencer |
| 多轮对话 | 本地模型 | MultiTurnGenInferencer |
| 多选题（MCQ） | API 模型 | PPLInferencer |
| 函数调用 | API 模型 | BFCLV3FunctionCallInferencer |

## 与相关组件的关系

推理器在 AISBench 的评测流程中与其他组件紧密协作：

1. **与 Retriever 的关系**：
   - 推理器通过 `get_data_list` 方法从 Retriever 获取数据
   - Retriever 负责生成 in-context examples 和 prompt

2. **与 Model 的关系**：
   - 推理器调用 Model 的 `generate` 方法进行推理
   - 对于 API 模型，推理器通过 HTTP 请求调用模型服务
   - 对于本地模型，推理器直接调用模型实例

3. **与 OutputHandler 的关系**：
   - 推理器使用 OutputHandler 管理和保存推理结果
   - 不同类型的推理器使用不同的 OutputHandler

4. **与 Dataset 的关系**：
   - 推理器从 Dataset 配置中获取推理相关的参数
   - 如 `max_out_len` 等参数可以从数据集配置中获取

## 进一步阅读

- [支持新的推理器](./new_inferencer.md)：了解如何实现自定义推理器
- [Prompt Template](../prompt/prompt_template.md)：了解提示词模板的定义
- [Meta Template](../prompt/meta_template.md)：了解模型元模板的定义
- [Retriever](../prompt/retriever.md)：了解检索器的工作原理
