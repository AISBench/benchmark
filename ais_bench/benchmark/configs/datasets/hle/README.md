# HLE (Humanity's Last Exam)
中文 | [English](README_en.md)

## 数据集简介

HLE（Humanity's Last Exam）是 Center for AI Safety 发布的前沿多模态基准测试数据集，旨在成为最后一个广泛覆盖学科领域的闭卷学术基准测试。该数据集包含 2,500 道高质量题目，涵盖数学、人文科学、自然科学等多个学科领域。题目由领域专家精心设计，确保了题目的专业性和挑战性。HLE 支持纯文本和图像输入，并通过 LLM Judge 评估协议进行自动评分，同时提供置信度校准指标，适合全面评估模型在多学科知识和多模态理解方面的能力。

> 🔗 数据集主页链接: [https://huggingface.co/datasets/cais/hle](https://huggingface.co/datasets/cais/hle)
> 
> 🔗 官方 GitHub 仓库: [https://github.com/centerforaisafety/hle](https://github.com/centerforaisafety/hle)


## 数据集部署

- 可以从huggingface的数据集链接🔗 [https://huggingface.co/datasets/cais/hle](https://huggingface.co/datasets/cais/hle)中获取数据集。
- HLE 数据集为 Parquet 格式，建议部署在 `{tool_root_path}/ais_bench/datasets/hle/data/` 目录下。

- 在 `{tool_root_path}/ais_bench/datasets/hle/data/` 目录下执行 `ls -la` 检查目录结构。如果目录结构如下所示，则数据集部署成功：
    ```
    {tool_root_path}/ais_bench/datasets/hle/data/
    └── test-00000-of-00001.parquet
    ```


## 可用数据集任务

| 任务名称 | 简介 | 评估指标 | Few-Shot | Prompt 格式 | 对应源码配置文件路径 |
| --- | --- | --- | --- | --- | --- |
| hle | HLE 数据集 | 准确率 (accuracy)、置信度校准误差 (calibration_error) | 0-shot | 对话格式 | hle_llmjudge.py |

