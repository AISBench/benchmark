# AA-LCR (Artificial Analysis Long Context Retrieval)
中文 | [English](README_en.md)

## 数据集简介

AA-LCR（Artificial Analysis Long Context Retrieval）是一个用于评估语言模型长上下文检索与推理能力的基准测试。该任务要求模型在多个文档中查找并综合信息以回答问题。

> 🔗 数据集主页链接: [https://modelscope.cn/datasets/evalscope/AA-LCR](https://modelscope.cn/datasets/evalscope/AA-LCR)


## 数据集部署

- 可以从 ModelScope 的数据集链接 🔗 [https://modelscope.cn/datasets/evalscope/AA-LCR](https://modelscope.cn/datasets/evalscope/AA-LCR) 中获取数据集。
- AA-LCR 数据集为压缩包格式，建议部署在 `{tool_root_path}/ais_bench/datasets/AA-LCR/` 目录下。

- 在 `{tool_root_path}/ais_bench/datasets/` 目录下执行 `tree AA-LCR/` 检查目录结构。如果目录结构如下所示，则数据集部署成功：
    ```
    AA-LCR/
    └── extracted_text/
        └── AA-LCR_extracted-text.zip
    ```


## 可用数据集任务

| 任务名称 | 简介 | 评估指标 | Few-Shot | Prompt 格式 | 对应源码配置文件路径 |
| --- | --- | --- | --- | --- | --- |
| aa_lcr_llmjudge | AA-LCR 数据集 | 准确率 (accuracy) | 0-shot | 对话格式 | aa_lcr_llmjudge.py |

