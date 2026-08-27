# CorpusQA
中文 | [English](README_en.md)

## 数据集简介

CorpusQA 是一个用于评估语言模型百万级（1M）长上下文问答能力的基准测试，包含金融（中/英）、教育、房地产四大领域。每条样本包含原始多轮对话 `prompt`、`question` 与参考答案 `answer`。评测遵循官方协议：模型根据原始 prompt 生成答案后，由 LLM 裁判（ORM，Output Reward Model）判断生成答案与参考答案是否等价，输出 `[[YES]]` 或 `[[NO]]`，准确率 = 正确样本数 / 总样本数。

> 🔗 数据集主页链接: [https://github.com/Tongyi-Zhiwen/CorpusQA](https://github.com/Tongyi-Zhiwen/CorpusQA)


## 数据集部署

- 可从数据集主页 🔗 [https://github.com/Tongyi-Zhiwen/CorpusQA](https://github.com/Tongyi-Zhiwen/CorpusQA) 获取 `1m_4domains.jsonl` 数据文件。
- 建议将数据文件部署在 `{tool_root_path}/ais_bench/datasets/CorpusQA/1m_4domains.jsonl` 路径下，与配置文件 `corpusqa_1m_gen.py` 中的 `path` 字段保持一致。

- 在 `{tool_root_path}/ais_bench/datasets/` 目录下检查目录结构。如果目录结构如下所示，则数据集部署成功：
    ```
    CorpusQA/
    └── 1m_4domains.jsonl
    ```


## 可用数据集任务

| 任务名称 | 简介 | 评估指标 | Few-Shot | Prompt 格式 | 配套文件导入方式 | 对应源码配置文件路径 |
| --- | --- | --- | --- | --- | --- | --- |
| corpusqa_1m_gen | CorpusQA 1M 长上下文问答数据集 | 准确率 (accuracy，基于 ORM LLM 裁判) | 0-shot | 对话格式（原始多轮 prompt） | `from ais_bench.benchmark.configs.datasets.corpusqa.corpusqa_1m_gen import corpusqa_1m_datasets as datasets` | corpusqa_1m_gen.py |
