# CorpusQA
[中文](README.md) | English

## Dataset Introduction

CorpusQA is a benchmark for evaluating million-token (1M) long-context question-answering capabilities of language models, covering four domains: finance (Chinese/English), education, and real estate. Each sample contains a raw multi-turn `prompt`, a `question`, and a reference `answer`. The evaluation follows the official protocol: after the model generates an answer from the raw prompt, an LLM judge (ORM, Output Reward Model) decides whether the generated answer is equivalent to the reference answer, outputting `[[YES]]` or `[[NO]]`. Accuracy = number of correct samples / total samples.

> 🔗 Dataset Homepage Link: [https://github.com/Tongyi-Zhiwen/CorpusQA](https://github.com/Tongyi-Zhiwen/CorpusQA)


## Dataset Deployment

- Obtain the `1m_4domains.jsonl` data file from the dataset homepage: 🔗 [https://github.com/Tongyi-Zhiwen/CorpusQA](https://github.com/Tongyi-Zhiwen/CorpusQA).
- It is recommended to deploy the data file at `{tool_root_path}/ais_bench/datasets/CorpusQA/1m_4domains.jsonl`, consistent with the `path` field in the configuration file `corpusqa_1m_gen.py`.

- Check the directory structure under `{tool_root_path}/ais_bench/datasets/`. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    CorpusQA/
    └── 1m_4domains.jsonl
    ```


## Available Dataset Tasks

| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Import Statement | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- | --- |
| corpusqa_1m_gen | CorpusQA 1M long-context Q&A dataset | Accuracy (based on ORM LLM judge) | 0-shot | Chat format (raw multi-turn prompt) | `from ais_bench.benchmark.configs.datasets.corpusqa.corpusqa_1m_gen import corpusqa_1m_datasets as datasets` | corpusqa_1m_gen.py |
