# LongBench v2
[中文](README.md) | English
## Dataset Introduction
LongBench v2 is designed to evaluate the ability of Large Language Models (LLMs) to tackle long-context tasks requiring in-depth understanding and reasoning. It features the following characteristics:
(1) **Length**: Context lengths range from 8k to 2M tokens, with most being under 128k tokens.
(2) **Difficulty**: Extremely high difficulty—even human experts cannot provide correct answers quickly, despite having access to in-document search tools.
(3) **Coverage**: Encompasses a variety of real-world scenarios.
(4) **Reliability**: All questions are presented in multiple-choice format to ensure the reliability of evaluations.

LongBench v2 contains 503 challenging multiple-choice questions, covering six major task categories: Single-Document QA, Multi-Document QA, Long-Context Learning, Long Conversation History Understanding, Code Repository Understanding, and Long Structured Data Understanding.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2)

## Dataset Deployment
It is recommended to download the dataset from Hugging Face: [https://huggingface.co/datasets/zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks).
- After successful deployment, execute `tree LongBench-v2/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    LongBench-v2/
    └── data.json
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| longbenchv2_gen | LongBench v2 task | Accuracy | 0-shot | Chat format | [longbenchv2_gen.py](longbenchv2_gen.py) |