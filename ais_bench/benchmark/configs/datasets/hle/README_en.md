# HLE (Humanity's Last Exam)
[中文](README.md) | English

## Dataset Introduction

HLE (Humanity's Last Exam) is a frontier multimodal benchmark dataset released by the Center for AI Safety, designed to be the last widely covering closed-book academic benchmark across multiple subject domains. The dataset contains 2,500 high-quality questions covering multiple subject areas including mathematics, humanities, and natural sciences. The questions are carefully designed by domain experts, ensuring professional quality and challenging difficulty. HLE supports both pure text and image inputs, and uses the LLM Judge evaluation protocol for automatic scoring while providing confidence calibration metrics. It is suitable for comprehensive evaluation of models' capabilities in multidisciplinary knowledge and multimodal understanding.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/cais/hle](https://huggingface.co/datasets/cais/hle)
> 
> 🔗 Official GitHub Repository: [https://github.com/centerforaisafety/hle](https://github.com/centerforaisafety/hle)


## Dataset Deployment

- The dataset can be obtained from the Hugging Face dataset link: 🔗 [https://huggingface.co/datasets/cais/hle](https://huggingface.co/datasets/cais/hle).
- The HLE dataset is in Parquet format and is recommended to be deployed in the `{tool_root_path}/ais_bench/datasets/hle/data/` directory.

- Execute `ls -la` in the `{tool_root_path}/ais_bench/datasets/hle/data/` directory to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    {tool_root_path}/ais_bench/datasets/hle/data/
    └── test-00000-of-00001.parquet
    ```


## Available Dataset Tasks

| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| hle | HLE dataset | Accuracy, Calibration Error | 0-shot | Chat format | hle_llmjudge.py |

