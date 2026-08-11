# AA-LCR (Artificial Analysis Long Context Retrieval)
[中文](README.md) | English

## Dataset Introduction

AA-LCR (Artificial Analysis Long Context Retrieval) is a benchmark for evaluating language models' long-context retrieval and reasoning capabilities. The task requires models to search for and synthesize information across multiple documents to answer questions.

> 🔗 Dataset Homepage Link: [https://modelscope.cn/datasets/evalscope/AA-LCR](https://modelscope.cn/datasets/evalscope/AA-LCR)


## Dataset Deployment

- The dataset can be obtained from the ModelScope dataset link: 🔗 [https://modelscope.cn/datasets/evalscope/AA-LCR](https://modelscope.cn/datasets/evalscope/AA-LCR).
- The AA-LCR dataset is in compressed archive format and is recommended to be deployed in the `{tool_root_path}/ais_bench/datasets/aa_lcr/` directory.

- Execute `ls -la` in the `{tool_root_path}/ais_bench/datasets/aa_lcr/` directory to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    {tool_root_path}/ais_bench/datasets/aa_lcr/
    └── extracted_text/AA-LCR_extracted-text.zip
    ```


## Available Dataset Tasks

| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| aa_lcr_llmjudge | AA-LCR dataset | Accuracy | 0-shot | Chat format | aa_lcr_llmjudge.py |

