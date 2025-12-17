# LiveCodeBench
[中文](README.md) | English
## Dataset Introduction
LiveCodeBench is a continuously updated "real-time" benchmarking platform designed to comprehensively evaluate the code-related capabilities of Large Language Models (LLMs). This platform primarily assesses models' performance across multiple dimensions, including code generation, self-repair, test output prediction, and code execution. The current version showcases its code generation scenario, which is also used to evaluate the model's self-repair capability through test case feedback.

The problems in this benchmark are collected from programming competition websites, with special emphasis on maintaining the quality of questions, the quality of test cases, and the diversity of question difficulty levels. The current version includes more than 500 questions from LeetCode, AtCoder, and Codeforces. Each problem instance consists of a problem description, input/output examples, and hidden test cases. All questions are labeled with difficulty levels and release times, facilitating the measurement of model performance across different time windows. The ultimate goal is to generate correct and efficient solutions for each problem instance.

The initial version of the code generation dataset had an excessively large size due to the inclusion of a large number of test cases. The current (lightweight) version has undergone test case filtering and sampling while maintaining performance similar to the original dataset. In the future, LiveCodeBench will use this lightweight version for code generation evaluation.

> 🔗 Dataset Homepage Link: [https://livecodebench.github.io/](https://livecodebench.github.io/)

## Dataset Deployment
- The dataset can be obtained from the Hugging Face dataset link 🔗: [https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/main](https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/main)
- Please deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks)，as deploying to a custom path may result in dataset-related errors (November 25, 2025). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
git lfs install
git clone https://huggingface.co/datasets/livecodebench/code_generation_lite
```
- Execute `tree code_generation_lite/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    code_generation_lite
    ├── code_generation_lite.py
    ├── test6.jsonl
    ├── test5.jsonl
    ├── test4.jsonl
    ├── test3.jsonl
    ├── test2.jsonl
    └── test.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
|livecodebench_0_shot_chat_v4_v5|Generative task for the code_generation_lite dataset, same with DeepSeek-R1 Evaluation: LiveCodeBench(2024-08 – 2025-01)|pass@1|0-shot|Chat format|[livecodebench_0_shot_chat_v4_v5.py](livecodebench_0_shot_chat_v4_v5.py)|
|livecodebench_0_shot_chat_v4_v5_v6|Generative task for the code_generation_lite dataset, same with DeepSeek-V3.1 and DeepSeek-V3.2 Evaluation: LiveCodeBench(2024-08 – 2025-05)|pass@1|0-shot|Chat format|[livecodebench_0_shot_chat_v4_v5_v6.py](livecodebench_0_shot_chat_v4_v5_v6.py)|
|livecodebench_0_shot_chat_v6|Generative task for the code_generation_lite dataset, same with Qwen3 Evaluation: LiveCodeBench(2025-05)|pass@1|0-shot|Chat format|[livecodebench_0_shot_chat_v6.py](livecodebench_0_shot_chat_v6.py)|
