# AIME2026
[中文](README.md) | English
## Dataset Introduction
The AIME2026 dataset is derived from the **2026 American Invitational Mathematics Examination (AIME)** and contains official problems from AIME I and AIME II. AIME is one of the most challenging high school mathematics competitions in the United States, designed to test creative mathematical reasoning and problem-solving abilities across algebra, geometry, number theory, combinatorics, probability, and related topics. The AIME2026 dataset contains 30 problems in total. Each problem has an integer answer from 0 to 999, making the dataset suitable for evaluating models on complex mathematical reasoning, step-by-step problem solving, and symbolic computation.

> 🔗 Dataset Homepage Link: [https://modelscope.cn/datasets/evalscope/aime26/summary](https://modelscope.cn/datasets/evalscope/aime26/summary)


## Dataset Deployment
- The dataset can be obtained from the ModelScope dataset homepage 🔗: [https://modelscope.cn/datasets/evalscope/aime26/summary](https://modelscope.cn/datasets/evalscope/aime26/summary), and organized as an `aime2026.jsonl` file.
- It is recommended to deploy the dataset in the directory `ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the target directory structure can be prepared as follows:
```bash
# Within the Linux server, under the ais_bench directory path
mkdir -p datasets/aime2026
# Place the prepared aime2026.jsonl file in this directory
```
- Execute `tree aime2026/` in the directory `ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    aime2026/
    └── aime2026.jsonl
    ```

## Data Format
The data file uses the JSON Lines format. Each line corresponds to one problem and contains the following main fields:

| Field | Introduction |
| --- | --- |
| problem | AIME mathematics problem statement |
| answer | Integer answer to the problem |

## Prompt Template
The default 0-shot task uses the following prompt template, asking the model to solve the problem step by step and put the final answer inside `\boxed{}` for answer extraction and evaluation.

```text
Solve the following math problem step by step. Put your answer inside \boxed{}.

{problem}

Remember to put your answer inside \boxed{}.
```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| aime2026_gen | Generative task for the AIME2026 dataset | Accuracy | 0-shot | Chat format | aime2026_gen_0_shot_chat_prompt.py |
| aime2026_gen_0_shot_str | Generative task for the AIME2026 dataset | Accuracy | 0-shot | String format | aime2026_gen_0_shot_str.py |
