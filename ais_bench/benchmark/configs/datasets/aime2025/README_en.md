# AIME2025
[中文](README.md) | English
## Dataset Introduction
The AIME2025 dataset is derived from the **2025 American Invitational Mathematics Examination (AIME)** and includes a series of high-difficulty mathematics competition problems for middle school level. This exam is specifically designed for high school students in the United States, aiming to select candidates for the United States of America Mathematical Olympiad (USAMO). The AIME2025 dataset contains a total of 30 official competition problems, covering multiple fields such as algebra, number theory, combinatorics, and geometry. Each problem has a single integer solution; the problem design emphasizes logical reasoning and mathematical modeling abilities, and the difficulty is much higher than that of regular middle school mathematics problems. This dataset is suitable for evaluating the ability of models in complex mathematical reasoning and symbolic computation.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/opencompass/AIME2025](https://huggingface.co/datasets/opencompass/AIME2025)


## Dataset Deployment
- The compressed package of the dataset can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip
unzip aime2025.zip
rm aime2025.zip
```
- Execute `tree aime2025/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    aime2025/
    └── aime2025.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| aime2025_gen | Generative task for the AIME2025 dataset | Accuracy | 0-shot | Chat format | aime2025_gen_0_shot_chat_prompt.py |