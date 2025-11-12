# AIME2024
[中文](README.md) | English
## Dataset Introduction
The AIME2024 dataset includes 30 problems from the **2024 American Invitational Mathematics Examination (AIME) I** ([https://artofproblemsolving.com/wiki/index.php/2024_AIME_I?srsltid=AfmBOoqP9aelPNCpuFLO2bLyoG9_elEBPgqcYyZAj8LtiywUeG5HUVfF](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I?srsltid=AfmBOoqP9aelPNCpuFLO2bLyoG9_elEBPgqcYyZAj8LtiywUeG5HUVfF)) and **2024 AIME II** ([https://artofproblemsolving.com/wiki/index.php/2024_AIME_II_Problems/Problem_15](https://artofproblemsolving.com/wiki/index.php/2024_AIME_II_Problems/Problem_15)). Its original source is [AI-MO/aimo-validation-aime](https://huggingface.co/datasets/AI-MO/aimo-validation-aime), which contains a larger problem set covering 90 problems from the 2022–2024 AIME.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
mkdir aime/
cd aime/
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip
unzip aime.zip
rm aime.zip
```
- Execute `tree aime/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    aime
    └── aime.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| aime2024_gen_0_shot_str | Generative task for the aime2024 dataset | accuracy (pass@1) | 0-shot | String format | [aime2024_gen_0_shot_str.py](aime2024_gen_0_shot_str.py) |
| aime2024_gen_0_shot_chat_prompt | Generative task for the aime2024 dataset (aligned with DeepSeek R1 accuracy test) | accuracy (pass@1) | 0-shot | Chat format | [aime2024_gen_0_shot_chat_prompt.py](aime2024_gen_0_shot_chat_prompt.py) |