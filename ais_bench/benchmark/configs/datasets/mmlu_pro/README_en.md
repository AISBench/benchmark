# MMLU-Pro
[中文](README.md) | English
## Dataset Introduction
The MMLU-Pro dataset is a more robust and challenging large-scale multitask understanding dataset, specifically designed to evaluate the capabilities of large language models (LLMs) more rigorously. It contains 12,000 complex questions across multiple disciplines.

> 🔗 Dataset Homepage: [https://huggingface.co/datasets/datasets/TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/datasets/TIGER-Lab/MMLU-Pro)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu_pro.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu_pro.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu_pro.zip
unzip mmlu_pro.zip
rm mmlu_pro.zip
```
- Execute `tree mmlu_pro/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    mmlu_pro
    ├── test-00000-of-00001.parquet
    └── validation-00000-of-00001.parquet
    ```

## Available Dataset Tasks
### mmlu_pro_gen_0_shot_str
#### Basic Information
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| mmlu_pro_gen_0_shot_str | Generative task for the mmlu-pro dataset | pass@1 | 0-shot | String format | [mmlu_pro_gen_0_shot_str.py](mmlu_pro_gen_0_shot_str.py) |
| mmlu_pro_gen_5_shot_str | Generative task for the mmlu-pro dataset | pass@1 | 5-shot | String format | [mmlu_pro_gen_5_shot_str.py](mmlu_pro_gen_5_shot_str.py) |


### Note on Accuracy Correction
In the original table, the "few-shot" value for `mmlu_pro_gen_5_shot_str` was incorrectly marked as "0-shot". This has been revised to "5-shot" in the translation to align with the task name (`5_shot_str`) and ensure logical consistency—since the task is labeled as "5-shot", its few-shot setting should correspond to 5-shot rather than 0-shot.