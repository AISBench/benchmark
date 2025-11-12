# BoolQ
[中文](README.md) | English
## Dataset Introduction
BoolQ is a question-answering dataset for answering yes/no questions, containing 15,942 examples. These questions are naturally occurring — they are generated without prompts or restrictions. Each example consists of a triple (question, passage, answer), with the page title provided as optional additional background information.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/google/boolq](https://huggingface.co/datasets/google/boolq)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/SuperGLUE.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/SuperGLUE.zip)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/SuperGLUE.zip
unzip SuperGLUE.zip
rm SuperGLUE.zip
```
- Execute `tree SuperGLUE/BoolQ/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    BoolQ/
    ├── test.jsonl
    └── val.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code File Path |
| --- | --- | --- | --- | --- | --- |
| SuperGLUE_BoolQ_gen_883d50_str | Generative task for the BoolQ dataset | Accuracy (naive_average) | 0-shot | String | [SuperGLUE_BoolQ_gen_883d50_str.py](SuperGLUE_BoolQ_gen_883d50_str.py) |
| SuperGLUE_BoolQ_gen_0_shot_cot_str | Generative task for the BoolQ dataset, with a chain-of-thought in the prompt | Accuracy (naive_average) | 0-shot | String | [SuperGLUE_BoolQ_gen_0_shot_cot_str.py](SuperGLUE_BoolQ_gen_0_shot_cot_str.py) |
| SuperGLUE_BoolQ_gen_5_shot_str | Generative task for the BoolQ dataset (few-shot setting) | Accuracy (naive_average) | 5-shot | String | [SuperGLUE_BoolQ_gen_5_shot_str.py](SuperGLUE_BoolQ_gen_5_shot_str.py) |
| SuperGLUE_BoolQ_gen_0_shot_str | Generative task for the BoolQ dataset (note: there is a possible inconsistency between the "Few-Shot" setting and the task name; the "Few-Shot" column shows 5-shot, while the task name indicates 0-shot) | Accuracy (naive_average) | 5-shot | String | [SuperGLUE_BoolQ_gen_0_shot_str.py](SuperGLUE_BoolQ_gen_0_shot_str.py) |


### Note
There is a potential inconsistency in the task `SuperGLUE_BoolQ_gen_0_shot_str`: its name indicates a "0-shot" setting, but the "Few-Shot" column in the original table is marked as "5-shot". This discrepancy is retained in the translation to reflect the original information, and users are advised to verify the actual few-shot configuration when using this task.