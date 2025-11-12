# RACE
[中文](README.md) | English
## Dataset Introduction
The RACE (Reading Comprehension from Examinations) dataset is a large-scale machine reading comprehension dataset. It consists of English exam questions designed for Chinese students aged 12-18, including 27,933 passages and 97,867 questions. The RACE dataset is divided into two subsets: RACE-M and RACE-H, corresponding to the difficulty levels of junior high school and senior high school questions respectively. RACE-M contains 28,293 questions, suitable for junior high school students; RACE-H contains 69,574 questions, suitable for senior high school students. Each question is accompanied by four optional answers, one of which is the correct answer.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)

## Dataset Deployment
- You can download the aggregated dataset from the link provided by OpenCompass 🔗: [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip), then copy the files under `data/race/` in the compressed package to the `race/` directory.
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
mkdir race/
cp -r OpenCompassData-core-20240207/data/race/* race/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- Execute `tree race/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    race/
    ├── test/
    │──── high.jsonl
    │──── middle.jsonl
    ├── validation/
    │──── high.jsonl
    │──── middle.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code File Path |
| --- | --- | --- | --- | --- | --- |
| race_middle_gen_5_shot_chat | Generative task for the RACE dataset (middle school level) | Accuracy | 5-shot | Chat Format | [race_middle_gen_5_shot_chat.py](race_middle_gen_5_shot_chat.py) |
| race_middle_gen_5_shot_cot_chat | Generative task for the RACE dataset (middle school level) with chain-of-thought in prompt | Accuracy | 5-shot | Chat Format | [race_middle_gen_5_shot_cot_chat.py](race_middle_gen_5_shot_cot_chat.py) |
| race_high_gen_5_shot_chat | Generative task for the RACE dataset (senior high school level) | Accuracy | 5-shot | Chat Format | [race_high_gen_5_shot_chat.py](race_high_gen_5_shot_chat.py) |
| race_high_gen_5_shot_cot_chat | Generative task for the RACE dataset (senior high school level) with chain-of-thought in prompt | Accuracy | 5-shot | Chat Format | [race_high_gen_5_shot_cot_chat.py](race_high_gen_5_shot_cot_chat.py) |