# HellaSwag
[中文](README.md) | English
## Dataset Introduction
HellaSwag is a benchmark dataset for evaluating natural language understanding capabilities, primarily used to test models' performance in commonsense reasoning. The dataset contains multiple multiple-choice questions, requiring models to select the most reasonable answer from several options.

> 🔗 Dataset Homepage: [https://huggingface.co/datasets/Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/hellaswag.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/hellaswag.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/hellaswag.zip
unzip hellaswag.zip
rm hellaswag.zip
```
- Execute `tree hellaswag/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    hellaswag
    ├── hellaswag.jsonl
    ├── hellaswag_train_sampled25.jsonl
    └── hellaswag_val_contamination_annotations.json
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| hellaswag_gen_0_shot_chat_prompt | Generative task for the HellaSwag dataset | Accuracy | 0-shot | Chat format | [hellaswag_gen_0_shot_chat_prompt.py](hellaswag_gen_0_shot_chat_prompt.py) |
| hellaswag_gen_10_shot_chat_prompt | Generative task for the HellaSwag dataset | Accuracy | 10-shot | Chat format | [hellaswag_gen_10_shot_chat_prompt.py](hellaswag_gen_10_shot_chat_prompt.py) |