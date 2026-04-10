# ARC Easy Set
[中文](README.md) | English
## Dataset Introduction
ARC (AI2 Reasoning Challenge) is a new dataset containing 7,787 real primary school-level science multiple-choice questions, designed to advance research in advanced question-answering technologies. The dataset is divided into two subsets: the **Challenge Set** and the **Easy Set**. Among them, the Challenge Set only includes difficult questions that both retrieval-based algorithms and word co-occurrence algorithms fail to answer correctly. The set covered in this document is the Easy Set.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)

## Dataset Deployment
- You can download the aggregated dataset from the link provided by OpenCompass 🔗: [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip), then copy the files under the `data/ARC/` folder in the compressed package to the target `ARC/` folder.
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir ARC/
cp -r OpenCompassData-core-20240207/data/AGIEval/data/v1/* ARC/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- Execute `tree ARC/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    ARC/
    └── ARC-e
        ├── ARC-Easy-Dev.jsonl
        └── ARC-Easy-Test.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| ARC_e_gen_0_shot_chat_prompt | Generative task for the ARC Easy Set dataset | Accuracy | 0-shot | Chat format | [ARC_e_gen_0_shot_chat_prompt.py](ARC_e_gen_0_shot_chat_prompt.py) |
| ARC_e_gen_25_shot_chat_prompt | Generative task for the ARC Easy Set dataset | Accuracy | 25-shot | Chat format | [ARC_e_gen_25_shot_chat_prompt.py](ARC_e_gen_25_shot_chat_prompt.py) |
| ARC_e_ppl_0_shot_str | PPL task for the ARC Easy Set dataset | Accuracy | 0-shot | String Format | [ARC_e_ppl_0_shot_str.py](ARC_e_ppl_0_shot_str.py) |