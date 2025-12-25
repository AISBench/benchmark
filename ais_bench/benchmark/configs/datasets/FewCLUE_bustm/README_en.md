# FewCLUE_bustm
[中文](README.md) | English
## Dataset Introduction
This dataset is a dialogue short-text semantic matching dataset derived from Xiaobu Assistant. It is a voice assistant independently developed by OPPO for brand mobile phones and IoT devices, providing users with convenient conversational services. Intent recognition is a core task in dialogue systems, and dialogue short-text semantic matching is one of the mainstream algorithmic solutions for intent recognition. It requires predicting whether short-text query-pairs belong to the same semantics.

> 🔗 Dataset Homepage Link: [https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/bustm](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/bustm)

## Dataset Deployment
- You can download the aggregated dataset from the link provided by OpenCompass 🔗: [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip), then copy the files under `data/FewCLUE/bustm` in the compressed package to `FewCLUE/bustm/`.
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir -p FewCLUE/bustm/
cp -r OpenCompassData-core-20240207/data/FewCLUE/bustm/* FewCLUE/bustm/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- Execute `tree FewCLUE/bustm` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    bustm/
    ├── dev_0.json
    ├── dev_1.json
    ├── dev_2.json
    ├── dev_3.json
    ├── dev_4.json
    ├── dev_few_all.json
    ├── test.json
    ├── test_public.json
    ├── train_0.json
    ├── train_1.json
    ├── train_2.json
    ├── train_3.json
    ├── train_4.json
    ├── train_few_all.json
    └── unlabeled.json
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| FewCLUE_bustm_ppl_0_shot_chat | PPL task for the FewCLUE_bustm dataset | Accuracy | 0-shot | Chat format | [FewCLUE_bustm_ppl_0_shot_chat.py](FewCLUE_bustm_ppl_0_shot_chat.py) |
