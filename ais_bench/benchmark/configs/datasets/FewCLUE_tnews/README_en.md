# FewCLUE_tnews
[中文](README.md) | English
## Dataset Introduction
This dataset task is a Chinese news classification task. Given a news text, it is required to determine which of the 15 categories the news belongs to, including agriculture news, travel news, game news, technology company news, sports news, junior high school education news, entertainment news, investment information, military common sense, vehicle news, real estate news, global news excluding China, books culture and history news, story news, stock market news, etc.

> 🔗 Dataset Homepage Link: [https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/tnews](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/tnews)

## Dataset Deployment
- You can download the aggregated dataset from the link provided by OpenCompass 🔗: [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip), then copy the files under `data/FewCLUE/tnews` in the compressed package to `FewCLUE/tnews/`.
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
mkdir -p FewCLUE/tnews/
cp -r OpenCompassData-core-20240207/data/FewCLUE/tnews/* FewCLUE/tnews/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- Execute `tree FewCLUE/tnews` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    tnews/
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
| FewCLUE_tnews_ppl_0_shot_chat | PPL task for the FewCLUE_tnews dataset | Accuracy | 0-shot | Chat format | [FewCLUE_tnews_ppl_0_shot_chat.py](FewCLUE_tnews_ppl_0_shot_chat.py) |

