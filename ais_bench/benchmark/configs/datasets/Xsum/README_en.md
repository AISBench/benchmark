# XSum
[中文](README.md) | English
## Dataset Introduction
The XSum (Extreme Summarization) dataset is designed for evaluating abstractive single-document summarization systems. Its goal is to create a concise, one-sentence new summary that answers the question "What is this article about?". The dataset contains 226,711 news articles, each accompanied by a one-sentence summary. These articles are sourced from the BBC (2010–2017) and cover a wide range of domains, including news, politics, sports, weather, business, technology, science, health, family, education, entertainment, and arts.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/EdinburghNLP/xsum](https://huggingface.co/datasets/EdinburghNLP/xsum)

## Dataset Deployment
- You can download the aggregated dataset from the link provided by OpenCompass 🔗: [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip), then copy the files under `data/Xsum/` in the compressed package to the `Xsum/` directory.
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir Xsum/
cp -r OpenCompassData-core-20240207/data/Xsum/* Xsum/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- Execute `tree Xsum/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    Xsum/
    ├── dev.csv
    ├── dev.json
    ├── dev.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code File Path |
| --- | --- | --- | --- | --- | --- |
| Xsum_gen_0_shot_chat | Generative task for the XSum dataset | Accuracy | 0-shot | Chat Format | [Xsum_gen_0_shot_chat.py](Xsum_gen_0_shot_chat.py) |
| Xsum_gen_0_shot_str | Generative task for the XSum dataset | Accuracy | 0-shot | String Format | [Xsum_gen_0_shot_str.py](Xsum_gen_0_shot_str.py) |