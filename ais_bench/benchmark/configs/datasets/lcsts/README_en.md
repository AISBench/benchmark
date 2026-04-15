# LCSTS
[中文](README.md) | English
## Dataset Introduction
The LCSTS dataset is a large-scale Chinese short text summarization dataset released by the Shenzhen Graduate School of Harbin Institute of Technology. It is mainly sourced from China's Weibo platform and contains over 2 million real Chinese short texts along with their corresponding concise summaries written by the original authors. Additionally, researchers have manually annotated the relevance between 10,666 of these summaries and their matching short texts.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/aligeniewcp22/LCSTS](https://huggingface.co/datasets/aligeniewcp22/LCSTS)

## Dataset Deployment
- You can download the aggregated dataset from the link provided by OpenCompass 🔗: [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip), then copy the files under the `data/LCSTS/` folder in the compressed package to the `LCSTS/` directory.
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir LCSTS/
cp -r OpenCompassData-core-20240207/data/LCSTS/* LCSTS/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- Execute `tree LCSTS/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    LCSTS/
    ├── test.src.txt
    ├── test.tgt.txt
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| lcsts_gen_0_shot_chat | Generative task for the LCSTS dataset | Accuracy | 0-shot | Chat format | [lcsts_gen_0_shot_chat.py](lcsts_gen_0_shot_chat.py) |
| lcsts_gen_0_shot_str | Generative task for the LCSTS dataset | Accuracy | 0-shot | String format | [lcsts_gen_0_shot_str.py](lcsts_gen_0_shot_str.py) |