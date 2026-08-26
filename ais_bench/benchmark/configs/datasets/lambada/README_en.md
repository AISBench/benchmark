# LAMBADA
[中文](README.md) | English
## Dataset Introduction
The LAMBADA (Language Modeling Broadened to Account for Discourse Aspects) dataset is an open-ended cloze task designed to evaluate the ability of computational models to understand text. It contains approximately 10,000 paragraphs extracted from the BooksCorpus, where the last sentence of each paragraph is missing a target word, and the model is required to predict this missing word.

> 🔗 Dataset Homepage: [https://huggingface.co/datasets/cimec/lambada](https://huggingface.co/datasets/cimec/lambada)

## Dataset Deployment
- You can download the aggregated dataset from the link provided by OpenCompass 🔗: [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip), then copy the files under the `data/lambada/` folder in the compressed package to the `lambada/` directory.
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir lambada/
cp -r OpenCompassData-core-20240207/data/lambada/* lambada/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- Execute `tree lambada/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    lambada/
    ├── test.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Import Statement | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- | --- |
| lambada_gen_0_shot_chat | Generative task for the LAMBADA dataset | Accuracy | 0-shot | Chat format |`from ais_bench.benchmark.configs.datasets.lambada.lambada_gen_0_shot_chat import lambada_datasets as datasets`| [lambada_gen_0_shot_chat.py](lambada_gen_0_shot_chat.py) |
| lambada_gen_0_shot_str | Generative task for the LAMBADA dataset | Accuracy | 0-shot | String format |`from ais_bench.benchmark.configs.datasets.lambada.lambada_gen_0_shot_str import lambada_datasets as datasets`| [lambada_gen_0_shot_str.py](lambada_gen_0_shot_str.py) |


### Translation Notes
1. **Acronym & Naming Consistency**: The full name of "LAMBADA" (Language Modeling Broadened to Account for Discourse Aspects) is retained in its original form to preserve the dataset’s official naming convention. "BooksCorpus" (a well-known text corpus in NLP) and "OpenCompass" (the platform name) are also kept unchanged for technical recognizability.
2. **Task Description Precision**: "开放式填空任务" is translated as "open-ended cloze task"—the standard term in NLP for tasks requiring filling in missing words/phrases in text. "预测这个缺失的词" is rendered as "predict this missing word" to accurately convey the core requirement of the dataset.
3. **Code & Path Integrity**: Linux commands (e.g., `wget`, `unzip`, `cp -r`), directory paths (e.g., `{tool_root_path}/ais_bench/datasets`), and filenames (e.g., `test.jsonl`, `lambada_gen_0_shot_chat.py`) are copied exactly to ensure the deployment instructions remain actionable for technical users.
4. **Semantic Clarity**: The description of the dataset’s purpose ("旨在评估计算模型对文本理解的能力") is translated to clearly link the task to its goal ("designed to evaluate the ability of computational models to understand text"), adhering to the concise and precise style of technical documentation.