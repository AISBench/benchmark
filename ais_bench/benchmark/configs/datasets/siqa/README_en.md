# SIQA
[中文](README.md) | English
## Dataset Introduction
SIQA (Social Interaction QA) is a question-answering benchmark designed to test social commonsense intelligence. Unlike many previous benchmarks that focus on physical or categorical knowledge, SIQA centers on reasoning about people’s behaviors and their social implications. For example, given an action like "Jesse attended a concert" and a question such as "Why did Jesse do this?", humans can easily infer that Jesse wanted to "see his favorite performer" or "enjoy the music," rather than "check what was happening inside" or "see if it worked."

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/allenai/social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa)

## Dataset Deployment
- You can download the aggregated dataset from the link provided by OpenCompass 🔗: [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip), then copy the files under `data/siqa/` in the compressed package to the `siqa/` directory.
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
mkdir siqa/
cp -r OpenCompassData-core-20240207/data/siqa/* siqa/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- Execute `tree siqa/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    siqa/
    ├── dev.jsonl
    ├── dev-labels.lst
    ├── train.jsonl
    ├── train-labels.lst
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code File Path |
| --- | --- | --- | --- | --- | --- |
| siqa_gen_0_shot_chat | Generative task for the SIQA dataset; The `EDAccEvaluator` accuracy evaluation method selects the closest answer using the `Levenshtein distance algorithm`, which may cause misjudgment and result in an artificially high accuracy score. | Accuracy | 0-shot | Chat Format | [siqa_gen_0_shot_chat.py](siqa_gen_0_shot_chat.py) |