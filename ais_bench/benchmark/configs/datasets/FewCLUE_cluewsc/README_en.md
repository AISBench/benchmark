# FewCLUE_cluewsc
[中文](README.md) | English
## Dataset Introduction
Winograd Scheme Challenge (WSC) is a type of pronoun disambiguation task, which involves determining which noun a pronoun in a sentence refers to. The questions appear in the form of true/false judgments, for example:
Sentence: At this moment, the [phone] placed next to the [pillow] on the [bed] rang, and I felt strange because the service had been suspended for two months due to unpaid fees, and now [it] suddenly rang. It is necessary to determine whether "it" refers to "bed", "pillow", or "phone"?
The data is extracted from modern and contemporary Chinese literary works and then manually selected and annotated by language experts.

> 🔗 Dataset Homepage Link: [https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/cluewsc](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/cluewsc)

## Dataset Deployment
- You can download the aggregated dataset from the link provided by OpenCompass 🔗: [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip), then copy the files under `data/FewCLUE/cluewsc` in the compressed package to `FewCLUE/cluewsc/`.
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir -p FewCLUE/cluewsc/
cp -r OpenCompassData-core-20240207/data/FewCLUE/cluewsc/* FewCLUE/cluewsc/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- Execute `tree FewCLUE/cluewsc` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    cluewsc/
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
| FewCLUE_cluewsc_ppl_0_shot_chat | PPL task for the FewCLUE_cluewsc dataset | Accuracy | 0-shot | Chat format | [FewCLUE_cluewsc_ppl_0_shot_chat.py](FewCLUE_cluewsc_ppl_0_shot_chat.py) |
