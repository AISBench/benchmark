# FewCLUE_chid
[中文](README.md) | English
## Dataset Introduction
This dataset task is Chinese idiom reading comprehension and fill-in-the-blank, implemented in the form of Chinese idiom cloze. Multiple idioms in the text are masked, and the candidate options contain synonymous idioms.

> 🔗 Dataset Homepage Link: [https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/chid](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/chid)

## Dataset Deployment
- You can download the aggregated dataset from the link provided by OpenCompass 🔗: [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip), then copy the files under `data/FewCLUE/chid` in the compressed package to `FewCLUE/chid/`.
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir -p FewCLUE/chid/
cp -r OpenCompassData-core-20240207/data/FewCLUE/chid/* FewCLUE/chid/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- Execute `tree FewCLUE/chid` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    chid/
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
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Import Statement | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- | --- |
| FewCLUE_chid_ppl_0_shot_str | PPL task for the FewCLUE_chid dataset | Accuracy | 0-shot | String format |`from ais_bench.benchmark.configs.datasets.FewCLUE_chid.FewCLUE_chid_ppl_0_shot_str import chid_datasets as datasets`| [FewCLUE_chid_ppl_0_shot_str.py](FewCLUE_chid_ppl_0_shot_str.py) |
