# FewCLUE_csl
[中文](README.md) | English
## Dataset Introduction
The Chinese Scientific Literature (CSL) dataset is derived from Chinese paper abstracts and their keywords. The papers are selected from some core journals in Chinese social sciences and natural sciences. The task objective is to determine whether all keywords are genuine keywords based on the abstract (genuine is 1, fake is 0).

> 🔗 Dataset Homepage Link: [https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/csl](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/csl)

## Dataset Deployment
- You can download the aggregated dataset from the link provided by OpenCompass 🔗: [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip), then copy the files under `data/FewCLUE/csl` in the compressed package to `FewCLUE/csl/`.
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir -p FewCLUE/csl/
cp -r OpenCompassData-core-20240207/data/FewCLUE/csl/* FewCLUE/csl/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- Execute `tree FewCLUE/csl` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    csl/
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
| FewCLUE_csl_ppl_0_shot_str | PPL task for the FewCLUE_csl dataset | Accuracy | 0-shot | String format |`from ais_bench.benchmark.configs.datasets.FewCLUE_csl.FewCLUE_csl_ppl_0_shot_str import csl_datasets as datasets`| [FewCLUE_csl_ppl_0_shot_str.py](FewCLUE_csl_ppl_0_shot_str.py) |

