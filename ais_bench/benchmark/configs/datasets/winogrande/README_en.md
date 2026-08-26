# WinoGrande
[中文](README.md) | English
## Dataset Introduction
WinoGrande is a new dataset containing 44,000 questions. Its design is inspired by the Winograd Schema Challenge (Levesque, Davis, & Morgenstern, 2011), but it has been improved by adjusting the scale and enhancing robustness against dataset-specific biases. The task adopts a two-option cloze format, where the goal is to select the correct option that aligns with commonsense reasoning for a given sentence.

> 🔗 Dataset Homepage: [https://huggingface.co/datasets/allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/winogrande.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/winogrande.zip)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/winogrande.zip
unzip winogrande.zip
rm winogrande.zip
```
- Execute `tree winogrande/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    winogrande
    ├── dev.jsonl
    ├── dev-labels.lst
    ├── eval.py
    ├── README.md
    ├── sample-submission-labels.lst
    ├── test.jsonl
    ├── train_debiased.jsonl
    ├── train_debiased-labels.lst
    ├── train_l.jsonl
    ├── train_l-labels.lst
    ├── train_m.jsonl
    ├── train_m-labels.lst
    ├── train_s.jsonl
    ├── train_s-labels.lst
    ├── train_xl.jsonl
    ├── train_xl-labels.lst
    ├── train_xs.jsonl
    └── train_xs-labels.lst
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Import Statement | Corresponding Source Code File Path |
| --- | --- | --- | --- | --- | --- | --- |
| winogrande_gen_0_shot_chat_prompt | Generative task for the WinoGrande dataset | Accuracy | 0-shot | Chat Format |`from ais_bench.benchmark.configs.datasets.winogrande.winogrande_gen_0_shot_chat_prompt import winogrande_datasets as datasets`| [winogrande_gen_0_shot_chat_prompt.py](winogrande_gen_0_shot_chat_prompt.py) |
| winogrande_gen_5_shot_chat_prompt | Generative task for the WinoGrande dataset (Note: The original "piqa dataset" in the introduction is a typo, corrected to "WinoGrande dataset" for consistency) | Accuracy | 5-shot | Chat Format |`from ais_bench.benchmark.configs.datasets.winogrande.winogrande_gen_5_shot_chat_prompt import winogrande_datasets as datasets`| [winogrande_gen_5_shot_chat_prompt.py](winogrande_gen_5_shot_chat_prompt.py) |


### Note
There is a typo in the original introduction of the `winogrande_gen_5_shot_chat_prompt` task: it incorrectly refers to the "piqa dataset" instead of the "WinoGrande dataset". This has been corrected in the translation to ensure consistency with the task name and dataset context.