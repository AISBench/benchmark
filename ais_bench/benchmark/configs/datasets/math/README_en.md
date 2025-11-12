# MATH
[中文](README.md) | English
## Dataset Introduction
MATH is a new dataset containing 12,500 challenging competition-level mathematics problems. Each problem in the MATH dataset is accompanied by a complete step-by-step solution, which can be used to train models to generate answer derivation processes and explanatory content.

> 🔗 Dataset Homepage Link: [https://github.com/hendrycks/math/](https://github.com/hendrycks/math/)

⏰ **Note**: Please install dependencies from [extra.txt](../../../../../requirements/extra.txt) before running the dataset.
```shell
# You need to be in the outermost "benchmark" folder and run the following command:
pip3 install -r requirements/extra.txt
```

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
unzip math.zip
rm math.zip
```
- Execute `tree math/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    math
    ├── convert_jsonl2json.py
    ├── math.json
    ├── test.jsonl
    ├── test_prm800k_500.json # MATH500
    ├── test_prm800k_500.jsonl # MATH500
    └── train.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| math_prm800k_500_0shot_cot_gen | Generative task for the MATH500 dataset. The default maximum output token length is 32768, with a logical chain in the prompt. | Accuracy (pass@1) | 0-shot | String format | [math_prm800k_500_0shot_cot_gen.py](math_prm800k_500_0shot_cot_gen.py) |
| math_prm800k_500_5shot_cot_gen | Generative task for the MATH500 dataset. The default maximum output token length is 32768, with a logical chain in the prompt. | Accuracy (pass@1) | 5-shot | String format | [math_prm800k_500_5shot_cot_gen.py](math_prm800k_500_5shot_cot_gen.py) |
| math500_gen_0_shot_cot_chat_prompt | Generative task for the MATH500 dataset, with a logical chain in the prompt (aligned with DeepSeek R1 accuracy test) | Accuracy (pass@1) | 0-shot | Chat format | [math500_gen_0_shot_cot_chat_prompt.py](math500_gen_0_shot_cot_chat_prompt.py) |