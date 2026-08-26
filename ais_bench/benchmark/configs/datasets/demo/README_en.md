# DEMO
[中文](README.md) | English
## Dataset Introduction
This dataset is used for quick document onboarding, and it截取s the first 8 entries from the GSM8K dataset for testing purposes.

> 🔗 Dataset Homepage: [https://github.com/openai/grade-school-math](https://github.com/openai/grade-school-math)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip
unzip gsm8k.zip
rm gsm8k.zip
```
- Execute `tree gsm8k/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    gsm8k/
    ├── test.jsonl
    ├── test_socratic.jsonl
    ├── train.jsonl
    └── train_socratic.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Import Statement | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- | --- |
| demo_gsm8k_gen_4_shot_cot_chat_prompt | Generative task for the GSM8K dataset (only 8 entries used) with logical chain | Accuracy | 4-shot | String format | `from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets` | [demo_gsm8k_gen_4_shot_cot_chat_prompt.py](demo_gsm8k_gen_4_shot_cot_chat_prompt.py) |
| demo_gsm8k_gen_0_shot_cot_str_perf | Generative task for the GSM8K dataset (only 8 entries used) with logical chain | Performance Evaluation | 0-shot | String format | `from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_0_shot_cot_str_perf import gsm8k_datasets as datasets` | [demo_gsm8k_gen_0_shot_cot_str_perf.py](demo_gsm8k_gen_0_shot_cot_str_perf.py) |


### Translation Notes
1. **Term Consistency**: Technical terms such as "生成式任务" (generative task), "逻辑链" (logical chain), and "性能评测" (performance evaluation) follow consistent English expressions in AI dataset documentation, ensuring clarity for technical users.
2. **Path & Code Preservation**: Linux commands (e.g., `cd`, `wget`), directory paths (e.g., `{tool_root_path}/ais_bench/datasets`), and filenames (e.g., `test.jsonl`, `demo_gsm8k_gen_0_shot_cot_str_perf.py`) are retained exactly as original to avoid disrupting deployment workflows.
3. **Acronym Clarity**: "GSM8K" (Grade School Math 8K) is kept as the standard acronym for the dataset (widely recognized in mathematical reasoning tasks), and "OpenCompass" (platform name) remains unchanged for brand consistency.
4. **Contextual Accuracy**: The phrase "用于文档快速入门使用" is translated as "used for quick document onboarding" to convey the dataset’s purpose of facilitating easy initial setup, while "只取8条数据" is rendered as "only 8 entries used" to accurately reflect the limited sample size for demo purposes.