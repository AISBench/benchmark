# GSM8K
[中文](README.md) | English
## Dataset Introduction
The GSM8K dataset consists of 8,500 high-quality elementary school mathematics problems written by human problem setters. We divide these problems into 7,500 training problems and 1,000 test problems. Solving these problems requires 2 to 8 steps, and the main solution method involves performing a series of basic calculations using fundamental arithmetic operations (addition, subtraction, division, multiplication) to derive the final answer. A competent middle school student should be able to solve every problem in this dataset.

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
| gsm8k_gen_4_shot_cot_str | Generative task for the GSM8K dataset with logical chain | Accuracy | 4-shot | String format |`from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_str import gsm8k_datasets as datasets`| [gsm8k_gen_4_shot_cot_str.py](gsm8k_gen_4_shot_cot_str.py) |
| gsm8k_gen_4_shot_cot_chat_prompt | Generative task for the GSM8K dataset with logical chain | Accuracy | 4-shot | Chat format |`from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets`| [gsm8k_gen_4_shot_cot_chat_prompt.py](gsm8k_gen_4_shot_cot_chat_prompt.py) |
| gsm8k_gen_0_shot_cot_str | Generative task for the GSM8K dataset | Accuracy | 0-shot | String format |`from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as datasets`| [gsm8k_gen_0_shot_cot_str.py](gsm8k_gen_0_shot_cot_str.py) |
| gsm8k_gen_0_shot_cot_chat_prompt | Generative task for the GSM8K dataset | Accuracy | 0-shot | Chat format |`from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_chat_prompt import gsm8k_datasets as datasets`| [gsm8k_gen_0_shot_cot_chat_prompt.py](gsm8k_gen_0_shot_cot_chat_prompt.py) |
| gsm8k_gen_0_shot_cot_str_perf | Generative task for the GSM8K dataset (for performance evaluation) | Performance Evaluation | 0-shot | String format |`from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str_perf import gsm8k_datasets as datasets`| [gsm8k_gen_0_shot_cot_str_perf.py](gsm8k_gen_0_shot_cot_str_perf.py) |