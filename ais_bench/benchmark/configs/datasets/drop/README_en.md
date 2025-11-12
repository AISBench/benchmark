# DROP
[中文](README.md) | English
## Dataset Introduction
DROP is a benchmark consisting of 96,000 questions created through crowdsourcing and adversarial methods. In this benchmark, systems must parse references in questions (which may involve multiple input positions) and perform discrete operations on these references (such as addition, counting, or sorting). These operations require a more comprehensive and in-depth understanding of paragraph content compared to previous datasets.

> 🔗 Dataset Homepage: [https://huggingface.co/datasets/ucinlp/drop](https://huggingface.co/datasets/ucinlp/drop)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/drop_simple_eval.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/drop_simple_eval.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/drop_simple_eval.zip
unzip drop_simple_eval.zip
rm drop_simple_eval.zip
```
- Execute `tree drop_simple_eval/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    drop_simple_eval
    └── dev.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| drop_gen_0_shot_str | Generative task for the DROP dataset | Accuracy (pass@1) | 0-shot | String format | [drop_gen_0_shot_str.py](drop_gen_0_shot_str.py) |
| drop_gen_3_shot_str | Generative task for the DROP dataset | Accuracy (pass@1) | 3-shot | String format | [drop_gen_3_shot_str.py](drop_gen_3_shot_str.py) |