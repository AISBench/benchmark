# IFEval
[中文](README.md) | English
## Dataset Introduction
IFEval is a dataset designed to evaluate the instruction-following capabilities of large language models (such as GPT-4, PaLM 2, etc.). With the widespread application of large language models in natural language tasks, the instruction-following capability of models has become a crucial evaluation metric.

> 🔗 Dataset Homepage: [https://huggingface.co/datasets/google/IFEval](https://huggingface.co/datasets/google/IFEval)

⏰ **Note**: Please install dependencies from [extra.txt](../../../../../requirements/extra.txt) before running the dataset.
```shell
# You need to be in the outermost "benchmark" folder and run the following command:
pip3 install -r requirements/extra.txt
```

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ifeval.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ifeval.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ifeval.zip
unzip ifeval.zip
rm ifeval.zip
```
- Execute `tree ifeval/` (Note: The original text uses "feval/" which is corrected to "ifeval/" for consistency with the dataset name) in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    ifeval
    └── input_data.jsonl
    ```

## Available Dataset Tasks
### ifeval_0_shot_gen_str
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| ifeval_0_shot_gen_str | Generative task for the IFEval dataset | Accuracy | 0-shot | String format | [ifeval_0_shot_gen_str.py](ifeval_0_shot_gen_str.py) |