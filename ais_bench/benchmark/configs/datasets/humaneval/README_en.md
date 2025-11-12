# HumanEval
[中文](README.md) | English
## Dataset Introduction
The HumanEval dataset released by OpenAI contains 164 programming problems, each providing a function signature, docstring, function body, and multiple unit tests. These problems are all manually written to ensure they do not appear in the training sets of code generation models.

> 🔗 Dataset Homepage: [https://huggingface.co/datasets/openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval)

⏰ **Note**: Please install dependencies from [extra.txt](../../../../../requirements/extra.txt) before running the dataset.
```shell
# You need to be in the outermost benchmark folder and run the following command:
pip3 install -r requirements/extra.txt
```

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humaneval.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humaneval.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humaneval.zip
unzip humaneval.zip
rm humaneval.zip
```
- Execute `tree humaneval/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    humaneval
    └── human-eval-v2-20210705.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| humaneval_gen_0_shot | Generative task for the HumanEval dataset | pass@1 | 0-shot | String format | [humaneval_gen_0_shot.py](humaneval_gen_0_shot.py) |