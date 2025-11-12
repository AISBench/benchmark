# mbpp
[中文](README.md) | English
## Dataset Introduction
The mbpp benchmark contains approximately 1,000 crowdsourced Python programming problems, designed to be solvable by entry-level programmers. It covers topics such as programming fundamentals and standard library functions. Each problem includes a task description, a code solution, and 3 automated test cases. As stated in the paper, we have manually verified a portion of the data.

> 🔗 Dataset Homepage: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip
unzip mbpp.zip
rm mbpp.zip
```
- Execute `tree mbpp/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    mbpp
    ├── mbpp.jsonl
    └── sanitized-mbpp.jsonl
    ```

## Available Dataset Tasks
### mbpp_passk_gen_3_shot_chat_prompt
#### Basic Information
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| mbpp_passk_gen_3_shot_chat_prompt | Generative task for the mbpp dataset, supporting pass@k evaluation (default: pass@1) | pass@1 | 3-shot | Chat format | [mbpp_passk_gen_3_shot_chat_prompt.py](mbpp_passk_gen_3_shot_chat_prompt.py) |
| sanitized_mbpp_passk_gen_3_shot_chat_prompt | Generative task for the sanitized mbpp dataset, supporting pass@k evaluation (default: pass@1) | pass@1 | 3-shot | Chat format | [sanitized_mbpp_passk_gen_3_shot_chat_prompt.py](sanitized_mbpp_passk_gen_3_shot_chat_prompt.py) |