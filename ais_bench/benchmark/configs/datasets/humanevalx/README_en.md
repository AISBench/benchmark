# HumanEvalX
[中文](README.md) | English
## Dataset Introduction
HumanEval-X is an evaluation benchmark for multilingual code generation models provided by the THUDM (Tsinghua University Department of Computer Science and Technology, Knowledge Engineering Group) KEG Laboratory of Tsinghua University. It contains 820 high-quality handwritten samples, covering the Python, C++, Java, JavaScript, and Go programming languages.

> 🔗 Dataset Homepage: [https://huggingface.co/datasets/THUDM/humaneval-x](https://huggingface.co/datasets/THUDM/humaneval-x)

⏰ **Note**: Please install dependencies from [extra.txt](../../../../../requirements/extra.txt) before running the dataset.
```shell
# You need to be in the outermost "benchmark" folder and run the following command:
pip3 install -r requirements/extra.txt
```

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humanevalx.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humanevalx.zip).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humanevalx.zip
unzip humanevalx.zip
rm humanevalx.zip
```
- Execute `tree humanevalx/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    humanevalx
    ├── humanevalx_cpp.jsonl
    ├── humanevalx_go.jsonl
    ├── humanevalx_java.jsonl
    ├── humanevalx_js.jsonl
    └── humanevalx_python.jsonl
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Import Statement | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- | --- |
| humanevalx_gen_0_shot | Generative task for the HumanEvalX dataset | pass@1 | 0-shot | String format |`from ais_bench.benchmark.configs.datasets.humanevalx.humanevalx_gen_0_shot import humanevalx_datasets as datasets`| [humanevalx_gen_0_shot.py](humanevalx_gen_0_shot.py) |