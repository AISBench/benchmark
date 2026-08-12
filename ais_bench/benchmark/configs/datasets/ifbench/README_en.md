# IFBench
[中文](README.md) | English

## Dataset Introduction

IFBench is a benchmark for evaluating the reliability of AI models in following novel, challenging, and diverse verifiable instructions, with particular emphasis on out-of-domain generalization. The benchmark was developed by AllenAI to address overfitting and data contamination issues present in existing benchmarks.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test)
> 
> 🔗 Official GitHub Repository: [https://github.com/allenai/IFBench](https://github.com/allenai/IFBench)


## Dataset Deployment

- The dataset can be obtained from the Hugging Face dataset link: 🔗 [https://huggingface.co/datasets/allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test).
- The IFBench dataset is in Parquet format and is recommended to be deployed in the `{tool_root_path}/ais_bench/datasets/IFBench_test/data/` directory.

- Execute `tree IFBench_test/` in the `{tool_root_path}/ais_bench/datasets/` directory to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    IFBench_test/
    └── data/
        └── train-00000-of-00001.parquet
    ```


## Available Dataset Tasks

| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| ifbench_0_shot_gen_str | IFBench dataset | prompt_level_strict, inst_level_strict, prompt_level_loose, inst_level_loose | 0-shot | String format | ifbench_0_shot_gen_str.py |

---

## ⚠️ Dataset Evaluation Preparation

### 1. Install Dependencies

Read the dependency list from `requirements/datasets/IFBench_test.txt` and install the required dependencies:

```shell
pip install -r requirements/datasets/IFBench_test.txt
```

### 2. Download NLTK Data Package

Download the compressed package from [NLTK Data](https://github.com/nltk/nltk_data/archive/refs/heads/gh-pages.zip), place it in the container where ais_bench is running (recommended under the workspace directory), then execute the following commands:

```shell
unzip nltk_data-gh-pages.zip
mkdir nltk_data
mv nltk_data-gh-pages/packages/* nltk_data/
cd nltk_data
find . -name "*.zip" -exec sh -c 'unzip -o "$1" -d "$(dirname "$1")"' _ {} \;
```

After decompression, set the `NLTK_DATA` environment variable to point to the nltk_data directory. For example, if nltk_data is located at `/workspace/nltk_data` in the container, execute in the terminal:

```shell
export NLTK_DATA=/workspace/nltk_data
```

### 3. Model Configuration Modification

When using this dataset, both the import statement and `pred_postprocessor` in `vllm_api_general_chat.py` must be changed to `extract_non_reasoning_content_raw`:

```python
# Before
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
...
pred_postprocessor=dict(type=extract_non_reasoning_content),

# After
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content_raw
...
pred_postprocessor=dict(type=extract_non_reasoning_content_raw),
```

This change ensures that IFBench evaluates the model's raw output correctly, preventing the removal of reasoning content from affecting evaluation results.

