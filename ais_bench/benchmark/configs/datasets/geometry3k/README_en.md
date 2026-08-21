# Geometry3K
[中文](README.md) | English

## Dataset Introduction

Geometry3K is a multimodal geometry reasoning dataset containing approximately 3,002 geometry math problems (with 601 in the test split). Each problem consists of a geometric figure (image) and a corresponding text question, requiring the model to perform geometry reasoning based on the visual information and provide an answer. This dataset is primarily used to evaluate the mathematical geometry reasoning capabilities of multimodal large language models.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)


## Dataset Deployment

- The dataset can be obtained from the Hugging Face dataset link: 🔗 [https://huggingface.co/datasets/hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k).
- The Geometry3K dataset is in Parquet format and is recommended to be deployed in the `{tool_root_path}/ais_bench/datasets/geometry3k/` directory.

- Execute `tree geometry3k/` in the `{tool_root_path}/ais_bench/datasets/` directory to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    geometry3k/
    └── data/
        └── test-00000-of-00001.parquet
    ```


## Available Dataset Tasks

| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| geometry3k_gen | Geometry3K geometry reasoning dataset | accuracy | 0-shot | Multimodal format | geometry3k_gen.py |

---

## ⚠️ Dataset Evaluation Preparation

### 1. Install Dependencies

The dependency file for the Geometry3K dataset is located at `requirements/datasets/geometry3k.txt`, which contains:

- `mathruler`
- `pylatexenc`

Install via the requirements file:

```shell
pip install -r requirements/datasets/geometry3k.txt
```

What each dependency does:

- `mathruler`: Used for extracting mathematical answers (`\boxed{}` content) and sympy-based equivalence checking.
- `pylatexenc`: Used for parsing and normalizing LaTeX mathematical expressions to ensure consistent answer comparison.

### 2. Model Configuration Modification

When using this dataset, both the import statement and `pred_postprocessor` in `vllm_api_general_chat.py` must be changed to `keep_reasoning_content`:

```python
# Before
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
...
pred_postprocessor=dict(type=extract_non_reasoning_content),

# After
from ais_bench.benchmark.utils.postprocess.model_postprocessors import keep_reasoning_content
...
pred_postprocessor=dict(type=keep_reasoning_content),
```

This change preserves the `<think>...</think>` reasoning tags in the model output, ensuring that both the `format_reward` format scoring and `\boxed{}` answer extraction work correctly. `keep_reasoning_content` is a pass-through postprocessor that does not remove any content from the model output.

