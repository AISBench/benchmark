# DAPO-math-17k
[中文](README.md) | English

## Dataset Introduction
DAPO-math-17k is a dataset containing approximately 17,000 math problems, primarily used for reinforcement learning (RL) reasoning evaluation scenarios. The dataset includes math problems and their standard answers, suitable for evaluating model performance on mathematical reasoning tasks.

The dataset is stored in Parquet format, with each sample containing:
- **prompt**: The prompt content of the math problem (in dialogue format)
- **answer**: The standard answer (extracted from the `ground_truth` field in `reward_model`)
- **ability**: Ability label (default: "MATH")
- **data_source**: Data source identifier (default: "math_dapo")

## Dataset Deployment
- You can download the dataset from the HuggingFace link 🔗: [https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- The dataset file should be in Parquet format (`.parquet` file)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets/dapo-math-17k/` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:

```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
git lfs install
git lfs clone https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k.git
mv DAPO-Math-17k dapo-math-17k
mv dapo-math-17k/data/dapo-math-17k.parquet dapo-math-17k/
rm -rf dapo-math-17k/data
```

- Execute `tree dapo-math-17k/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    dapo-math-17k
    ├── dapo-math-17k.parquet
    └── README.md
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| dapo_math_gen_0_shot_str | Generative task for DAPO-math-17k dataset, using Minerva method to extract answers | accuracy | 0-shot | String Format | [dapo_math_gen_0_shot_str.py](dapo_math_gen_0_shot_str.py) |
| dapo_math_gen_0_shot_cot_str | Generative task for DAPO-math-17k dataset, using strict boxed method to extract answers | accuracy | 0-shot | String Format | [dapo_math_gen_0_shot_cot_str.py](dapo_math_gen_0_shot_cot_str.py) |

## Evaluation Method Description
The dataset supports two answer extraction and evaluation methods:

1. **Minerva Method** (`dapo_math_postprocess`):
   - Extracts content after "Answer:" from model output
   - Normalizes the answer (removes units, formatting, etc.)
   - Suitable for general mathematical reasoning evaluation

2. **Strict Boxed Method** (`dapo_math_postprocess_v2`):
   - Extracts answers in `\boxed{...}` format from the last part of model output
   - Requires answers to be presented in LaTeX boxed format
   - Suitable for evaluation scenarios requiring strict formatting

Both methods normalize answers, including removing spaces, units, LaTeX format conversion, etc., to ensure evaluation accuracy.

