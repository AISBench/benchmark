# IFBench
[中文](README.md) | English

## Dataset Introduction

IFBench is a benchmark for evaluating the reliability of AI models in following novel, challenging, and diverse verifiable instructions, with particular emphasis on out-of-domain generalization. The benchmark was developed by AllenAI to address overfitting and data contamination issues present in existing benchmarks.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test)
> 
> 🔗 Official GitHub Repository: [https://github.com/allenai/IFBench](https://github.com/allenai/IFBench)


## Dataset Deployment

- The dataset can be obtained from the Hugging Face dataset link: 🔗 [https://huggingface.co/datasets/allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test).
- The IFBench dataset is in Parquet format and is recommended to be deployed in the `{tool_root_path}/ais_bench/datasets/ifbench/data/` directory.

- Execute `ls -la` in the `{tool_root_path}/ais_bench/datasets/ifbench/data/` directory to check the directory structure. If the directory structure is as shown below, the dataset has been deployed successfully:
    ```
    {tool_root_path}/ais_bench/datasets/ifbench/data/
    └── train-00000-of-00001.parquet
    ```


## Available Dataset Tasks

| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
| ifbench_0_shot_gen_str | IFBench dataset | prompt_level_strict, inst_level_strict, prompt_level_loose, inst_level_loose | 0-shot | String format | ifbench_0_shot_gen_str.py |

---

## ⚠️ Model Configuration Notes

When using this dataset, the `pred_postprocessor` in `vllm_api_general_chat.py` must be changed to `extract_non_reasoning_content_raw`:

```python
# Before
pred_postprocessor=dict(type=extract_non_reasoning_content),

# After
pred_postprocessor=dict(type=extract_non_reasoning_content_raw),
```

This change ensures that IFBench evaluates the model's raw output correctly, preventing the removal of reasoning content from affecting evaluation results.

