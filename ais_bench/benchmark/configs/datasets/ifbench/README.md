# IFBench
中文 | [English](README_en.md)

## 数据集简介

IFBench 是一个用于评估 AI 模型在遵循新颖、具有挑战性且多样化的可验证指令方面可靠性的基准测试，特别强调模型在域外（out-of-domain）的泛化能力。该基准由 AllenAI 开发，旨在解决现有基准中存在的过拟合和数据污染问题。

> 🔗 数据集主页链接: [https://huggingface.co/datasets/allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test)
> 
> 🔗 官方 GitHub 仓库: [https://github.com/allenai/IFBench](https://github.com/allenai/IFBench)


## 数据集部署

- 可以从 Hugging Face 的数据集链接 🔗 [https://huggingface.co/datasets/allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test) 中获取数据集。
- IFBench 数据集为 Parquet 格式，建议部署在 `{tool_root_path}/ais_bench/datasets/ifbench/data/` 目录下。

- 在 `{tool_root_path}/ais_bench/datasets/ifbench/data/` 目录下执行 `ls -la` 检查目录结构。如果目录结构如下所示，则数据集部署成功：
    ```
    {tool_root_path}/ais_bench/datasets/ifbench/data/
    └── train-00000-of-00001.parquet
    ```


## 可用数据集任务

| 任务名称 | 简介 | 评估指标 | Few-Shot | Prompt 格式 | 对应源码配置文件路径 |
| --- | --- | --- | --- | --- | --- |
| ifbench_0_shot_gen_str | IFBench 数据集 | prompt_level_strict, inst_level_strict, prompt_level_loose, inst_level_loose | 0-shot | 字符串格式 | ifbench_0_shot_gen_str.py |

---

## ⚠️ 模型配置注意事项

若使用此数据集，需要在 `vllm_api_general_chat.py` 中将 `pred_postprocessor` 后处理函数修改为 `extract_non_reasoning_content_raw`：

```python
# 修改前
pred_postprocessor=dict(type=extract_non_reasoning_content),

# 修改后
pred_postprocessor=dict(type=extract_non_reasoning_content_raw),
```

此修改是为了在 IFBench 评测中正确处理模型的原始输出，避免误移除推理过程（reasoning）内容影响评估结果。

