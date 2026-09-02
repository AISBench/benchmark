# Geometry3K
中文 | [English](README_en.md)

## 数据集简介

Geometry3K 是一个多模态几何推理数据集，包含约 3,002 道几何数学题（其中 test 划分 601 道），每道题由几何图形（图片）和对应的文字问题组成，要求模型根据图形信息进行几何推理并给出答案。该数据集主要用于评估多模态大语言模型在数学几何推理方面的能力。

> 🔗 数据集主页链接: [https://huggingface.co/datasets/hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k)


## 数据集部署

- 可以从 Hugging Face 的数据集链接 🔗 [https://huggingface.co/datasets/hiyouga/geometry3k](https://huggingface.co/datasets/hiyouga/geometry3k) 中获取数据集。
- Geometry3K 数据集为 Parquet 格式，建议部署在 `{tool_root_path}/ais_bench/datasets/geometry3k/` 目录下。

- 在 `{tool_root_path}/ais_bench/datasets/` 目录下执行 `tree geometry3k/` 检查目录结构。如果目录结构如下所示，则数据集部署成功：
    ```
    geometry3k/
    └── data/
        └── test-00000-of-00001.parquet
    ```


## 可用数据集任务

| 任务名称 | 简介 | 评估指标 | Few-Shot | Prompt 格式 | 对应源码配置文件路径 |
| --- | --- | --- | --- | --- | --- |
| geometry3k_gen | Geometry3K 几何推理数据集 | accuracy | 0-shot | 多模态格式 | geometry3k_gen.py |

---

## ⚠️ 数据集评测准备事项

### 1. 安装依赖

Geometry3K 数据集的依赖文件位于 `requirements/datasets/geometry3k.txt`，内容如下：

- `mathruler`
- `pylatexenc`

使用以下命令安装：

```shell
pip install -r requirements/datasets/geometry3k.txt
```

各依赖包的作用：

- `mathruler`：用于数学答案的提取（`\boxed{}` 内容）和 sympy 等价性判定。
- `pylatexenc`：用于 LaTeX 数学表达式的解析和标准化，确保答案比对的一致性。

### 2. 模型配置修改

若使用此数据集，需要在 `vllm_api_general_chat.py` 中同时修改引入语句和 `pred_postprocessor` 后处理函数：

```python
# 修改前
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
...
pred_postprocessor=dict(type=extract_non_reasoning_content),

# 修改后
from ais_bench.benchmark.utils.postprocess.model_postprocessors import keep_reasoning_content
...
pred_postprocessor=dict(type=keep_reasoning_content),
```

此修改是为了保留模型输出中的 `<think>...</think>` 推理标签，确保 `format_reward` 格式评分和 `\boxed{}` 答案提取能够正常工作。`keep_reasoning_content` 是一个透传后处理函数，不会移除模型输出的任何内容。

