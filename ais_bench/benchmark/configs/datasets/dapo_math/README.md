# DAPO-math-17k
中文 | [English](README_en.md)

## 数据集简介
DAPO-math-17k 是一个包含约 17,000 道数学题的数据集，主要用于强化学习（RL）推理评估场景。该数据集包含数学问题及其标准答案，适用于评估模型在数学推理任务上的表现。

数据集采用 Parquet 格式存储，每个样本包含：
- **prompt**: 数学问题的提示内容（对话格式）
- **answer**: 标准答案（从 reward_model 的 ground_truth 字段提取）
- **ability**: 能力标签（默认为 "MATH"）
- **data_source**: 数据来源标识（默认为 "math_dapo"）

## 数据集部署
- 可以从 HuggingFace 提供的链接下载数据集 🔗: [https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- 数据集文件应为 Parquet 格式（`.parquet` 文件）
- 建议部署在 `{工具根路径}/ais_bench/datasets/dapo-math-17k/` 目录下（数据集任务中设置的默认路径），以 linux 上部署为例，具体执行步骤如下：

```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
git lfs install
git lfs clone https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k.git
mv DAPO-Math-17k dapo-math-17k
mv dapo-math-17k/data/dapo-math-17k.parquet dapo-math-17k/
rm -rf dapo-math-17k/data
```

- 在 `{工具根路径}/ais_bench/datasets` 目录下执行 `tree dapo-math-17k/` 查看目录结构，若目录结构如下所示，则说明数据集部署成功。
   ```
   dapo-math-17k
   ├── dapo-math-17k.parquet
   └── README.md
   ```

## 可用数据集任务
| 任务名称 | 简介 | 评估指标 | Few-Shot | Prompt 格式 | 配套文件导入方式 | 对应源码配置文件路径 |
| --- | --- | --- | --- | --- | --- | --- |
| dapo_math_gen_0_shot_str | DAPO-math-17k 数据集生成式任务，使用 Minerva 方式提取答案 | accuracy | 0-shot | 字符串格式 | `from ais_bench.benchmark.configs.datasets.dapo_math.dapo_math_gen_0_shot_str import dapo_math_datasets as datasets` | [dapo_math_gen_0_shot_str.py](dapo_math_gen_0_shot_str.py) |
| dapo_math_gen_0_shot_cot_str | DAPO-math-17k 数据集生成式任务，使用严格 boxed 方式提取答案 | accuracy | 0-shot | 字符串格式 | `from ais_bench.benchmark.configs.datasets.dapo_math.dapo_math_gen_0_shot_cot_str import dapo_math_datasets as datasets` | [dapo_math_gen_0_shot_cot_str.py](dapo_math_gen_0_shot_cot_str.py) |

## 评估方式说明
数据集支持两种答案提取和评估方式：

1. **Minerva 方式** (`dapo_math_postprocess`): 
   - 从模型输出中提取 "Answer:" 后的内容
   - 对答案进行标准化处理（去除单位、格式化等）
   - 适用于一般的数学推理评估

2. **严格 boxed 方式** (`dapo_math_postprocess_v2`):
   - 从模型输出的最后部分提取 `\boxed{...}` 格式的答案
   - 要求答案以 LaTeX boxed 格式呈现
   - 适用于需要严格格式的评估场景

两种方式都会对答案进行标准化处理，包括去除空格、单位、LaTeX 格式转换等，以确保评估的准确性。

