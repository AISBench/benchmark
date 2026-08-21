# IFBench
中文 | [English](README_en.md)

## 数据集简介

IFBench 是一个用于评估 AI 模型在遵循新颖、具有挑战性且多样化的可验证指令方面可靠性的基准测试，特别强调模型在域外（out-of-domain）的泛化能力。该基准由 AllenAI 开发，旨在解决现有基准中存在的过拟合和数据污染问题。

> 🔗 数据集主页链接: [https://huggingface.co/datasets/allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test)
> 
> 🔗 官方 GitHub 仓库: [https://github.com/allenai/IFBench](https://github.com/allenai/IFBench)


## 数据集部署

- 可以从 Hugging Face 的数据集链接 🔗 [https://huggingface.co/datasets/allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test) 中获取数据集。
- IFBench 数据集为 Parquet 格式，建议部署在 `{tool_root_path}/ais_bench/datasets/IFBench_test/data/` 目录下。

- 在 `{tool_root_path}/ais_bench/datasets/` 目录下执行 `tree IFBench_test/` 检查目录结构。如果目录结构如下所示，则数据集部署成功：
    ```
    IFBench_test/
    └── data/
        └── train-00000-of-00001.parquet
    ```


## 可用数据集任务

| 任务名称 | 简介 | 评估指标 | Few-Shot | Prompt 格式 | 对应源码配置文件路径 |
| --- | --- | --- | --- | --- | --- |
| ifbench_0_shot_gen_str | IFBench 数据集 | prompt_level_strict, inst_level_strict, prompt_level_loose, inst_level_loose | 0-shot | 字符串格式 | ifbench_0_shot_gen_str.py |

---

## ⚠️ 数据集评测准备事项

### 1. 安装依赖

下载 pyarrow 依赖：

```shell
pip install pyarrow==15.0.0
```

### 2. 下载 NLTK 数据包

从 [NLTK Data](https://github.com/nltk/nltk_data/archive/refs/heads/gh-pages.zip) 下载压缩包，放置到 ais_bench 启动的容器内（建议放在 workspace 目录下），然后执行以下命令：

```shell
unzip nltk_data-gh-pages.zip
mkdir nltk_data
mv nltk_data-gh-pages/packages/* nltk_data/
cd nltk_data
find . -name "*.zip" -exec sh -c 'unzip -o "$1" -d "$(dirname "$1")"' _ {} \;
```

解压完成后，设置环境变量 `NLTK_DATA` 指向 nltk_data 目录。例如 nltk_data 在容器内的路径为 `/workspace/nltk_data`，则在终端中执行：

```shell
export NLTK_DATA=/workspace/nltk_data
```

### 3. 模型配置修改

若使用此数据集，需要在 `vllm_api_general_chat.py` 中同时修改引入语句和 `pred_postprocessor` 后处理函数：

```python
# 修改前
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
...
pred_postprocessor=dict(type=extract_non_reasoning_content),

# 修改后
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content_raw
...
pred_postprocessor=dict(type=extract_non_reasoning_content_raw),
```

此修改是为了在 IFBench 评测中正确处理模型的原始输出，避免误移除推理过程（reasoning）内容影响评估结果。

