# Herding Coreset Selector

## 简介

Herding Coreset Selector 是一个用于评测数据集代表性样本筛选的独立 Coreset 工具。

工具使用指定语言模型提取样本 Prompt 的隐藏状态特征，并基于 RBF Kernel 的 Kernel Herding 方法，从完整数据集中选择指定比例的代表性样本。生成结果保持原数据格式，同时保存样本在完整数据集中的索引，便于结果复现和追溯。

## 工作流程

```text
原始评测数据
    │
    ▼
构造样本 Prompt
    │
    ▼
语言模型提取隐藏状态特征
    │
    ▼
RBF Kernel
    │
    ▼
Kernel Herding 选择代表性样本
    │
    ├── origin/   完整数据及索引
    └── coreset/  压缩后的数据及索引
    │
    ▼
复制到 ais_bench/datasets/ 对应目录
```

# Coreset 使用方法

## 1. 环境准备

安装运行依赖：

```shell
pip install numpy torch transformers tqdm
```

如果数据集适配器复用了 AISBench 中的数据集或 Prompt 组件，需要保证当前环境可以导入 `ais_bench`。在 Benchmark 仓库中执行：

```shell
cd benchmark
pip install -e .
```

> AISBench 在这里主要用于提供数据集和 Prompt 相关组件；执行 Coreset 时不会启动 AISBench 模型评测流程。

## 2. 准备数据

数据集通过 `herding/eval_datasets/` 下的适配器接入。适配器负责：

1. 获取数据集样本数量；
2. 将样本转换为用于特征提取的 Prompt；
3. 根据筛选出的索引，以原数据格式保存 Coreset。

如果原始数据已经放在 AISBench 的数据目录：

```text
benchmark/ais_bench/datasets/
```

可以通过 `CORESET_BASE_DIR` 指定数据根目录：

```shell
cd benchmark
export CORESET_BASE_DIR=$(pwd)/ais_bench/datasets
```

也可以通过 `DATASET_PATH` 直接指定当前数据集目录：

```shell
export DATASET_PATH=$(pwd)/ais_bench/datasets/<dataset_name>
```

`DATASET_PATH` 的具体形式由对应的数据集适配器决定。

## 3. 配置参数

### 必选参数

| 环境变量 | 说明 | 示例 |
| --- | --- | --- |
| `EVAL_DATASET` | 数据集适配器注册名称 | `gpqa` |
| `CORESET_METRIC` | Coreset 方法名称，同时用于输出目录命名 | `herding` |
| `LLM_MODEL` | 特征模型标识名称，用于组织输出目录 | `qwen25_7b` |
| `CORESET_RATIO` | Coreset 占完整数据集的比例 | `0.2` |

### 模型与路径参数

| 环境变量 | 说明 |
| --- | --- |
| `MODEL_PATH` | 用于提取隐藏状态特征的本地 Hugging Face 模型路径，建议显式设置 |
| `CORESET_MODEL_PATH` | 特征模型路径的备用配置；未设置 `MODEL_PATH` 时使用 |
| `CORESET_BASE_DIR` | 原始数据的基础目录 |
| `DATASET_PATH` | 当前数据集的具体路径，可覆盖适配器默认路径 |

通用配置示例：

```shell
export EVAL_DATASET=<dataset_name>
export CORESET_METRIC=herding
export CORESET_RATIO=0.2
export LLM_MODEL=<model_name>
export MODEL_PATH=/path/to/model
export DATASET_PATH=/path/to/datasets/<dataset_name>
```

其中：

```text
CORESET_RATIO=0.2
```

表示从完整数据集中选择约 20% 的样本作为 Coreset。

## 4. 运行 Coreset 压缩

进入工具目录：

```shell
cd tools/herding_coreset_selector
```

执行：

```shell
python -m herding
```

工具会依次完成：

```text
读取数据
  ↓
构造 Prompt
  ↓
特征模型提取隐藏状态
  ↓
计算 RBF Kernel
  ↓
Kernel Herding 选择样本
  ↓
保存 origin 与 coreset
```

## 5. Coreset 输出

从 `tools/herding_coreset_selector` 目录运行时，结果默认写入：

```text
datasets/
└── <EVAL_DATASET>/
    └── <CORESET_METRIC>/
        └── <LLM_MODEL>/
            ├── origin/
            │   ├── <dataset_file>
            │   └── indices.json
            └── coreset/
                ├── <dataset_file>
                └── indices.json
```

其中：

- `origin/<dataset_file>`：本次筛选对应的完整数据；
- `origin/indices.json`：完整数据对应的原始索引；
- `coreset/<dataset_file>`：压缩后筛选出的 Coreset 数据；
- `coreset/indices.json`：Coreset 样本在完整数据中的原始索引。

实际需要保留的压缩数据位于：

```text
coreset/<dataset_file>
```

建议同时保留 `coreset/indices.json`，方便后续追溯样本来源。

## 6. 将压缩结果保存到 AISBench 数据目录

推荐不要覆盖原始完整数据，而是在 `ais_bench/datasets/` 下为 Coreset 建立独立目录：

```text
benchmark/ais_bench/datasets/
├── <dataset_name>/
│   └── <original_dataset_file>
└── <dataset_name>_coreset/
    ├── <dataset_file>
    └── indices.json
```

例如：

```shell
cd benchmark

mkdir -p ais_bench/datasets/<dataset_name>_coreset

cp \
  tools/herding_coreset_selector/datasets/<dataset_name>/herding/<model_name>/coreset/<dataset_file> \
  ais_bench/datasets/<dataset_name>_coreset/<dataset_file>

cp \
  tools/herding_coreset_selector/datasets/<dataset_name>/herding/<model_name>/coreset/indices.json \
  ais_bench/datasets/<dataset_name>_coreset/indices.json
```

这样原始数据和压缩后的 Coreset 会分别保存在 AISBench 数据目录中，互不覆盖。

# GPQA 压缩示例

下面给出已经验证可以运行的 GPQA Coreset 压缩流程。

## 1. 准备 GPQA 数据

将 GPQA 数据放到：

```text
benchmark/
└── ais_bench/
    └── datasets/
        └── gpqa/
            ├── gpqa_diamond.csv
            ├── gpqa_main.csv
            ├── gpqa_extended.csv
            └── ...
```

当前使用 `gpqa_diamond.csv` 进行 Coreset 压缩。

运行前可以先确认文件存在：

```shell
cd benchmark
ls -lh ais_bench/datasets/gpqa/gpqa_diamond.csv
```

## 2. 配置 GPQA 压缩参数

```shell
cd benchmark

export EVAL_DATASET=gpqa
export CORESET_METRIC=herding
export CORESET_RATIO=0.2

# 用于输出目录命名
export LLM_MODEL=qwen25_7b

# 替换为实际的本地 Hugging Face 模型路径
export MODEL_PATH=/path/to/Qwen2.5-7B-Instruct

# GPQA 数据目录
export DATASET_PATH=$(pwd)/ais_bench/datasets/gpqa
```

参数说明：

- `EVAL_DATASET=gpqa`：加载 GPQA 数据集适配器；
- `CORESET_METRIC=herding`：使用 Kernel Herding 进行样本筛选；
- `CORESET_RATIO=0.2`：选择约 20% 的 GPQA 样本；
- `LLM_MODEL=qwen25_7b`：用于组织输出目录；
- `MODEL_PATH`：真正用于提取 Prompt 隐藏状态特征的模型路径；
- `DATASET_PATH`：包含 `gpqa_diamond.csv` 的 GPQA 数据目录。

## 3. 运行 GPQA 压缩

```shell
cd tools/herding_coreset_selector
python -m herding
```

运行流程为：

```text
gpqa_diamond.csv
        │
        ▼
构造 GPQA Prompt
        │
        ▼
特征模型提取隐藏状态
        │
        ▼
RBF Kernel + Kernel Herding
        │
        ▼
按 CORESET_RATIO 选择代表性样本
        │
        ├── origin/gpqa_diamond.csv
        └── coreset/gpqa_diamond.csv
```

## 4. 检查 GPQA 输出

如果设置：

```shell
export LLM_MODEL=qwen25_7b
```

则输出目录为：

```text
benchmark/tools/herding_coreset_selector/datasets/
└── gpqa/
    └── herding/
        └── qwen25_7b/
            ├── origin/
            │   ├── gpqa_diamond.csv
            │   └── indices.json
            └── coreset/
                ├── gpqa_diamond.csv
                └── indices.json
```

可以执行：

```shell
cd benchmark/tools/herding_coreset_selector

ls -lh datasets/gpqa/herding/qwen25_7b/origin/
ls -lh datasets/gpqa/herding/qwen25_7b/coreset/
```

其中：

```text
datasets/gpqa/herding/qwen25_7b/coreset/gpqa_diamond.csv
```

就是压缩后的 GPQA 数据集。

## 5. 保存 GPQA Coreset 到 AISBench

推荐将压缩后的 GPQA 单独保存到：

```text
benchmark/ais_bench/datasets/gpqa_coreset/
```

执行：

```shell
cd benchmark

mkdir -p ais_bench/datasets/gpqa_coreset

cp \
  tools/herding_coreset_selector/datasets/gpqa/herding/qwen25_7b/coreset/gpqa_diamond.csv \
  ais_bench/datasets/gpqa_coreset/gpqa_diamond.csv

cp \
  tools/herding_coreset_selector/datasets/gpqa/herding/qwen25_7b/coreset/indices.json \
  ais_bench/datasets/gpqa_coreset/indices.json
```

保存后的目录结构为：

```text
benchmark/ais_bench/datasets/
├── gpqa/
│   └── gpqa_diamond.csv
└── gpqa_coreset/
    ├── gpqa_diamond.csv
    └── indices.json
```

这样原始 GPQA 和压缩后的 GPQA Coreset 都保留在 AISBench 数据目录中，且不会相互覆盖。

## 6. 修改压缩比例

例如只保留约 10% 的样本：

```shell
export CORESET_RATIO=0.1
python -m herding
```

如果需要比较多个压缩比例，建议在每次运行后保存或重命名输出目录，避免不同实验结果相互覆盖。

# 接入新的数据集

如果需要处理新的数据格式，可以在 `herding/eval_datasets/` 中增加数据集适配器。

适配器继承 `EvalDatasetBase` 并实现：

```python
class MyDataset(EvalDatasetBase):
    def dataset_size(self):
        ...

    def dataset_prompts(self):
        ...

    def save_data_by_indices(self, indices, outpath):
        ...
```

然后使用 `reg_eval_dataset` 注册名称：

```python
@reg_eval_dataset("my_dataset")
class MyDataset(EvalDatasetBase):
    ...
```

并在 `herding/eval_datasets/__init__.py` 中加入对应模块的导入逻辑。之后即可通过：

```shell
export EVAL_DATASET=my_dataset
```

加载该数据集。

建议 `save_data_by_indices()` 保持原始数据格式不变，以便压缩结果可以直接保存到 AISBench 对应的数据目录。
