# Herding Coreset Selector

## 简介

Herding Coreset Selector 是一个用于评测数据集代表性样本筛选的独立 Coreset 工具。

工具使用指定语言模型提取样本 Prompt 的隐藏状态特征，并基于 RBF Kernel 的 Kernel Herding 方法，从完整数据集中选择指定比例的代表性样本。生成结果保持原数据格式，同时保存样本在完整数据集中的索引，便于结果复现和追溯。

## 1. 环境准备

安装运行依赖：

```shell
pip install numpy torch transformers tqdm
```

数据集适配器会复用 AISBench 中的数据集或 Prompt 组件，因此需要保证当前环境可以导入 `ais_bench`。在 Benchmark 仓库中执行：

```shell
cd benchmark
pip install -e .
```

## 2. 命令行帮助

进入工具目录：

```shell
cd benchmark/tools/herding_coreset_selector
```

查看完整参数说明：

```shell
python -m herding --help
```

主要参数：

| 参数 | 是否必选 | 说明 |
| --- | --- | --- |
| `--eval-dataset` | 是 | 数据集适配器名称，当前支持 `gpqa`、`aime2025` |
| `--dataset-path` | 是 | 原始评测数据所在目录 |
| `--model-path` | 是 | 用于提取隐藏状态特征的本地 Hugging Face 模型路径 |
| `--coreset-ratio` | 否 | Coreset 占完整数据集的比例，范围 `(0, 1]`，默认 `0.2` |
| `--output-dir` | 否 | 结果输出根目录，默认 `./datasets` |

工具运行时配置通过命令行显式传入，不依赖环境变量。

## 3. 运行 Coreset 压缩

通用命令：

```shell
python -m herding \
  --eval-dataset <dataset_name> \
  --dataset-path /path/to/datasets/<dataset_name> \
  --model-path /path/to/model \
  --coreset-ratio 0.2
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

## 4. 输出目录

默认输出结构为：

```text
datasets/
└── <eval_dataset>/
    └── herding/
        └── <model_name>/
            ├── origin/
            │   ├── <dataset_file>
            │   └── indices.json
            └── coreset/
                ├── <dataset_file>
                └── indices.json
```

其中 `<model_name>` 会从 `--model-path` 的最后一级目录名自动获取。

- `origin/<dataset_file>`：本次筛选对应的完整数据；
- `origin/indices.json`：完整数据对应的原始索引；
- `coreset/<dataset_file>`：筛选后的 Coreset；
- `coreset/indices.json`：Coreset 样本在完整数据中的原始索引。

## 5. GPQA 压缩示例

准备数据：

```text
benchmark/ais_bench/datasets/gpqa/gpqa_diamond.csv
```

执行：

```shell
cd benchmark/tools/herding_coreset_selector

python -m herding \
  --eval-dataset gpqa \
  --dataset-path ../../ais_bench/datasets/gpqa \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --coreset-ratio 0.2
```

如果模型目录名为 `Qwen2.5-7B-Instruct`，则结果写入：

```text
datasets/gpqa/herding/Qwen2.5-7B-Instruct/
├── origin/
│   ├── gpqa_diamond.csv
│   └── indices.json
└── coreset/
    ├── gpqa_diamond.csv
    └── indices.json
```

例如只保留约 10% 的样本：

```shell
python -m herding \
  --eval-dataset gpqa \
  --dataset-path ../../ais_bench/datasets/gpqa \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --coreset-ratio 0.1
```

## 6. AIME2025 压缩示例

准备数据：

```text
benchmark/ais_bench/datasets/aime2025/aime2025.jsonl
```

执行：

```shell
python -m herding \
  --eval-dataset aime2025 \
  --dataset-path ../../ais_bench/datasets/aime2025 \
  --model-path /path/to/Qwen2.5-7B-Instruct \
  --coreset-ratio 0.2
```

## 7. 接入新的数据集

在 `herding/eval_datasets/` 中新增适配器，并继承 `EvalDatasetBase`：

```python
@reg_eval_dataset("my_dataset")
class MyDataset(EvalDatasetBase):
    def __init__(self, dataset_path, output_dir):
        super().__init__(dataset_path, output_dir)
        ...

    def dataset_size(self):
        ...

    def dataset_prompts(self):
        ...

    def save_data_by_indices(self, indices, outpath):
        ...
```

然后在 `herding/eval_datasets/__init__.py` 中导入该适配器模块以完成注册，并在 `herding/__main__.py` 的 `SUPPORTED_DATASETS` 中增加对应名称。

建议 `save_data_by_indices()` 保持原始数据格式不变，以便压缩结果可以直接用于后续 Benchmark 测评。
