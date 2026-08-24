# 纯模型精度测评
在本地环境加载模型与数据集，通过统一推理流程比对输出与参考答案，评估模型固有准确率。自定义批量大小、序列长度等参数，适用于**Huggingface Transformers**推理框架。
## 测试准备
在执行服务化推理前，需要满足以下条件：

- 可用的模型权重：确保本地已有需测试的模型权重文件，开源权重可从🔗 [huggingface社区](https://huggingface.co/models)获取。
- 数据集任务准备：从📚 [开源数据集](../../get_started/datasets.md#开源数据集)中选择数据集，并且在数据集对应的"详细介绍"文档中选择要执行的数据集任务。参考选取的数据集任务对应的"详细介绍"文档准备好数据集文件，建议将开源数据集手动放置在默认目录 `ais_bench/datasets/`下，程序将在任务执行时自动加载数据集文件。
- 模型任务准备：从📚 [本地模型后端](../all_params/models.md#本地模型后端)中选择要执行的模型任务。

## 主要功能

纯模型精度测评场景下主要功能与服务化精度测评场景相似，但需要将模型任务替换为本地 HuggingFace 模型任务（如 [`HuggingFacewithChatTemplate`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/models/huggingface_chat_model.py) 或 [`HuggingFaceBaseModel`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/models/huggingface_base_model.py)）。

### 纯模型多任务测评

支持同时配置多个数据集任务，通过单次命令进行批量测评。完整样例请参考 [multi_task_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/multi_task_zh_cn.py)：

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import HuggingFacewithChatTemplate
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets
    from ais_bench.benchmark.configs.datasets.aime2024.aime2024_gen_0_shot_chat_prompt import aime2024_datasets

datasets = gsm8k_datasets + aime2024_datasets

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='hf-chat-model',
        path='THUDM/chatglm-6b', # 替换为实际的本地模型权重路径
        tokenizer_path='THUDM/chatglm-6b',
        # ...其余参数配置详见配置文件
    )
]
```

执行命令：

```bash
ais_bench ais_bench/configs/accuracy_benchmark_local/multi_task_zh_cn.py
```

#### 自定义模型-数据集配对（可选）

默认情况下，上述配置中 `models` 列表与 `datasets` 列表会自动按笛卡尔积组合，子任务数为模型数 × 数据集数（本例为 1 × 2 = 2 个）。若希望精确控制哪些模型与哪些数据集配对（例如只让该模型跑部分数据集），可在配置文件中通过 `model_dataset_combinations` 字段显式声明配对关系：

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import HuggingFacewithChatTemplate
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets
    from ais_bench.benchmark.configs.datasets.aime2024.aime2024_gen_0_shot_chat_prompt import aime2024_datasets

datasets = gsm8k_datasets + aime2024_datasets

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='hf-chat-model',
        path='THUDM/chatglm-6b', # 替换为实际的本地模型权重路径
        tokenizer_path='THUDM/chatglm-6b',
    )
]

# 关键：通过 model_dataset_combinations 精确控制配对
# 下例仅生成 1 个子任务（笛卡尔积会生成 2 个）：
#   - hf-chat-model + gsm8k
model_dataset_combinations = [
    dict(models=[models[0]], datasets=[datasets[0]]),
]
```

> ⚠️ **注意**：模型与数据集的唯一标识由 `abbr` 字段决定。同一配置文件中，相同 `abbr` 的模型或数据集重复出现的组合会被视为重复任务而被跳过。当通过 `.copy()` 等方式复用模型/数据集配置时，必须显式修改 `abbr` 以保证唯一性。详见 📚 [自定义模型与数据集组合](../../advanced_tutorials/run_custom_config.md#自定义模型与数据集组合)。

> 💡 详细使用方法也可参考[服务化精度多任务测评使用方法](accuracy_benchmark.md#多任务测评)。

### 纯模型多任务并行测评

支持通过 [`--max-num-workers`](../all_params/cli_args.md#公共参数) 命令行参数实现多任务并行。配置文件样例与[纯模型多任务测评](#纯模型多任务测评)完全一致，区别仅在执行命令。

执行命令（以 `max-num-workers 4` 为例）：

```bash
ais_bench ais_bench/configs/accuracy_benchmark_local/multi_task_zh_cn.py --max-num-workers 4
```

> ⚠️ 注意：纯模型精度测评多任务并行会占用不同GPU单元，并行任务所需的GPU单元应小于等于可使用的GPU总数。

> 💡 详细使用方法也可参考[服务化精度多任务并行测评使用方法](accuracy_benchmark.md#多任务并行测评)。

### 纯模型中断续测

在纯模型精度测评过程中，如遇任务中断，可通过 `--reuse` 参数指定任务时间戳目录，继续未完成的推理任务，实现断点续测。该功能无需重复运行全部任务，仅对未完成部分进行补充推理。

首次执行命令：

```bash
ais_bench ais_bench/configs/accuracy_benchmark_local/single_task_zh_cn.py
```

通过 `--reuse` 参数指定任务时间戳目录续推（`--reuse` 是公共参数，使用自定义配置文件时仍可通过命令行追加）：

```bash
ais_bench ais_bench/configs/accuracy_benchmark_local/single_task_zh_cn.py --reuse 20250628_151326
```

> ⚠️ 注意，纯模型精度测评当前不支持失败用例自动重测。

> 💡 详细使用方法也可参考[服务化精度中断续测使用方法](accuracy_benchmark.md#中断续测--失败用例重测)。

### 纯模型合并子数据集推理

支持将存在多个小规模数据集的数据集合并为一个任务进行统一测评。完整样例请参考 [ceval_merge_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/ceval_merge_zh_cn.py)：

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import HuggingFacewithChatTemplate
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_5_shot_str import ceval_datasets as datasets

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='hf-chat-model',
        path='THUDM/chatglm-6b', # 替换为实际的本地模型权重路径
        tokenizer_path='THUDM/chatglm-6b',
        # ...其余参数配置详见配置文件
    )
]
```

执行命令（`--merge-ds` 是公共参数，使用自定义配置文件时仍可通过命令行追加）：

```bash
ais_bench ais_bench/configs/accuracy_benchmark_local/ceval_merge_zh_cn.py --merge-ds
```

> 💡 详细使用方法也可参考[服务化精度合并子数据集推理使用方法](accuracy_benchmark.md#合并子数据集推理)。

## 通过自定义配置文件实现

> 💡 上述所有功能场景（多任务测评、多任务并行、中断续测、合并子数据集等）均可以通过 [自定义配置文件方式](../../advanced_tutorials/run_custom_config.md) 实现。配置文件本质上是 Python 脚本，支持循环、条件判断、列表推导等所有 Python 语法，可将模型、数据集、summarizer 等配置写入一个文件，一次编写、多次复用。

本章节涉及的所有自定义配置文件样例已统一存放在 `ais_bench/configs/accuracy_benchmark_local/` 目录下，便于查阅与复用：

| 文件名 | 对应场景 |
| --- | --- |
| [single_task_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/single_task_zh_cn.py) | 单任务测评 |
| [multi_task_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/multi_task_zh_cn.py) | 纯模型多任务测评 / 多任务并行测评 |
| [ceval_merge_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/ceval_merge_zh_cn.py) | 合并子数据集推理 |
| [inference_re_eval_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/inference_re_eval_zh_cn.py) | 纯模型推理结果重评估 |

详见 [自定义配置文件运行AISBench](../../advanced_tutorials/run_custom_config.md#各场景自定义配置文件示例) 中"纯模型精度测评"示例。

## 其他功能

### 纯模型推理结果重评估

完整样例请参考 [inference_re_eval_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/inference_re_eval_zh_cn.py)：

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import HuggingFacewithChatTemplate
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask
from ais_bench.benchmark.datasets import gsm8k_postprocess, gsm8k_dataset_postprocess

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='hf-chat-model',
        path='THUDM/chatglm-6b', # 替换为实际的本地模型权重路径
        tokenizer_path='THUDM/chatglm-6b',
        # ...其余参数配置详见配置文件
    )
]

# 关键：替换或修改答案的提取函数实现
datasets[0]['eval_cfg']['pred_postprocessor'] = dict(type=gsm8k_postprocess)
datasets[0]['eval_cfg']['dataset_postprocessor'] = dict(type=gsm8k_dataset_postprocess)
```

执行命令（`--mode eval` 与 `--reuse` 是公共参数，使用自定义配置文件时仍可通过命令行追加）：

```bash
ais_bench ais_bench/configs/accuracy_benchmark_local/inference_re_eval_zh_cn.py --mode eval --reuse 20250628_151326
```

> 💡 详细使用方法也可参考[服务化精度推理结果重评估使用方法](accuracy_benchmark.md#推理结果重评估)。