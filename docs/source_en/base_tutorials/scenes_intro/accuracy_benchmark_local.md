# Pure Model Accuracy Evaluation
Load models and datasets in a local environment, compare outputs with reference answers through a unified inference process, and evaluate the inherent accuracy of the model. Customize parameters such as batch size and sequence length, applicable to the **Huggingface Transformers** inference framework.

## Test Preparation
Before performing service-oriented inference, the following conditions must be met:

- Available model weights: Ensure that the model weight files to be tested are already available locally. Open-source weights can be obtained from 🔗 [Hugging Face Community](https://huggingface.co/models).
- Dataset task preparation: Select a dataset from 📚 [Open-Source Datasets](../../get_started/datasets.md#open-source-datasets), and choose the dataset task to execute in the "detailed introduction" document corresponding to the dataset. Prepare the dataset files according to the "detailed introduction" document of the selected dataset task. It is recommended to manually place the open-source dataset in the default directory `ais_bench/datasets/`, and the program will automatically load the dataset files during task execution.
- Model task preparation: Select the model task to execute from 📚 [Local Model Backend](../all_params/models.md#local-model-backend).

## Main Functions

The main functions in the pure model accuracy evaluation scenario are similar to those in the service-oriented accuracy evaluation scenario, but the model task needs to be replaced with a local HuggingFace model task (such as [`HuggingFacewithChatTemplate`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/models/huggingface_chat_model.py) or [`HuggingFaceBaseModel`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/models/huggingface_base_model.py)).

### Pure Model Multi-Task Evaluation

Supports simultaneous configuration of multiple dataset tasks through a single command for batch evaluation. For a complete example, refer to [multi_task_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/multi_task_en.py):

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
        path='THUDM/chatglm-6b', # Replace with the actual local model weight path
        tokenizer_path='THUDM/chatglm-6b',
        # ...For other parameter configurations, see the configuration file
    )
]
```

Execution command:

```bash
ais_bench ais_bench/configs/accuracy_benchmark_local/multi_task_en.py
```

#### Custom Model-Dataset Pairings (Optional)

By default, the `models` list and `datasets` list in the above configuration will automatically be combined in a Cartesian product, and the number of sub-tasks is the number of models × the number of datasets (1 × 2 = 2 in this example). If you want to precisely control which models are paired with which datasets (for example, only let the model run a subset of datasets), you can explicitly declare the pairing relationship through the `model_dataset_combinations` field in the configuration file:

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
        path='THUDM/chatglm-6b', # Replace with the actual local model weight path
        tokenizer_path='THUDM/chatglm-6b',
    )
]

# Key: Precisely control pairings through model_dataset_combinations
# The following example generates only 1 sub-task (the Cartesian product would generate 2):
#   - hf-chat-model + gsm8k
model_dataset_combinations = [
    dict(models=[models[0]], datasets=[datasets[0]]),
]
```

> ⚠️ **Note**: The unique identifier of a model or dataset is determined by the `abbr` field. In the same configuration file, combinations where models or datasets with the same `abbr` appear repeatedly will be considered duplicate tasks and will be skipped. When reusing model/dataset configurations through methods like `.copy()`, you must explicitly modify `abbr` to ensure uniqueness. For details, refer to 📚 [Custom Model-Dataset Combinations](../../advanced_tutorials/run_custom_config.md#custom-model-and-dataset-combinations).

> 💡 For detailed usage, you can also refer to [Usage of Service-Oriented Accuracy Multi-Task Evaluation](accuracy_benchmark.md#multi-task-evaluation).

### Pure Model Multi-Task Parallel Evaluation

Supports multi-task parallelism through the [`--max-num-workers`](../all_params/cli_args.md#common-parameters) command-line parameter. The configuration file example is exactly the same as [Pure Model Multi-Task Evaluation](#pure-model-multi-task-evaluation), the only difference is the execution command.

Execution command (taking `max-num-workers 4` as an example):

```bash
ais_bench ais_bench/configs/accuracy_benchmark_local/multi_task_en.py --max-num-workers 4
```

> ⚠️ Note: Multi-task parallel evaluation in pure model accuracy evaluation will occupy different GPU units. The number of GPU units required for parallel tasks should be less than or equal to the total number of available GPUs.

> 💡 For detailed usage, you can also refer to [Usage of Service-Oriented Accuracy Multi-Task Parallel Evaluation](accuracy_benchmark.md#multi-task-parallel-evaluation).

### Pure Model Resumption After Interruption

During the pure model accuracy evaluation, if the task is interrupted, you can use the `--reuse` parameter to specify the task timestamp directory to continue the unfinished inference task, realizing breakpoint resumption. This function does not require re-running all tasks, but only performs supplementary inference on the unfinished parts.

First execution command:

```bash
ais_bench ais_bench/configs/accuracy_benchmark_local/single_task_en.py
```

Specify the task timestamp directory through the `--reuse` parameter to continue (`--reuse` is a common parameter, and can still be appended through the command line when using a custom configuration file):

```bash
ais_bench ais_bench/configs/accuracy_benchmark_local/single_task_en.py --reuse 20250628_151326
```

> ⚠️ Note: Currently, pure model accuracy evaluation does not support automatic retesting of failed cases.

> 💡 For detailed usage, you can also refer to [Usage of Service-Oriented Accuracy Resumption After Interruption](accuracy_benchmark.md#resumption-after-interruption--retesting-of-failed-cases).

### Pure Model Merged Sub-Dataset Inference

Supports merging datasets containing multiple small-scale sub-datasets into a single task for unified evaluation. For a complete example, refer to [ceval_merge_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/ceval_merge_en.py):

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
        path='THUDM/chatglm-6b', # Replace with the actual local model weight path
        tokenizer_path='THUDM/chatglm-6b',
        # ...For other parameter configurations, see the configuration file
    )
]
```

Execution command (`--merge-ds` is a common parameter, and can still be appended through the command line when using a custom configuration file):

```bash
ais_bench ais_bench/configs/accuracy_benchmark_local/ceval_merge_en.py --merge-ds
```

> 💡 For detailed usage, you can also refer to [Usage of Service-Oriented Accuracy Merged Sub-Dataset Inference](accuracy_benchmark.md#merging-sub-dataset-inference).

## Implementation via Custom Configuration Files

> 💡 All the above functional scenarios (multi-task evaluation, multi-task parallel, resumption after interruption, merged sub-dataset, etc.) can be implemented through the [Custom Configuration File](../../advanced_tutorials/run_custom_config.md) approach. The configuration file is essentially a Python script, which supports all Python syntaxes such as loops, conditional judgments, and list comprehensions. Model, dataset, summarizer, and other configurations can be written into one file for one-time writing and multiple reuse.

All custom configuration file examples involved in this section are uniformly stored in the `ais_bench/configs/accuracy_benchmark_local/` directory for easy reference and reuse:

| File Name | Corresponding Scenario |
| --- | --- |
| [single_task_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/single_task_en.py) | Single-Task Evaluation |
| [multi_task_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/multi_task_en.py) | Pure Model Multi-Task Evaluation / Multi-Task Parallel Evaluation |
| [ceval_merge_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/ceval_merge_en.py) | Merged Sub-Dataset Inference |
| [inference_re_eval_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/inference_re_eval_en.py) | Re-Evaluation of Pure Model Inference Results |

For details, refer to the "Pure Model Accuracy Evaluation" example in [Running AISBench via Custom Configuration Files](../../advanced_tutorials/run_custom_config.md#custom-configuration-file-examples-for-each-scenario).

## Other Functions

### Re-Evaluation of Pure Model Inference Results

For a complete example, refer to [inference_re_eval_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/inference_re_eval_en.py):

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
        path='THUDM/chatglm-6b', # Replace with the actual local model weight path
        tokenizer_path='THUDM/chatglm-6b',
        # ...For other parameter configurations, see the configuration file
    )
]

# Key: Replace or modify the answer extraction function implementation
datasets[0]['eval_cfg']['pred_postprocessor'] = dict(type=gsm8k_postprocess)
datasets[0]['eval_cfg']['dataset_postprocessor'] = dict(type=gsm8k_dataset_postprocess)
```

Execution command (`--mode eval` and `--reuse` are common parameters, and can still be appended through the command line when using a custom configuration file):

```bash
ais_bench ais_bench/configs/accuracy_benchmark_local/inference_re_eval_en.py --mode eval --reuse 20250628_151326
```

> 💡 For detailed usage, you can also refer to [Usage of Service-Oriented Accuracy Re-Evaluation of Inference Results](accuracy_benchmark.md#re-evaluation-of-inference-results).