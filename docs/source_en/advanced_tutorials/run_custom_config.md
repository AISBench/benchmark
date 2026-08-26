# Running AISBench with a Custom Configuration File
The standard command invocation method for AISBench specifies the model task via `--models`, the dataset task via `--datasets`, and the result presentation task via `--summarizer` to run an evaluation task. Additionally, AISBench supports specifying a **custom configuration file** that combines the configuration information of these three types of tasks, enabling the execution of custom task combinations.

## Why Use a Custom Configuration File
AISBench provides two ways to run tasks: **Command-Line Interface (CLI)** and **custom configuration file**. In actual use, it is recommended to prioritize the custom configuration file approach, for the following reasons:

| Comparison Dimension | CLI Approach | Configuration File Approach |
| --- | --- | --- |
| **Reusability** | The complete command must be re-entered for each run | Configuration files can be saved, version-managed, and reused repeatedly |
| **Expressiveness** | Only model/dataset names can be specified via parameters | Allows precise control over all details including model parameters, dataset sampling range, and inference configuration |
| **Combination Flexibility** | Only Cartesian product combinations are supported | Supports `model_dataset_combinations` for arbitrary custom model-dataset pairings |
| **Parameter Override** | Internal parameters of preset models/datasets cannot be modified | Any field such as `abbr`, `test_range`, `host_ip`, `host_port` can be modified directly |
| **Batch Execution** | Requires running the command multiple times | A single configuration file can run multiple model and dataset combinations at once |
| **Team Collaboration** | Commands are hard to share and trace | Configuration files are code and can be committed to a repository for review and reuse |

**Summary**: The CLI approach is suitable for quick validation, while the configuration file approach is suitable for formal, reproducible, and complex evaluation scenarios.

## Configuration Files Are Python Scripts
The AISBench custom configuration file is essentially a Python script. This means you can use all Python syntax features in the configuration file to flexibly construct evaluation tasks.

### Using `for` Loop to Batch Build Model Configurations

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPIChat

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str

datasets = gsm8k_0_shot_cot_str

models = []
for port in [8080, 8081, 8082]:
    models.append(
        dict(
            attr="service",
            type=VLLMCustomAPIChat,
            abbr=f'vllm-api-chat-port-{port}',
            path="",
            model="",
            request_rate=0,
            retry=2,
            host_ip="localhost",
            host_port=port,
            max_out_len=512,
            batch_size=1,
            generation_kwargs=dict(temperature=0.5, top_k=10, top_p=0.95),
        )
    )

work_dir = 'outputs/multi_port_benchmark/'
```

### Using List Comprehension to Batch Add Datasets

```python
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat

datasets = gsm8k_0_shot_cot_str + math500_gen_0_shot_cot_chat
datasets = [
    dict(d, abbr=f'my_{d["abbr"]}', reader_cfg=dict(d.get('reader_cfg', {}), test_range='[0:100]'))
    for d in datasets
]

models = vllm_api_general_chat
work_dir = 'outputs/my_benchmark/'
```

### Conditional Configuration: Switch Based on Environment Variables

```python
import os
from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPIChat, VLLMCustomAPI

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str

datasets = gsm8k_0_shot_cot_str

use_stream = os.environ.get('USE_STREAM', 'false').lower() == 'true'
model_type = VLLMCustomAPIChat if use_stream else VLLMCustomAPI

models = [
    dict(
        attr="service",
        type=model_type,
        abbr='vllm-api-conditional',
        path="",
        model="",
        stream=use_stream,
        request_rate=0,
        retry=2,
        host_ip=os.environ.get('HOST_IP', 'localhost'),
        host_port=int(os.environ.get('HOST_PORT', '8080')),
        max_out_len=512,
        batch_size=1,
        generation_kwargs=dict(temperature=0.5, top_k=10, top_p=0.95),
    )
]

work_dir = 'outputs/conditional_benchmark/'
```

### Using `.copy()` to Reuse and Modify Model Configurations

```python
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat

datasets = gsm8k_0_shot_cot_str

model_high_temp = vllm_api_general_chat.copy()
model_high_temp[0]['abbr'] = vllm_api_general_chat[0]['abbr'] + '-high-temp'
model_high_temp[0]['generation_kwargs']['temperature'] = 0.9

model_low_temp = vllm_api_general_chat.copy()
model_low_temp[0]['abbr'] = vllm_api_general_chat[0]['abbr'] + '-low-temp'
model_low_temp[0]['generation_kwargs']['temperature'] = 0.1

models = model_high_temp + model_low_temp
work_dir = 'outputs/temperature_comparison/'
```

## Complete Configuration File Variable Reference
The following top-level variables can be defined in the custom configuration file. All variables are optional, but at least `models` and `datasets` must be defined to run an inference task.

| Variable Name | Type | Required | Description |
| --- | --- | --- | --- |
| `models` | `list[dict]` | Yes (for inference) | List of model configurations. Each element is a dict that must at least include `type` (model class) and `abbr` (unique identifier) fields. Service-oriented models additionally require `attr="service"`, `host_ip`, `host_port`, etc.; local models additionally require `path`, `tokenizer_path`, etc. |
| `datasets` | `list[dict]` | Yes (for inference) | List of dataset configurations. Each element is a dict that must at least include `type` (dataset class), `abbr` (unique identifier), `reader_cfg`, `infer_cfg`, and `eval_cfg` fields |
| `summarizer` | `dict` | No | Result summarizer configuration. Usually imported from `ais_bench.benchmark.configs.summarizers.example`. Contains `attr` and `summary_groups` fields |
| `model_dataset_combinations` | `list[dict]` | No | List of custom model-dataset pairings. Each element is `dict(models=[...], datasets=[...])`. When not specified, the Cartesian product of `models` and `datasets` is used by default |
| `work_dir` | `str` | No | Working directory; inference results and logs will be output to this directory. Defaults to `outputs/default/` |
| `infer` | `dict` | No | Inference process configuration. Contains `partitioner` (partitioner) and `runner` (runner, with `max_num_workers` and `task` inside). Uses the default inference process when not specified |
| `eval` | `dict` | No | Evaluation process configuration. Same structure as `infer`. Only used when an independent evaluation phase is needed (e.g., SWE-Bench, VBench scenarios) |

### Detailed `models` Field Description

Common fields for each model configuration dict:

| Field | Type | Description |
| --- | --- | --- |
| `type` | class | Model class, such as `VLLMCustomAPIChat`, `VLLMCustomAPI`, `HuggingFaceBaseModel`, `HuggingFacewithChatTemplate`, etc. |
| `abbr` | `str` | Unique identifier of the model, used as the column name in the result table. Model-dataset combinations with the same `abbr` in the same configuration file will be treated as duplicate tasks and skipped |
| `attr` | `str` | Model attribute; `"service"` for service-oriented models, `"local"` for local models |
| `path` | `str` | Model path (required for local models; can be an empty string for service-oriented models) |
| `model` | `str` | Model name specified for service-oriented inference |
| `host_ip` | `str` | IP address of the inference service (for service-oriented models) |
| `host_port` | `int` | Port of the inference service (for service-oriented models) |
| `stream` | `bool` | Whether to use streaming inference |
| `max_out_len` | `int` | Maximum output token count |
| `batch_size` | `int` | Inference batch size |
| `max_seq_len` | `int` | Maximum input sequence length |
| `request_rate` | `int` | Request rate limit; 0 means unlimited |
| `retry` | `int` | Number of retries for failed requests |
| `generation_kwargs` | `dict` | Generation parameters, such as `temperature`, `top_k`, `top_p`, `seed`, etc. |
| `tokenizer_path` | `str` | Tokenizer path (for local models) |
| `model_kwargs` | `dict` | Model loading parameters (for local models), such as `device_map` |
| `tokenizer_kwargs` | `dict` | Tokenizer parameters (for local models), such as `padding_side` |
| `run_cfg` | `dict` | Multi-GPU/multi-machine run configuration (for local models), such as `dict(num_gpus=1, num_procs=1)` |
| `pred_postprocessor` | `dict` | Model output post-processor, such as `dict(type=extract_non_reasoning_content)` |

### Detailed `datasets` Field Description

Common fields for each dataset configuration dict:

| Field | Type | Description |
| --- | --- | --- |
| `type` | class | Dataset class, such as `GSM8KDataset`, `MATHDataset`, `SyntheticDataset`, etc. |
| `abbr` | `str` | Unique identifier of the dataset, used as the row name in the result table |
| `path` | `str` | Dataset file path |
| `reader_cfg` | `dict` | Reader configuration, containing `input_columns`, `output_column`, and optional `test_range` to control the sampling range (e.g., `'[0:100]'`) |
| `infer_cfg` | `dict` | Inference configuration, containing `prompt_template`, `retriever`, `inferencer` |
| `eval_cfg` | `dict` | Evaluation configuration, containing `evaluator` and optional `pred_postprocessor` |
| `judge_infer_cfg` | `dict` | Judge model inference configuration (for datasets requiring LLM Judge), containing `judge_model`, `judge_dataset_type`, `prompt_template`, `retriever`, `inferencer` |

### Detailed `infer` Field Description

```python
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,
        task=dict(type=OpenICLApiInferTask),
    ),
)
```

## Usage Instructions
```bash
ais_bench ais_bench/configs/{model_type}_examples/{task_config_filename}
# Example:
ais_bench ais_bench/configs/api_examples/infer_vllm_api_general.py
```


## Custom Configuration File Examples for Each Scenario

### 1. Service-Oriented Accuracy Evaluation
Access the inference service via API and perform accuracy evaluation using real datasets. Applicable to service-oriented deployment scenarios such as vLLM, MindIE, TGI, Triton, etc.

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_chat_prompt import gsm8k_datasets as gsm8k_0_shot_cot_chat

datasets = [*gsm8k_0_shot_cot_chat]

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        model="",
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=8080,
        max_out_len=512,
        batch_size=1,
        generation_kwargs=dict(
            temperature=0.5,
            top_k=10,
            top_p=0.95,
            seed=None,
            repetition_penalty=1.03,
        )
    )
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,
        task=dict(type=OpenICLApiInferTask),
    ),
)

work_dir = 'outputs/api-vllm-general-chat/'
```

### 2. Pure Model Accuracy Evaluation
Use a HuggingFace local model for direct inference and evaluation without deploying a service.

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import HuggingFaceBaseModel
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_chat_prompt import gsm8k_datasets as gsm8k_0_shot_cot_chat

datasets = [*gsm8k_0_shot_cot_chat]

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='hf-base-model',
        path='THUDM/chatglm-6b',
        tokenizer_path='THUDM/chatglm-6b',
        model_kwargs=dict(device_map='auto'),
        tokenizer_kwargs=dict(padding_side='left'),
        generation_kwargs=dict(
            temperature=0.5,
            top_k=10,
            top_p=0.95,
            do_sample=True,
            seed=None,
            repetition_penalty=1.03,
        ),
        max_out_len=100,
        batch_size=1,
        max_seq_len=2048,
        batch_padding=True,
    )
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,
        task=dict(type=OpenICLApiInferTask),
    ),
)

work_dir = 'outputs/hf-base-model/'
```

### 3. Service-Oriented Performance Evaluation
Use a synthetic dataset to perform performance stress testing on the inference service, outputting metrics such as TTFT (Time To First Token), TPOT (Time Per Output Token), and E2EL (End-to-End Latency).

```python
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import (
        synthetic_datasets,
    )
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import (
        models as vllm_api_general_stream,
    )
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import (
        models as vllm_api_stream_chat,
    )

datasets = synthetic_datasets

vllm_api_general_stream[0]["abbr"] = "demo-" + vllm_api_general_stream[0]["abbr"]
vllm_api_stream_chat[0]["abbr"] = "demo-" + vllm_api_stream_chat[0]["abbr"]

models = vllm_api_general_stream + vllm_api_stream_chat

work_dir = "outputs/demo_api-vllm-stream-perf/"
```

Run command:

```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_perf.py -m perf
```

### 4. Synthetic Dataset Performance Evaluation
Customize the parameters of the synthetic dataset to control the number of requests and the input/output token length distribution.

```python
from mmengine.config import read_base
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import SyntheticDataset, MATHEvaluator, math_postprocess_v2

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import (
        models as vllm_api_general_stream,
    )

synthetic_config = {
    "Type": "string",
    "RequestCount": 100,
    "TrustRemoteCode": False,
    "StringConfig": {
        "Input": {
            "Method": "uniform",
            "Params": {"MinValue": 1, "MaxValue": 500}
        },
        "Output": {
            "Method": "gaussian",
            "Params": {"Mean": 200, "Var": 100, "MinValue": 1, "MaxValue": 500}
        }
    },
}

datasets = [
    dict(
        abbr='synthetic_custom',
        type=SyntheticDataset,
        config=synthetic_config,
        reader_cfg=dict(input_columns=['question', 'max_out_len'], output_column='answer'),
        infer_cfg=dict(
            prompt_template=dict(type=PromptTemplate, template="{question}"),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            evaluator=dict(type=MATHEvaluator, version='v2'),
            pred_postprocessor=dict(type=math_postprocess_v2),
        ),
    )
]

models = vllm_api_general_stream
work_dir = 'outputs/synthetic_perf_custom/'
```

### 5. Multi-Model Multi-Dataset Combinations
Simultaneously evaluate the performance of multiple models on multiple datasets, automatically combined via the Cartesian product.

```python
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat
    from ais_bench.benchmark.configs.datasets.mmlu.mmlu_gen_5_shot_str import mmlu_datasets as mmlu_5_shot_str
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

datasets = gsm8k_0_shot_cot_str + math500_gen_0_shot_cot_chat + mmlu_5_shot_str
models = vllm_api_general + vllm_api_general_chat + vllm_api_stream_chat

work_dir = 'outputs/multi_model_multi_dataset/'
```

### 6. Custom Model-Dataset Pairings
Precisely control which models are paired with which datasets via `model_dataset_combinations` to avoid unnecessary Cartesian products.

```python
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

models = vllm_api_general + vllm_api_general_chat + vllm_api_stream_chat
datasets = gsm8k_0_shot_cot_str + math500_gen_0_shot_cot_chat

model_dataset_combinations = [
    dict(models=[models[0]], datasets=[datasets[0]]),
    dict(models=[models[1]], datasets=[datasets[1]]),
    dict(models=[models[2]], datasets=[datasets[0], datasets[1]]),
]

work_dir = 'outputs/custom_combinations/'
```

### 7. Judge Model Evaluation
For datasets that require LLM Judge evaluation (e.g., AIME 2025), configure the judge model in the dataset's `judge_infer_cfg`.

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.aime2025.aime2025_gen_0_shot_llmjudge import aime2025_datasets

datasets = aime2025_datasets

datasets[0]['judge_infer_cfg']['judge_model']['host_ip'] = 'localhost'
datasets[0]['judge_infer_cfg']['judge_model']['host_port'] = 8081

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-judge-eval',
        path="",
        model="",
        stream=True,
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=8080,
        max_out_len=512,
        batch_size=1,
        generation_kwargs=dict(temperature=0.01, ignore_eos=False),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,
        task=dict(type=OpenICLApiInferTask),
    ),
)

work_dir = 'outputs/judge_eval/'
```

### 8. Steady-State Performance Evaluation
Simulate performance under steady-state load by controlling the `request_rate` parameter and `stream` parameter.

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPI

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import (
        synthetic_datasets,
    )

datasets = synthetic_datasets

models = []
for rate in [0, 5, 10, 20]:
    model_cfg = dict(
        attr="service",
        type=VLLMCustomAPI,
        abbr=f'vllm-api-steady-rate-{rate}',
        path="",
        model="",
        stream=True,
        request_rate=rate,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8080,
        url="",
        max_out_len=512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(temperature=0.01, ignore_eos=False),
    )
    models.append(model_cfg)

work_dir = 'outputs/steady_state_perf/'
```

### 9. Multi-Turn Dialogue Performance Evaluation
Use the ShareGPT or MTBench multi-turn dialogue datasets for performance evaluation.

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.sharegpt.sharegpt_gen import sharegpt_datasets

datasets = sharegpt_datasets

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-multiturn-api-chat-stream",
        path="",
        model="",
        stream=True,
        request_rate=0,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8080,
        url="",
        max_out_len=512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(temperature=0.01, ignore_eos=False),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,
        task=dict(type=OpenICLApiInferTask),
    ),
)

work_dir = 'outputs/multi_turn_benchmark/'
```

### 10. Custom Dataset Evaluation
When you need to use your own dataset for evaluation, you can do so by customizing the dataset configuration.

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer

datasets = [
    dict(
        abbr='my_custom_dataset',
        type=CustomDataset,
        path='/path/to/your/dataset.jsonl',
        reader_cfg=dict(
            input_columns=['question'],
            output_column='answer',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template='{question}',
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            evaluator=dict(type=AccEvaluator),
            pred_role='BOT',
        ),
        meta_path='',
    )
]

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-custom-dataset',
        model="",
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=8080,
        max_out_len=512,
        batch_size=1,
        generation_kwargs=dict(temperature=0.5, top_k=10, top_p=0.95),
    )
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=2,
        task=dict(type=OpenICLApiInferTask),
    ),
)

work_dir = 'outputs/custom_dataset/'
```


## Example of Using a Custom Configuration File for Accuracy Evaluation
### Editing the Example Content
The following example demonstrates how to evaluate the performance of two service interfaces ([`v1/chat/completions`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py) and [`v1/completions`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)) on the [GSM8K](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/gsm8k/README_en.md) and [MATH datasets](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/math/README_en.md). Refer to the sample file: [demo_infer_vllm_api.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/demo_infer_vllm_api.py):

```python
from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask
from ais_bench.benchmark.models import VLLMCustomAPIChat

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general

gsm8k_0_shot_cot_str[0]['abbr'] = 'demo_' + gsm8k_0_shot_cot_str[0]['abbr']
gsm8k_0_shot_cot_str[0]['reader_cfg']['test_range'] = '[0:8]'

math500_gen_0_shot_cot_chat[0]['abbr'] = 'demo_' + math500_gen_0_shot_cot_chat[0]['abbr']
math500_gen_0_shot_cot_chat[0]['reader_cfg']['test_range'] = '[0:8]'

datasets = gsm8k_0_shot_cot_str + math500_gen_0_shot_cot_chat
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='demo-vllm-api-general-chat',
        path="",
        model="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",
        host_port = 8080,
        max_out_len = 512,
        batch_size=1,
        generation_kwargs = dict(
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        )
    )
]

work_dir = 'outputs/demo_api-vllm-general-chat/'
```


### Executing the Custom Task Combination
After modifying the configuration file, run the following command to start the accuracy evaluation:
```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api.py
```

If you need to execute multiple tasks in parallel, you can add the [`--max-num-workers`](../base_tutorials/all_params/cli_args.md#common-parameters) parameter to the command line to specify the maximum number of parallel tasks. Example:
```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api.py --max-num-workers 4
```


### Output Results
```bash
dataset                 version  metric   mode  demo-vllm-api-general-chat demo-vllm-api-general
----------------------- -------- -------- ----- -------------------------- ---------------------
demo_gsm8k              401e4c   accuracy gen                     62.50                62.50
demo_math_prm800k_500   c4b6f0   accuracy gen                     50.00                62.50
```

## Example of Using a Custom Configuration File for Performance Evaluation
### Editing the Example Content
The following example demonstrates how to evaluate the performance of two service interfaces ([`v1/chat/completions`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py) and [`v1/completions`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)) using synthetic datasets for performance evaluation. Refer to the sample file: [demo_infer_vllm_api_perf.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/demo_infer_vllm_api_perf.py):

```python
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import (
        synthetic_datasets,
    )
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import (
        models as vllm_api_general_stream,
    )
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import (
        models as vllm_api_stream_chat,
    )

datasets = synthetic_datasets

vllm_api_general_stream[0]["abbr"] = "demo-" + vllm_api_general_stream[0]["abbr"]
vllm_api_stream_chat[0]["abbr"] = "demo-" + vllm_api_stream_chat[0]["abbr"]

models = vllm_api_general_stream + vllm_api_stream_chat

work_dir = "outputs/demo_api-vllm-stream-perf/"
```

### Executing the Custom Task Combination
After modifying the configuration file, run the following command to start the performance evaluation:
```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_perf.py -m perf
```

If you need to execute multiple tasks in parallel, you can add the [`--max-num-workers`](../base_tutorials/all_params/cli_args.md#common-parameters) parameter to the command line to specify the maximum number of parallel tasks. Example:
```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_perf.py -m perf --max-num-workers 2
```

### Output Results
```bash
[2025-12-05 12:10:44,147] [ais_bench] [INFO] Performance Results of task [demo-vllm-api-general-stream/syntheticdataset]:
╒══════════════════════════╤═════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════╕
│ Performance Parameters   │ Stage   │ Average         │ Min             │ Max             │ Median          │ P75             │ P90             │ P99             │  N  │
╞══════════════════════════╪═════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════╡
│ E2EL                     │ total   │ 1734.3 ms       │ 544.8 ms        │ 3692.3 ms       │ 1664.0 ms       │ 2081.5 ms       │ 2748.4 ms       │ 3597.9 ms       │ 10  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ TTFT                     │ total   │ 103.5 ms        │ 102.4 ms        │ 107.0 ms        │ 103.1 ms        │ 103.3 ms        │ 104.2 ms        │ 106.8 ms        │ 10  │
...
[2025-12-05 12:10:44,149] [ais_bench] [INFO] Performance Result files located in outputs/demo_api-vllm-general-stream-chat-perf/20251205_121020/performances/demo-vllm-api-general-stream-chat.
[2025-12-05 12:10:44,149] [ais_bench] [INFO] Performance Results of task [demo-vllm-api-stream-chat/syntheticdataset]:
╒══════════════════════════╤═════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤════════════════╤═════════════════╤═════════════════╤═════╕
│ Performance Parameters   │ Stage   │ Average         │ Min             │ Max             │ Median          │ P75            │ P90             │ P99             │  N  │
╞══════════════════════════╪═════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪════════════════╪═════════════════╪═════════════════╪═════╡
│ E2EL                     │ total   │ 3406.7 ms       │ 372.4 ms        │ 5772.4 ms       │ 3589.8 ms       │ 4476.6 ms      │ 4921.1 ms       │ 5647.1 ms       │ 10  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼────────────────┼─────────────────┼─────────────────┼─────┤
│ TTFT                     │ total   │ 103.2 ms        │ 102.0 ms        │ 107.5 ms        │ 102.9 ms        │ 103.4 ms       │ 104.3 ms        │ 107.2 ms        │ 10  │
```

## Custom Model and Dataset Combinations
By default, model and dataset combinations in custom configuration files are automatically generated as a Cartesian product based on the `models` list in the model configuration file and the `datasets` list in the dataset configuration file. The number of combinations equals the product of the lengths of the `models` list and the `datasets` list. Users can customize model-dataset combinations by configuring `model_dataset_combinations` in the configuration file.

```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

models = vllm_api_general + vllm_api_general_chat + vllm_api_stream_chat
datasets = gsm8k_0_shot_cot_str + math500_gen_0_shot_cot_chat
model_dataset_combinations = [
    dict(models=[models[0]], datasets=[datasets[0]]),
    dict(models=[models[1]], datasets=[datasets[1]]),
    dict(models=[models[2]], datasets=[datasets[0], datasets[1]]),
    ...
]
```

> ⚠️ **Note**: The `abbr` parameter must be used to specify a unique identifier for models and datasets. In the same configuration file, models and datasets with the same `abbr` can only be combined once. In the following example, `vllm_api_general_copy` and `vllm_api_general` have the same `abbr`, so combination 2 will be considered the same task as combination 1 and will be skipped, even if the internal parameters differ:

```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat

vllm_api_general_copy = vllm_api_general.copy()
vllm_api_general_copy[0]['port'] = 8081
models = vllm_api_general_copy + vllm_api_general
datasets = math500_gen_0_shot_cot_chat
model_dataset_combinations = [
    dict(models=[models[1]], datasets=datasets),
    dict(models=[models[0]], datasets=datasets),
]
```

Correct approach: When reusing model or dataset configurations, modify the `abbr` parameter to make it different from the original model or dataset. For example:

```python
vllm_api_general_copy = vllm_api_general.copy()
vllm_api_general_copy[0]['abbr'] = vllm_api_general[0]['abbr'] + '-copy'
```

In this way, `vllm_api_general_copy[0]` and `vllm_api_general[0]` have different `abbr` values, so combination 2 and combination 1 are different tasks and will be executed normally.

## List of Preset Custom Configuration File Samples

### Quick Start

| Filename | Description |
| --- | --- |
| [model_api_test_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/model_api_test_en.py) | Quick-start example (English): configures the `vllm_api_general_chat` service model and the `demo_gsm8k_gen_4_shot_cot_chat_prompt` dataset for a single accuracy evaluation task. |
| [model_api_test_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/model_api_test_zh_cn.py) | Quick-start example (Chinese): same as `model_api_test_en.py`, with Chinese comments. |

### Service-Oriented Accuracy Evaluation (`api_examples/`)

| Filename | Description |
| --- | --- |
| [infer_vllm_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_general.py) | Evaluates the `v1/completions` sub-service using vLLM API (version 0.6+) on the GSM8K dataset. The prompt format is a string, and the dataset path is customized. |
| [infer_vllm_api_general_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_general_chat.py) | Evaluates the `v1/chat/completions` sub-service using vLLM API (version 0.6+) on the GSM8K dataset. The prompt format is a conversation format, and the dataset path is customized. |
| [infer_vllm_api_stream_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_stream_chat.py) | Evaluates the `v1/chat/completions` sub-service with streaming inference using vLLM API (version 0.6+) on the GSM8K dataset. The prompt format is a conversation format, and the dataset path is customized. |
| [infer_vllm_api_old.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_old.py) | Evaluates the `v1/completions` sub-service using older vLLM API on the GSM8K dataset. The prompt format is a string. |
| [infer_mindie_stream_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_mindie_stream_api_general.py) | Evaluates the `infer` sub-service using MindIE Stream API on the GSM8K dataset. The prompt format is a string, and the dataset path is customized. |
| [demo_infer_vllm_api.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/demo_infer_vllm_api.py) | Demo example: Evaluates the accuracy of two interfaces `v1/chat/completions` and `v1/completions` simultaneously on the GSM8K and MATH datasets. |
| [infer_vllm_api_multi_model_multi_dataset.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_multi_model_multi_dataset.py) | Multi-model multi-dataset accuracy evaluation: combines 3 vLLM service models (`general`, `general_chat`, `stream_chat`) with the GSM8K, MATH, and MMLU datasets via the Cartesian product. |
| [infer_vllm_api_with_model_dataset_combinations.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_with_model_dataset_combinations.py) | Custom model-dataset pairings: precisely controls which models are paired with which datasets via `model_dataset_combinations`. |
| [infer_vllm_api_with_judge_model.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_with_judge_model.py) | Judge model evaluation: evaluates the AIME 2025 dataset that requires an LLM Judge, configuring the judge model in `judge_infer_cfg`. |

### Service-Oriented Performance Evaluation (`api_examples/`)

| Filename | Description |
| --- | --- |
| [demo_infer_vllm_api_perf.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/demo_infer_vllm_api_perf.py) | Demo example: Evaluates the streaming performance of two interfaces `v1/chat/completions` and `v1/completions` simultaneously using synthetic datasets. |
| [perf_vllm_api_synthetic.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/perf_vllm_api_synthetic.py) | Synthetic dataset performance evaluation: customizes the input/output token length distributions of the synthetic dataset. |
| [perf_vllm_api_stable_stage.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/perf_vllm_api_stable_stage.py) | Steady-state performance evaluation: sends the synthetic dataset at multiple `request_rate`s (0/5/10/20) for steady-state performance testing. |
| [perf_vllm_api_multiturn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/perf_vllm_api_multiturn.py) | Multi-turn dialogue performance evaluation: uses the ShareGPT multi-turn dialogue dataset. |
| [perf_vllm_api_custom_dataset.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/perf_vllm_api_custom_dataset.py) | Custom dataset performance evaluation: evaluates performance on your own CSV/JSONL dataset. |
| [perf_vllm_api_rps_distribution.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/perf_vllm_api_rps_distribution.py) | RPS distribution control performance evaluation: configures `traffic_cfg` (burstiness, ramp-up strategy) to control the request arrival distribution. |

### Pure Model Accuracy Evaluation (`hf_example/`)

| Filename | Description |
| --- | --- |
| [infer_hf_base_model.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_base_model.py) | Evaluates using the inference interface of a Hugging Face base model on the GSM8K dataset. The prompt format is a string, and the dataset path is customized. |
| [infer_hf_chat_model.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_chat_model.py) | Evaluates using the inference interface of a Hugging Face chat model on the GSM8K dataset. The prompt format is a conversation format, and the dataset path is customized. |
| [infer_hf_multi_model_multi_dataset.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_multi_model_multi_dataset.py) | Multi-model multi-dataset pure model evaluation: evaluates multiple Hugging Face local models on multiple datasets. |

### Multimodal Evaluation (`lmm_example/`)

| Filename | Description |
| --- | --- |
| [multi_device_run_qwen_image_edit.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/lmm_example/multi_device_run_qwen_image_edit.py) | Multimodal image-edit model evaluation (Qwen image edit, multi-device). |
| [infer_lmm_multi_dataset.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/lmm_example/infer_lmm_multi_dataset.py) | Multimodal multi-dataset accuracy evaluation: evaluates a multimodal model on multiple multimodal datasets. |

### Accuracy Evaluation Scenario Samples (`accuracy_benchmark/`)

| Filename | Description |
| --- | --- |
| [single_task_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/single_task_en.py) | Single-task accuracy evaluation |
| [multi_task_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/multi_task_en.py) | Multi-task accuracy evaluation |
| [multi_task_parallel_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/multi_task_parallel_en.py) | Multi-task parallel accuracy evaluation |
| [multi_task_resume_partial_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/multi_task_resume_partial_en.py) | Resumption after interruption & retesting of failed cases (partial tasks) |
| [ceval_merge_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/ceval_merge_en.py) | Merging sub-dataset inference |
| [fixed_prompts_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/fixed_prompts_en.py) | Fixed request count evaluation |
| [multi_repeat_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/multi_repeat_en.py) | Multiple independent repeat inference |
| [inference_re_eval_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/inference_re_eval_en.py) | Re-evaluation of inference results |

### Pure Model Accuracy Evaluation Scenario Samples (`accuracy_benchmark_local/`)

| Filename | Description |
| --- | --- |
| [single_task_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/single_task_en.py) | Single-task pure model evaluation |
| [multi_task_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/multi_task_en.py) | Pure model multi-task / multi-task parallel evaluation |
| [ceval_merge_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/ceval_merge_en.py) | Merged sub-dataset inference |
| [inference_re_eval_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/inference_re_eval_en.py) | Re-evaluation of pure model inference results |

### Performance Evaluation Scenario Samples (`performance_benchmark/`)

| Filename | Description |
| --- | --- |
| [performance_qwen2_7b_sharegpt.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_qwen2_7b_sharegpt.py) | Single-task performance evaluation (ShareGPT) |
| [performance_multi_dataset.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_multi_dataset.py) | Multi-dataset performance evaluation |
| [performance_multi_rate.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_multi_rate.py) | Multi-rate performance evaluation |
| [performance_multi_model.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_multi_model.py) | Multi-model performance evaluation |
| [performance_synthetic.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_synthetic.py) | Synthetic dataset multi-task combinations |
| [performance_seq_combinations.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_seq_combinations.py) | Custom sequence multi-task combinations |
| [performance_fixed_request.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_fixed_request.py) | Fixed request count performance evaluation |
| [performance_re_eval.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_re_eval.py) | Performance result recalculation |

### Common Utilities

| Filename | Description |
| --- | --- |
| [all_dataset_configs.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/all_dataset_configs.py) | A consolidated import of all supported dataset configurations; can be used directly via `from ... import` in custom configuration files. |

**Note**: To evaluate other datasets using the above custom configuration files, import additional datasets from [ais_bench/configs/api_examples/all_dataset_configs.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/all_dataset_configs.py).