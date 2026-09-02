# 自定义配置文件运行AISBench

AISBench常规命令调用方式是通过`--models`指定模型任务，通过`--datasets`指定数据集任务，通过`--summarizer`指定结果呈现任务来绝对运行的测评任务，AISBench同样也支持指定自定义的配置文件将这三类任务对应的配置文件信息组合在一起，从而实现自定义的任务组合运行。

## 为什么使用自定义配置文件

AISBench 提供了两种运行方式：**命令行参数方式（CLI）** 与 **自定义配置文件方式**。在实际使用中，推荐优先使用自定义配置文件方式，原因如下：

| 对比维度 | CLI 方式 | 配置文件方式 |
| --- | --- | --- |
| **可复用性** | 每次运行需要重新输入完整命令 | 配置文件可保存、版本管理、反复使用 |
| **表达能力** | 只能通过参数指定模型/数据集名称 | 可以精确控制模型参数、数据集采样范围、推理配置等所有细节 |
| **组合灵活性** | 仅支持笛卡尔积组合 | 支持 `model_dataset_combinations` 自定义任意模型-数据集配对 |
| **参数覆盖** | 无法修改预设模型/数据集内部参数 | 可直接修改 `abbr`、`test_range`、`host_ip`、`host_port` 等任意字段 |
| **批量运行** | 需要多次执行命令 | 一个配置文件即可同时运行多模型、多数据集组合 |
| **团队协作** | 命令难以共享和追溯 | 配置文件即代码，可提交到代码仓库进行 review 和复用 |

**总结**：CLI 方式适合快速验证，配置文件方式适合正式的、可复现的、复杂的测评场景。

## 配置文件即 Python 脚本

AISBench 的自定义配置文件本质上就是一个 Python 脚本。这意味着你可以在配置文件中使用所有 Python 语法特性来灵活构建测评任务。

### 使用 for 循环批量构建模型配置

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

### 使用列表推导式批量添加数据集

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

### 条件配置：根据环境变量切换

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

### 使用 `.copy()` 复用并修改模型配置

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

## 配置文件完整变量参考

自定义配置文件中可以定义以下顶层变量。所有变量均为可选，但至少需要定义 `models` 和 `datasets` 才能运行推理任务。

| 变量名 | 类型 | 是否必需 | 说明 |
| --- | --- | --- | --- |
| `models` | `list[dict]` | 是（推理时） | 模型配置列表。每个元素是一个字典，至少包含 `type`（模型类）、`abbr`（唯一标识）字段。服务化模型还需 `attr="service"`、`host_ip`、`host_port` 等；本地模型还需 `path`、`tokenizer_path` 等 |
| `datasets` | `list[dict]` | 是（推理时） | 数据集配置列表。每个元素是一个字典，至少包含 `type`（数据集类）、`abbr`（唯一标识）、`reader_cfg`、`infer_cfg`、`eval_cfg` 字段 |
| `summarizer` | `dict` | 否 | 结果汇总器配置。通常从 `ais_bench.benchmark.configs.summarizers.example` 导入。包含 `attr` 和 `summary_groups` 字段 |
| `model_dataset_combinations` | `list[dict]` | 否 | 自定义模型-数据集配对列表。每个元素为 `dict(models=[...], datasets=[...])`。不指定时，默认对 `models` 和 `datasets` 做笛卡尔积组合 |
| `work_dir` | `str` | 否 | 工作目录，推理结果和日志将输出到此目录下。默认为 `outputs/default/` |
| `infer` | `dict` | 否 | 推理流程配置。包含 `partitioner`（分区器）、`runner`（运行器，内含 `max_num_workers` 和 `task`）。不指定时使用默认推理流程 |
| `eval` | `dict` | 否 | 评测流程配置。结构同 `infer`。仅在需要独立评测阶段时使用（如 SWE-Bench、VBench 等场景） |

### models 字段详解

每个模型配置字典的常用字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `type` | class | 模型类，如 `VLLMCustomAPIChat`、`VLLMCustomAPI`、`HuggingFaceBaseModel`、`HuggingFacewithChatTemplate` 等 |
| `abbr` | `str` | 模型唯一标识，用于结果表格中的列名。同一配置文件中相同 `abbr` 的模型与数据集组合会被视为重复任务而跳过 |
| `attr` | `str` | 模型属性，服务化模型为 `"service"`，本地模型为 `"local"` |
| `path` | `str` | 模型路径（本地模型必填，服务化模型可为空字符串） |
| `model` | `str` | 服务化推理时指定的模型名称 |
| `host_ip` | `str` | 推理服务 IP 地址（服务化模型） |
| `host_port` | `int` | 推理服务端口（服务化模型） |
| `stream` | `bool` | 是否使用流式推理 |
| `max_out_len` | `int` | 最大输出 token 数 |
| `batch_size` | `int` | 推理 batch size |
| `max_seq_len` | `int` | 最大输入序列长度 |
| `request_rate` | `int` | 请求速率限制，0 表示不限制 |
| `retry` | `int` | 请求失败重试次数 |
| `generation_kwargs` | `dict` | 生成参数，如 `temperature`、`top_k`、`top_p`、`seed` 等 |
| `tokenizer_path` | `str` | Tokenizer 路径（本地模型） |
| `model_kwargs` | `dict` | 模型加载参数（本地模型），如 `device_map` |
| `tokenizer_kwargs` | `dict` | Tokenizer 参数（本地模型），如 `padding_side` |
| `run_cfg` | `dict` | 多卡/多机运行配置（本地模型），如 `dict(num_gpus=1, num_procs=1)` |
| `pred_postprocessor` | `dict` | 模型输出后处理器，如 `dict(type=extract_non_reasoning_content)` |

### datasets 字段详解

每个数据集配置字典的常用字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `type` | class | 数据集类，如 `GSM8KDataset`、`MATHDataset`、`SyntheticDataset` 等 |
| `abbr` | `str` | 数据集唯一标识，用于结果表格中的行名 |
| `path` | `str` | 数据集文件路径 |
| `reader_cfg` | `dict` | 读取器配置，包含 `input_columns`、`output_column`，可选 `test_range` 控制采样范围（如 `'[0:100]'`） |
| `infer_cfg` | `dict` | 推理配置，包含 `prompt_template`、`retriever`、`inferencer` |
| `eval_cfg` | `dict` | 评测配置，包含 `evaluator` 和可选的 `pred_postprocessor` |
| `judge_infer_cfg` | `dict` | 裁判模型推理配置（需要 LLM Judge 的数据集），包含 `judge_model`、`judge_dataset_type`、`prompt_template`、`retriever`、`inferencer` |

### infer 字段详解

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

## 使用说明

```bash
ais_bench ais_bench/configs/{模型类型}_examples/{任务配置文件名}
# 示例：
ais_bench ais_bench/configs/api_examples/infer_vllm_api_general.py
```

## 各场景自定义配置文件示例

### 1. 服务化精度测评

通过 API 访问推理服务，使用真实数据集进行精度测评。适用于 vLLM、MindIE、TGI、Triton 等服务化部署场景。

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

### 2. 纯模型精度测评

使用 HuggingFace 本地模型直接进行推理测评，无需部署服务。

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

### 3. 服务化性能测评

使用合成数据集对推理服务进行性能压测，输出 TTFT（首 Token 延迟）、TPOT（每 Token 延迟）、E2EL（端到端延迟）等指标。

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

运行命令：

```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_perf.py -m perf
```

### 4. 合成数据集性能测评

自定义合成数据集的参数，控制请求数量、输入/输出 token 长度分布等。

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

### 5. 多模型多数据集组合

同时测评多个模型在多个数据集上的表现，利用笛卡尔积自动组合。

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

### 6. 自定义模型-数据集配对

通过 `model_dataset_combinations` 精确控制哪些模型与哪些数据集组合，避免不必要的笛卡尔积。

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

### 7. 裁判模型测评

对于需要 LLM Judge 评判的数据集（如 AIME 2025），在数据集的 `judge_infer_cfg` 中配置裁判模型。

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

### 8. 稳态性能测评

通过控制 `request_rate` 参数和 `stream` 参数，模拟稳态负载下的性能表现。

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

### 9. 多轮对话性能测评

使用 ShareGPT 或 MTBench 多轮对话数据集进行性能测评。

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

### 10. 自定义数据集测评

当需要使用自己的数据集进行测评时，可以通过自定义数据集配置实现。

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

## 自定义配置文件精度测评使用样例

### 样例内容编辑

以下示例展示如何同时评测两个服务接口（[`v1/chat/completions`](../../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py) 与 [`v1/completions`](../../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)）在 [GSM8K](../../../ais_bench/benchmark/configs/datasets/gsm8k/README.md) 与 [MATH数据集](../../../ais_bench/benchmark/configs/datasets/math/README.md)上的表现。参考示例：[demo_infer_vllm_api.py](../../../ais_bench/configs/api_examples/demo_infer_vllm_api.py)：

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

### 执行自定义任务组合

修改好配置文件后，执行如下命令启动精度评测：

```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api.py
```

如果需要执行多任务并行，可以在命令行中添加 [`--max-num-workers`](../base_tutorials/all_params/cli_args.md#公共参数)参数指定最大任务并行数，示例如下：

```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api.py --max-num-workers 4
```

### 输出结果

```bash
dataset                 version  metric   mode  demo-vllm-api-general-chat demo-vllm-api-general
----------------------- -------- -------- ----- -------------------------- ---------------------
demo_gsm8k              401e4c   accuracy gen                     62.50                62.50
demo_math_prm800k_500   c4b6f0   accuracy gen                     50.00                62.50
```

## 自定义配置文件性能测评使用样例

### 样例内容编辑

以下示例展示如何同时评测两个服务接口（[`v1/chat/completions`](../../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py) 与 [`v1/completions`](../../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)）使用合成数据集进行性能测评的表现。参考示例：[demo_infer_vllm_api_perf.py](../../../ais_bench/configs/api_examples/demo_infer_vllm_api_perf.py)：

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

### 执行自定义任务组合

修改好配置文件后，执行如下命令启动性能评测：

```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_perf.py -m perf
```

如果需要执行多任务并行，可以在命令行中添加 [`--max-num-workers`](../base_tutorials/all_params/cli_args.md#公共参数)参数指定最大任务并行数，示例如下：

```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_perf.py -m perf --max-num-workers 2
```

### 输出结果

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

## 自定义模型与数据集组合

默认情况下，自定义配置文件中的模型与数据集组合会自动根据模型配置文件中的`models`列表和数据集配置文件中的`datasets`列表进行笛卡尔组合，组合数量为模型配置文件中的`models`列表长度与数据集配置文件中的`datasets`列表长度之积。用户可以通过在配置文件中配置`model_dataset_combinations`自定义模型数据集组合。

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

> ⚠️ **注意**：需要用`abbr`参数指定模型与数据集的唯一标识。同一配置文件中，相同`abbr`的模型与数据集只能组合一次。如下实例中，vllm_api_general_copy与vllm_api_general的abbr相同，所以会被认为与组合1是相同任务，会被跳过，即便内部参数存在区别：

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

正确做法：在对模型或数据集配置进行复用时，修改`abbr`参数，使其与原模型或数据集不同，例如:

```python
vllm_api_general_copy = vllm_api_general.copy()
vllm_api_general_copy[0]['abbr'] = vllm_api_general[0]['abbr'] + '-copy'
```

这样vllm_api_general_copy[0]与vllm_api_general[0]的abbr不同，组合2与组合1是不同任务，会被正常执行。

## 预设自定义配置文件样例列表

### 快速上手

| 文件名 | 简介 |
| --- | --- |
| [model_api_test_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/model_api_test_zh_cn.py) | 快速上手样例（中文注释）：配置 `vllm_api_general_chat` 服务化模型与 `demo_gsm8k_gen_4_shot_cot_chat_prompt` 数据集，执行单任务精度测评 |
| [model_api_test_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/model_api_test_en.py) | 快速上手样例（英文注释）：与 `model_api_test_zh_cn.py` 内容一致，注释为英文 |

### 服务化精度测评（`api_examples/`）

| 文件名 | 简介 |
| --- | --- |
| [infer_vllm_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_general.py) | 基于gsm8k数据集使用vllm api(0.6+版本)访问v1/completions子服务进行评测，prompt格式为字符串格式，自定义了数据集路径 |
| [infer_vllm_api_general_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_general_chat.py) | 基于gsm8k数据集使用vllm api(0.6+版本)访问v1/chat/completions子服务进行评测，prompt格式为对话格式，自定义了数据集路径 |
| [infer_vllm_api_stream_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_stream_chat.py) | 基于gsm8k数据集使用vllm api(0.6+版本)访问v1/chat/completions子服务使用流式推理进行评测，prompt格式为对话格式，自定义了数据集路径 |
| [infer_vllm_api_old.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_old.py) | 基于gsm8k数据集使用旧版vllm api访问v1/completions子服务进行评测，prompt格式为字符串格式 |
| [infer_mindie_stream_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_mindie_stream_api_general.py) | 基于gsm8k数据集使用mindie stream api访问infer子服务进行评测，prompt格式为字符串格式，自定义了数据集路径 |
| [demo_infer_vllm_api.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/demo_infer_vllm_api.py) | Demo示例：同时评测v1/chat/completions与v1/completions两个接口在GSM8K与MATH数据集上的精度表现 |
| [infer_vllm_api_multi_model_multi_dataset.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_multi_model_multi_dataset.py) | 多模型多数据集精度测评：将3个vllm服务化模型（general/general_chat/stream_chat）与GSM8K、MATH、MMLU数据集进行笛卡尔积组合 |
| [infer_vllm_api_with_model_dataset_combinations.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_with_model_dataset_combinations.py) | 自定义模型-数据集配对：通过 `model_dataset_combinations` 精确控制模型与数据集的配对关系 |
| [infer_vllm_api_with_judge_model.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_with_judge_model.py) | 裁判模型测评：评测需要LLM Judge的AIME 2025数据集，在 `judge_infer_cfg` 中配置裁判模型 |

### 服务化性能测评（`api_examples/`）

| 文件名 | 简介 |
| --- | --- |
| [demo_infer_vllm_api_perf.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/demo_infer_vllm_api_perf.py) | Demo示例：同时评测v1/chat/completions与v1/completions两个接口使用合成数据集进行流式性能测评 |
| [perf_vllm_api_synthetic.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/perf_vllm_api_synthetic.py) | 合成数据集性能测评：自定义合成数据集的输入输出token长度分布 |
| [perf_vllm_api_stable_stage.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/perf_vllm_api_stable_stage.py) | 稳态性能测评：以多个 `request_rate`（0/5/10/20）发送合成数据集进行稳态性能测试 |
| [perf_vllm_api_multiturn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/perf_vllm_api_multiturn.py) | 多轮对话性能测评：使用ShareGPT多轮对话数据集 |
| [perf_vllm_api_custom_dataset.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/perf_vllm_api_custom_dataset.py) | 自定义数据集性能测评：在自定义CSV/JSONL数据集上进行性能测评 |
| [perf_vllm_api_rps_distribution.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/perf_vllm_api_rps_distribution.py) | RPS分布控制性能测评：通过 `traffic_cfg`（burstiness、ramp-up策略）控制请求到达分布 |

### 纯模型精度测评（`hf_example/`）

| 文件名 | 简介 |
| --- | --- |
| [infer_hf_base_model.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_base_model.py) | 基于gsm8k数据集使用huggingface base模型的推理接口进行评测，prompt格式为字符串格式，自定义了数据集路径 |
| [infer_hf_chat_model.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_chat_model.py) | 基于gsm8k数据集使用huggingface chat模型的推理接口进行评测，prompt格式为对话格式，自定义了数据集路径 |
| [infer_hf_multi_model_multi_dataset.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_multi_model_multi_dataset.py) | 多模型多数据集纯模型测评：在多个数据集上评测多个HuggingFace本地模型 |

### 多模态测评（`lmm_example/`）

| 文件名 | 简介 |
| --- | --- |
| [multi_device_run_qwen_image_edit.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/lmm_example/multi_device_run_qwen_image_edit.py) | 多模态图像编辑模型测评（Qwen图像编辑，多设备） |
| [infer_lmm_multi_dataset.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/lmm_example/infer_lmm_multi_dataset.py) | 多模态多数据集精度测评：在多个多模态数据集上评测多模态模型 |

### 精度测评场景样例（`accuracy_benchmark/`）

| 文件名 | 简介 |
| --- | --- |
| [single_task_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/single_task_zh_cn.py) | 单任务精度测评 |
| [multi_task_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/multi_task_zh_cn.py) | 多任务精度测评 |
| [multi_task_parallel_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/multi_task_parallel_zh_cn.py) | 多任务并行精度测评 |
| [multi_task_resume_partial_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/multi_task_resume_partial_zh_cn.py) | 中断续跑与失败用例重测（部分任务） |
| [ceval_merge_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/ceval_merge_zh_cn.py) | 子数据集合并推理 |
| [fixed_prompts_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/fixed_prompts_zh_cn.py) | 固定请求数测评 |
| [multi_repeat_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/multi_repeat_zh_cn.py) | 多次独立重复推理 |
| [inference_re_eval_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark/inference_re_eval_zh_cn.py) | 推理结果重评估 |

### 纯模型精度测评场景样例（`accuracy_benchmark_local/`）

| 文件名 | 简介 |
| --- | --- |
| [single_task_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/single_task_zh_cn.py) | 纯模型单任务测评 |
| [multi_task_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/multi_task_zh_cn.py) | 纯模型多任务/多任务并行测评 |
| [ceval_merge_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/ceval_merge_zh_cn.py) | 子数据集合并推理 |
| [inference_re_eval_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/accuracy_benchmark_local/inference_re_eval_zh_cn.py) | 纯模型推理结果重评估 |

### 性能测评场景样例（`performance_benchmark/`）

| 文件名 | 简介 |
| --- | --- |
| [single_task_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/single_task_zh_cn.py) | 单任务性能测评 |
| [multi_task_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/multi_task_zh_cn.py) | 多任务性能测评 |
| [synthetic_gen_string_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/synthetic_gen_string_zh_cn.py) | 自定义序列长度性能测评 |
| [multi_task_synthetic_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/multi_task_synthetic_zh_cn.py) | 自定义序列的多任务组合性能测评 |
| [fixed_prompts_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/fixed_prompts_zh_cn.py) | 固定请求数性能测评 |
| [perf_recalculate_zh_cn.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/perf_recalculate_zh_cn.py) | 性能结果重计算 |

### 通用工具

| 文件名 | 简介 |
| --- | --- |
| [all_dataset_configs.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/all_dataset_configs.py) | 所有支持的数据集配置导入汇总，可在自定义配置文件中直接 `from ... import` 使用 |

**注**: 上述自定义配置文件如果要评测其他数据集，请从[ais_bench/configs/api_examples/all_dataset_configs.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/all_dataset_configs.py)导入其他数据集。
