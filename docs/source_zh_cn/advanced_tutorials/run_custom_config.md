# 自定义配置文件运行AISBench

AISBench常规命令调用方式是通过`--models`指定模型任务，通过`--datasets`指定数据集任务，通过`--summarizer`指定结果呈现任务来绝对运行的测评任务，AISBench同样也支持指定自定义的配置文件将这三类任务对应的配置文件信息组合在一起，从而实现自定义的任务组合运行。

## 使用说明

```bash
ais_bench ais_bench/configs/{模型类型}_examples/{任务配置文件名}
# 示例：
ais_bench ais_bench/configs/api_examples/infer_vllm_api_general.py
  ```

## 自定义配置文件精度测评使用样例

### 样例内容编辑

以下示例展示如何同时评测两个服务接口（[`v1/chat/completions`](../../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py) 与 [`v1/completions`](../../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)）在 [GSM8K](../../../ais_bench/benchmark/configs/datasets/gsm8k/README.md) 与 [MATH数据集](../../../ais_bench/benchmark/configs/datasets/math/README.md)上的表现。参考示例：[demo_infer_vllm_api.py](../../../ais_bench/configs/api_examples/demo_infer_vllm_api.py)：

```python
from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local_api import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask
from ais_bench.benchmark.models import VLLMCustomAPIChat

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general

# 只取部分样本进行 demo 测试
gsm8k_0_shot_cot_str[0]['abbr'] = 'demo_' + gsm8k_0_shot_cot_str[0]['abbr']
gsm8k_0_shot_cot_str[0]['reader_cfg']['test_range'] = '[0:8]'

math500_gen_0_shot_cot_chat[0]['abbr'] = 'demo_' + math500_gen_0_shot_cot_chat[0]['abbr']
math500_gen_0_shot_cot_chat[0]['reader_cfg']['test_range'] = '[0:8]'

datasets = gsm8k_0_shot_cot_str + math500_gen_0_shot_cot_chat # 指定数据集列表，可通过累加添加不同的数据集配置
models = [      # 指定模型配置列表
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='demo-vllm-api-general-chat',
        path="",
        model="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",  # 指定推理服务的IP
        host_port = 8080,       # 指定推理服务的端口
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
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_general_chat.py
```

如果需要执行多任务并行，可以在命令行中添加 [`--max-num-workers`](../base_tutorials/all_params/cli_args.md#公共参数)参数指定最大任务并行数，示例如下：

```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_general_chat.py --max-num-workers 4
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

datasets = synthetic_datasets  # 指定数据集列表

vllm_api_general_stream[0]["abbr"] = "demo-" + vllm_api_general_stream[0]["abbr"]
vllm_api_stream_chat[0]["abbr"] = "demo-" + vllm_api_stream_chat[0]["abbr"]

models = vllm_api_general_stream + vllm_api_stream_chat # 指定模型列表

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
    dict(models=[models[0]], datasets=[datasets[0]]), # 组合1，使用模型0(vllm_api_general)与数据集0(gsm8k_0_shot_cot_str)进行组合
    dict(models=[models[1]], datasets=[datasets[1]]), # 组合2，使用模型1(vllm_api_general_chat)与数据集1(math500_gen_0_shot_cot_chat)进行组合
    dict(models=[models[2]], datasets=[datasets[0], datasets[1]]), # 组合3，使用模型2(vllm_api_stream_chat)与数据集0(gsm8k_0_shot_cot_str)和数据集1(math500_gen_0_shot_cot_chat)进行组合
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
    dict(models=[models[1]], datasets=datasets), # 组合1，使用模型1(vllm_api_general)与数据集(math500_gen_0_shot_cot_chat)进行组合
    dict(models=[models[0]], datasets=datasets), # 组合2，使用模型0(vllm_api_general_copy)与数据集0(math500_gen_0_shot_cot_chat)进行组合，由于vllm_api_general_copy与vllm_api_general的abbr相同，所以会被认为与组合1是相同任务，会被跳过，即便内部参数存在区别
]
```

正确做法：在对模型或数据集配置进行复用时，修改`abbr`参数，使其与原模型或数据集不同，例如:

```python
vllm_api_general_copy = vllm_api_general.copy()
vllm_api_general_copy[0]['abbr'] = vllm_api_general[0]['abbr'] + '-copy' # 修改abbr,标识模型
```

这样vllm_api_general_copy[0]与vllm_api_general[0]的abbr不同，组合2与组合1是不同任务，会被正常执行。

## 预设自定义配置文件文件样例列表

|文件名|简介|
| --- | --- |
|[infer_vllm_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_general.py)|基于gsm8k数据集使用vllm api(0.6+版本)访问v1/completions子服务进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_mindie_stream_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_mindie_stream_api_general.py)|基于gsm8k数据集使用mindie stream api访问infer子服务进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_vllm_api_general_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_general_chat.py)|基于gsm8k数据集使用vllm api(0.6+版本)访问v1/chat/completions子服务进行评测，prompt格式为对话格式，自定义了数据集路径|
|[infer_vllm_api_stream_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_stream_chat.py)|基于gsm8k数据集使用vllm api(0.6+版本)访问v1/chat/completions子服务使用流式推理进行评测，prompt格式为对话格式，自定义了数据集路径|
|[infer_hf_base_model.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_base_model.py)|基于gsm8k数据集使用huggingface base模型的推理接口进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_hf_chat_model.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_chat_model.py)|基于gsm8k数据集使用huggingface chat模型的推理接口进行评测，prompt格式为字符串格式，自定义了数据集路径|

**注**: 上述自定义配置文件如果要评测其他数据集，请从[ais_bench/configs/api_examples/all_dataset_configs.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/all_dataset_configs.py)导入其他数据集。
