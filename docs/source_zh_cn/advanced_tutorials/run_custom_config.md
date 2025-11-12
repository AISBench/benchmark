# 自定义配置文件运行AISBench
AISBench常规命令调用方式是通过`--models`指定模型任务，通过`--datasets`指定数据集任务，通过`--summarizer`指定结果呈现任务来绝对运行的测评任务，AISBench同样也支持指定自定义的配置文件将这三类任务对应的配置文件信息组合在一起，从而实现自定义的任务组合运行。
## 使用说明
```bash
ais_bench ais_bench/configs/{模型类型}_examples/{任务配置文件名}
# 示例：
ais_bench ais_bench/configs/api_examples/infer_vllm_api_general.py
  ```
## 自定义配置文件使用样例
### 样例内容编辑
以下示例展示如何同时评测两个服务接口（[`v1/chat/completions`](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py) 与 [`v1/completions`](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)）在 [GSM8K](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/gsm8k/README.md) 与 [MATH数据集](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/math/README.md)上的表现。参考示例：[demo_infer_vllm_api.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/demo_infer_vllm_api.py)：

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
### 输出结果
```bash
dataset                 version  metric   mode  demo-vllm-api-general-chat demo-vllm-api-general
----------------------- -------- -------- ----- -------------------------- ---------------------
demo_gsm8k              401e4c   accuracy gen                     62.50                62.50
demo_math_prm800k_500   c4b6f0   accuracy gen                     50.00                62.50
```

## 预设自定义配置文件文件样例列表

|文件名|简介|
| --- | --- |
|[infer_vllm_api_general.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_general.py)|基于gsm8k数据集使用vllm api(0.6+版本)访问v1/completions子服务进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_mindie_stream_api_general.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/infer_mindie_stream_api_general.py)|基于gsm8k数据集使用mindie stream api访问infer子服务进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_vllm_api_old.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_old.py)|基于gsm8k数据集使用vllm api(0.2.6版本)访问generate子服务进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_vllm_api_general_chat.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_general_chat.py)|基于gsm8k数据集使用vllm api(0.6+版本)访问v1/chat/completions子服务进行评测，prompt格式为对话格式，自定义了数据集路径|
|[infer_vllm_api_stream_chat.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_stream_chat.py)|基于gsm8k数据集使用vllm api(0.6+版本)访问v1/chat/completions子服务使用流式推理进行评测，prompt格式为对话格式，自定义了数据集路径|
|[infer_hf_base_model.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_base_model.py)|基于gsm8k数据集使用huggingface base模型的推理接口进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_hf_chat_model.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_chat_model.py)|基于gsm8k数据集使用huggingface chat模型的推理接口进行评测，prompt格式为字符串格式，自定义了数据集路径|

**注**: 上述自定义配置文件如果要评测其他数据集，请从[ais_bench/configs/api_examples/all_dataset_configs.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/all_dataset_configs.py)导入其他数据集。