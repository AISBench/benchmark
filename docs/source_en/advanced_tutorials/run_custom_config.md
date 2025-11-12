# Running AISBench with a Custom Configuration File
The standard command invocation method for AISBench specifies the model task via `--models`, the dataset task via `--datasets`, and the result presentation task via `--summarizer` to run an evaluation task. Additionally, AISBench supports specifying a **custom configuration file** that combines the configuration information of these three types of tasks, enabling the execution of custom task combinations.


## Usage Instructions
```bash
ais_bench ais_bench/configs/{model_type}_examples/{task_config_filename}
# Example:
ais_bench ais_bench/configs/api_examples/infer_vllm_api_general.py
```


## Example of Using a Custom Configuration File
### Editing the Example Content
The following example demonstrates how to evaluate the performance of two service interfaces ([`v1/chat/completions`](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py) and [`v1/completions`](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)) on the [GSM8K](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/gsm8k/README_en.md) and [MATH datasets](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/math/README_en.md). Refer to the sample file: [demo_infer_vllm_api.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/demo_infer_vllm_api.py):

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

# Use only a subset of samples for demo testing
gsm8k_0_shot_cot_str[0]['abbr'] = 'demo_' + gsm8k_0_shot_cot_str[0]['abbr']
gsm8k_0_shot_cot_str[0]['reader_cfg']['test_range'] = '[0:8]'

math500_gen_0_shot_cot_chat[0]['abbr'] = 'demo_' + math500_gen_0_shot_cot_chat[0]['abbr']
math500_gen_0_shot_cot_chat[0]['reader_cfg']['test_range'] = '[0:8]'

# Specify the dataset list; add different dataset configurations by concatenation
datasets = gsm8k_0_shot_cot_str + math500_gen_0_shot_cot_chat
# Specify the model configuration list
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='demo-vllm-api-general-chat',
        path="",
        model="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",  # Specify the IP address of the inference service
        host_port = 8080,       # Specify the port of the inference service
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
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_general_chat.py
```


### Output Results
```bash
dataset                 version  metric   mode  demo-vllm-api-general-chat demo-vllm-api-general
----------------------- -------- -------- ----- -------------------------- ---------------------
demo_gsm8k              401e4c   accuracy gen                     62.50                62.50
demo_math_prm800k_500   c4b6f0   accuracy gen                     50.00                62.50
```


## List of Preset Custom Configuration File Samples

| Filename | Description |
| --- | --- |
| [infer_vllm_api_general.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_general.py) | Evaluates the `v1/completions` sub-service using vLLM API (version 0.6+) on the GSM8K dataset. The prompt format is a string, and the dataset path is customized. |
| [infer_mindie_stream_api_general.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/infer_mindie_stream_api_general.py) | Evaluates the `infer` sub-service using MindIE Stream API on the GSM8K dataset. The prompt format is a string, and the dataset path is customized. |
| [infer_vllm_api_old.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_old.py) | Evaluates the `generate` sub-service using vLLM API (version 0.2.6) on the GSM8K dataset. The prompt format is a string, and the dataset path is customized. |
| [infer_vllm_api_general_chat.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_general_chat.py) | Evaluates the `v1/chat/completions` sub-service using vLLM API (version 0.6+) on the GSM8K dataset. The prompt format is a conversation format, and the dataset path is customized. |
| [infer_vllm_api_stream_chat.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_stream_chat.py) | Evaluates the `v1/chat/completions` sub-service with streaming inference using vLLM API (version 0.6+) on the GSM8K dataset. The prompt format is a conversation format, and the dataset path is customized. |
| [infer_hf_base_model.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_base_model.py) | Evaluates using the inference interface of a Hugging Face base model on the GSM8K dataset. The prompt format is a string, and the dataset path is customized. |
| [infer_hf_chat_model.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_chat_model.py) | Evaluates using the inference interface of a Hugging Face chat model on the GSM8K dataset. The prompt format is a string, and the dataset path is customized. |


**Note**: To evaluate other datasets using the above custom configuration files, import additional datasets from [ais_bench/configs/api_examples/all_dataset_configs.py](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/configs/api_examples/all_dataset_configs.py).