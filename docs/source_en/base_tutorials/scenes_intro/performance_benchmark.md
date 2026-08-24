# Service-Oriented Performance Evaluation
Send batch requests to the service through a unified request interface to evaluate the service performance of the model in actual deployment scenarios. The request sending mode and request data can be customized to obtain performance indicators such as throughput and latency. It supports two deployment frameworks: **vLLM** and **vLLM-Ascend**, and provides complete performance analysis reports.

## Quick Start

### Prerequisite

The performance evaluation requires **first preparing a service environment** (i.e., a service program that provides OpenAI-compatible interfaces).

Here is the reference service startup method (vLLM OpenAI-compatible service):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8080 --max-model-len 4096
```

Wait for the service to start successfully (the port shows that the service process is listening), then use the following configuration file for evaluation.

:::{admonition} Recommended Practice
:class: tip

For details on how to write the following custom configuration file, please refer to [Custom Configuration Files](../../advanced_tutorials/run_custom_config.md#custom-configuration-file-examples-for-each-scenario). Using a custom configuration file can support richer custom parameter configurations, such as supporting `num_prompts`, `request_rate` (QPS sending mode), etc.
:::

### One-Click Evaluation

After the service is started, the following **custom configuration file** can be used to send the `ShareGPT` dataset to the service at `request_rate=1` (QPS) for performance evaluation.

- Configuration file content:
  ```python
  from mmengine.config import read_base

  with read_base():
      from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
      from ais_bench.benchmark.configs.datasets.sharegpt.sharegpt_gen import sharegpt_datasets as datasets
      from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

  models = vllm_api_stream_chat
  models[0]["host_ip"] = "localhost"
  models[0]["host_port"] = 8080
  models[0]["max_out_len"] = 1024
  models[0]["batch_size"] = 50
  models[0]["request_rate"] = 1  # Request sending frequency: send 1 request to the server every 1/request_rate seconds; if less than 0.001, all requests are sent at once
  models[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)  # When testing performance and needing to limit the output length, ignore_eos must be set to True

  work_dir = "outputs/default/"
  ```

- Execution command (you can also append `--num-prompts N` to limit the number of requests sent):
  ```bash
  ais_bench ais_bench/configs/performance_benchmark/performance_qwen2_7b_sharegpt.py
  ```

After the task is completed, you can view the performance result report in the `summary/` directory under the task output directory.

### Command Meaning

The meaning of the AISBench service-oriented performance evaluation command is the same as explained in 📚 [Tool Quick Start/Command Meaning](../../get_started/quick_start.md#start-evaluation-choose-one-of-two-methods). On this basis, you need to add `--mode perf` or `-m perf` to enter the performance evaluation scenario. Take the following AISBench command as an example:

```shell
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer default_perf --mode perf
```

Among them:

- `--models` specifies the model task, i.e., the `vllm_api_stream_chat` model task.
- `--datasets` specifies the dataset task, i.e., the `demo_gsm8k_gen_4_shot_cot_chat_prompt` dataset task.
- `--summarizer` specifies the result presentation task, i.e., the `default_perf` result presentation task (if `--summarizer` is not specified, the `default_perf` task is used by default in performance evaluation scenarios). It is generally used by default and does not need to be specified in the command line; subsequent commands will omit this parameter.

### Task Meaning Query (Optional)

Specific information (introduction, usage constraints, etc.) about the selected model task `vllm_api_stream_chat`, dataset task `demo_gsm8k_gen_4_shot_cot_chat_prompt`, and result presentation task `default_perf` can be queried from the following links:

- `--models`: 📚 [Service-Oriented Inference Backend](../all_params/models.md#service-oriented-inference-backend)
- `--datasets`: 📚 [Open-Source Datasets](../../get_started/datasets.md#open-source-datasets) → 📚 [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/demo/README_en.md)
- `--summarizer`: 📚 [Result Summary Tasks](../all_params/summarizer.md#supported-result-summary-tasks)

### Preparations Before Running the Command

- `--models`: To use the `vllm_api_stream_chat` model task, you need to prepare an inference service that supports the `v1/chat/completions` sub-service. You can refer to 🔗 [VLLM Launch OpenAI-Compatible Server](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html#openai-compatible-server) to start the inference service.
- `--datasets`: To use the `demo_gsm8k_gen_4_shot_cot_chat_prompt` dataset task, you need to prepare the GSM8K dataset, which can be downloaded from 🔗 [GSM8K Dataset Compressed Package Provided by OpenCompass](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip). Deploy the unzipped `gsm8k/` folder to the `ais_bench/datasets` folder in the root path of the AISBench evaluation tool.

### Modification of Configuration Files Corresponding to Tasks

Each model task, dataset task, and result presentation task corresponds to a configuration file. The content of these configuration files must be modified before executing commands. The paths of these configuration files can be queried by adding `--search` to the original AISBench command. For example:

```shell
# Note: Whether to add "--mode perf" to the search command does not affect the search results
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --mode perf --search
```

> ⚠️ **Note**: Executing a command with the `search` option will print the absolute path of the configuration file corresponding to the task.

Executing the query command will yield the following results:

```shell
╒══════════════╤═══════════════════════════════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
│ Task Type    │ Task Name                             │ Config File Path                                                                                                               │
╞══════════════╪═══════════════════════════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
│ --models     │ vllm_api_stream_chat                  │ /your_workspace/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py                                 │
├──────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ --datasets   │ demo_gsm8k_gen_4_shot_cot_chat_prompt │ /your_workspace/benchmark/ais_bench/benchmark/configs/datasets/demo/demo_gsm8k_gen_4_shot_cot_chat_prompt.py                   │
╘══════════════╧═══════════════════════════════════════╧════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛
```

- The dataset task configuration file `demo_gsm8k_gen_4_shot_cot_chat_prompt.py` in the quick start does not require additional modifications. For an introduction to the content of the dataset task configuration file, please refer to 📚 [Configure Open-Source Datasets](../../get_started/datasets.md#configuring-open-source-datasets)

The model configuration file `vllm_api_stream_chat.py` contains configuration content related to model operation and needs to be modified according to actual conditions. The content that needs to be modified in the quick start is marked with comments.

```python
from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="",                    # Specify the absolute path of the model serialized vocabulary file (generally not required for accuracy testing scenarios)
        model="",        # Specify the name of the model loaded on the server, configured according to the actual model name pulled by the VLLM inference service (configuring an empty string will automatically retrieve it)
        stream=True, # Service performance only supports evaluating streaming interfaces
        request_rate=0,           # Request sending frequency: send 1 request to the server every 1/request_rate seconds; if less than 0.1, all requests are sent at once
        use_timestamp=False,      # Whether to schedule requests by dataset timestamp; used with timestamped datasets (e.g. Mooncake Trace)
        retry=2,                  # Maximum number of retries for each request
        api_key="",               # Custom API key, default is an empty string
        host_ip="localhost",      # Specify the IP of the inference service
        host_port=8080,           # Specify the port of the inference service
        url="",                     # Custom URL path for accessing the inference service (required when the base URL is not a combination of http://host_ip:host_port; host_ip and host_port will be ignored after configuration)
        max_out_len=512,          # Maximum number of tokens output by the inference service
        batch_size=1,               # Maximum concurrency for sending requests
        trust_remote_code=False,    # Whether the tokenizer trusts remote code, default is False;
        generation_kwargs=dict(   # Model inference parameters, configured with reference to VLLM documentation; the AISBench evaluation tool does not process them and attaches them to the sent request
            temperature=0.01,
            ignore_eos=True, # When testing performance and needing to limit the output length, ignore_eos must be set to True
        )
    )
]
```


### Execute Commands
After modifying the configuration files, execute the command to start the service performance evaluation:
```bash
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt -m perf
```

After executing the AISBench command, the status of the ongoing task will be displayed on a real-time refreshing dashboard in the command line (press the "P" key on the keyboard to stop refreshing for copying dashboard information, and press "P" again to resume refreshing). For example:

```
Base path of result&log : outputs/default/20251106_103326
Task Progress Table (Updated at: 2025-11-06 10:34:41)
Page: 1/1  Total 2 rows of data
Press Up/Down arrow to page,  'P' to PAUZE/RESUME screen refresh, 'Ctrl + C' to exit

+---------------------------------+-----------+-------------------------------------------------+-------------+-------------+------------------------------------------------+------------------------------------------------+
| Task Name                       |   Process | Progress                                        | Time Cost   | Status      | Log Path                                       | Extend Parameters                              |
+=================================+===========+=================================================+=============+=============+================================================+================================================+
| vllm-api-stream-chat/demo_gsm8k |    744887 | [###########                   ] 3/8 [0.1 it/s] | 0:00:54     | inferencing | logs/infer/vllm-api-stream-chat/demo_gsm8k.out | {'POST': 4, 'RECV': 3, 'FINISH': 3, 'FAIL': 0} |
+---------------------------------+-----------+-------------------------------------------------+-------------+-------------+------------------------------------------------+------------------------------------------------+
```

Detailed logs of task execution will be continuously saved to the default output path, which is displayed on the real-time refreshing dashboard as `Log Path`. The `Log Path` (`logs/infer/vllm-api-stream-chat/demo_gsm8k.out`) is a subpath under the `Base path` (`outputs/default/20251106_103326`). Taking the above dashboard information as an example, the path to the detailed logs of task execution is:

```shell
# {Base path}/{Log Path}
outputs/default/20251106_103326/logs/infer/vllm-api-stream-chat/demo_gsm8k.out
```

> 💡 If you want detailed logs to be printed directly during execution, you can add `--debug` to the command:
>
> ```bash
> ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt -m perf --debug
> ```

### View Performance Results

The on-screen performance results are displayed as follows:

```bash
[2025-11-06 10:35:43,667] [ais_bench] [INFO] Performance Results of task: vllm-api-stream-chat/demo_gsm8k:
╒══════════════════════════╤═════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════╕
│ Performance Parameters   │ Stage   │ Average         │ Min             │ Max             │ Median          │ P75             │ P90             │ P99             │  N  │
╞══════════════════════════╪═════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════╡
│ E2EL                     │ total   │ 12300.2 ms      │ 12295.9 ms      │ 12305.2 ms      │ 12300.0 ms      │ 12302.1 ms      │ 12304.3 ms      │ 12305.1 ms      │  8  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ TTFT                     │ total   │ 2006.0 ms       │ 2005.1 ms       │ 2007.4 ms       │ 2006.1 ms       │ 2006.2 ms       │ 2006.6 ms       │ 2007.3 ms       │  8  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ TPOT                     │ total   │ 20.1 ms         │ 20.1 ms         │ 20.2 ms         │ 20.1 ms         │ 20.1 ms         │ 20.2 ms         │ 20.2 ms         │  8  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ ITL                      │ total   │ 20.1 ms         │ 19.8 ms         │ 21.3 ms         │ 20.1 ms         │ 20.2 ms         │ 20.2 ms         │ 20.4 ms         │  8  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ InputTokens              │ total   │ 1512.5          │ 1481.0          │ 1566.0          │ 1511.5          │ 1520.25         │ 1536.6          │ 1563.06         │  8  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ OutputTokens             │ total   │ 512.0           │ 512.0           │ 512.0           │ 512.0           │ 512.0           │ 512.0           │ 512.0           │  8  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ OutputTokenThroughput    │ total   │ 41.6254 token/s │ 41.6085 token/s │ 41.6398 token/s │ 41.6261 token/s │ 41.6338 token/s │ 41.6375 token/s │ 41.6395 token/s │  8  │
╘══════════════════════════╧═════════╧═════════════════╧═════════════════╧═════════════════╧═════════════════╧═════════════════╧═════════════════╧═════════════════╧═════╛
╒══════════════════════════╤═════════╤══════════════════╕
│ Common Metric            │ Stage   │ Value            │
╞══════════════════════════╪═════════╪══════════════════╡
│ Benchmark Duration       │ total   │ 98409.4916 ms    │
├──────────────────────────┼─────────┼──────────────────┤
│ Total Requests           │ total   │ 8                │
├──────────────────────────┼─────────┼──────────────────┤
│ Failed Requests          │ total   │ 0                │
├──────────────────────────┼─────────┼──────────────────┤
│ Success Requests         │ total   │ 8                │
├──────────────────────────┼─────────┼──────────────────┤
│ Concurrency              │ total   │ 0.9999           │
├──────────────────────────┼─────────┼──────────────────┤
│ Max Concurrency          │ total   │ 1                │
├──────────────────────────┼─────────┼──────────────────┤
│ Request Throughput       │ total   │ 0.0813 req/s     │
├──────────────────────────┼─────────┼──────────────────┤
│ Total Input Tokens       │ total   │ 12100            │
├──────────────────────────┼─────────┼──────────────────┤
│ Prefill Token Throughput │ total   │ 753.9843 token/s │
├──────────────────────────┼─────────┼──────────────────┤
│ Total Generated Tokens   │ total   │ 4096             │
├──────────────────────────┼─────────┼──────────────────┤
│ Input Token Throughput   │ total   │ 122.9556 token/s │
├──────────────────────────┼─────────┼──────────────────┤
│ Output Token Throughput  │ total   │ 41.622 token/s   │
├──────────────────────────┼─────────┼──────────────────┤
│ Total Token Throughput   │ total   │ 164.5776 token/s │
╘══════════════════════════╧═════════╧══════════════════╛
[2025-11-06 10:35:43,672] [ais_bench] [INFO] Performance Result files located in outputs/default/20251106_103326/performances/vllm-api-stream-chat.
```

💡 For the meaning of specific performance parameters, refer to 📚 [Performance Evaluation Results Description](../results_intro/performance_metric.md)

### Performance Details View

After executing the AISBench command, more details of task execution will eventually be saved to the `Base path` (`outputs/default/20251106_103326`).

After the command execution ends, the task execution details in `outputs/default/20250628_151326` are as follows:

```shell
20251106_103326          # Unique directory generated based on timestamp for each experiment
├── configs               # Automatically stored configuration files of all dumped configurations
├── logs                  # Logs during execution; if --debug is added to the command, there will be no on-disk logs (all printed directly)
│   └── performance/      # Log files from the inference phase
└── performance           # Performance evaluation results
    └── vllm-api-stream-chat/          # "Service-oriented model configuration" name, corresponding to the abbr parameter of models in the model task configuration file
        ├── demo_gsm8k.csv          # Single-request performance output (CSV), consistent with the Performance Parameters table in the on-screen performance results
        ├── demo_gsm8k.json         # End-to-end performance output (JSON), consistent with the Common Metric table in the on-screen performance results
        ├── demo_gsm8k_plot.html    # Request concurrency visualization report (HTML)
        └── ......
```

💡 The `demo_gsm8k_plot.html` request concurrency visualization report is recommended to be opened with browsers such as Chrome or Edge, where you can see the latency of each request and the number of concurrent service requests perceived by the client at each moment:
![full_plot_example](../../img/request_concurrency/full_plot_example.png)

For instructions on using this HTML visualization file, please refer to 📚 [Instructions for Using Performance Test Visualization Concurrency Graphs](../results_intro/performance_visualization.md)

## Test Preparation

Before performing service-oriented inference, the following conditions must be met:

- Available model weights: Ensure that the model weight files to be tested are already available locally. Open-source weights can be obtained from 🔗 [Hugging Face Community](https://huggingface.co/models).
- Service environment preparation: Ensure that the model inference service is started through inference engines such as vLLM/vLLM-Ascend. The startup parameters need to ensure that the server's `max-model-len` and other configurations can accommodate the length of the prompt and output to be sent.
- Dataset preparation: Select a dataset suitable for performance evaluation scenarios, such as `ShareGPT`. For details, refer to 📚 [Datasets](../../get_started/datasets.md#open-source-datasets). The user can also prepare a custom dataset, see [Custom Dataset Evaluation](../../advanced_tutorials/custom_dataset.md).
- Model task preparation: Select the model task to execute from 📚 [vLLM Model Backend](../all_params/models.md#service-oriented-inference-backend).

:::{admonition} Service Startup Precautions
:class: warning

- It is recommended to ensure that the service is fully started before starting the evaluation task, otherwise the task may fail due to connection failure.
- When the service fails, the tool will record the failure cause in the logs, and the user can troubleshoot based on the error information.
:::

## Main Functional Scenarios

### Single-Task Performance Evaluation

#### Using a Custom Configuration File (Recommended)

:::{tab-set}
:::{tab-item} ⭐ Custom Configuration File

The configuration file content is consistent with the [Quick Start One-Click Evaluation](#one-click-evaluation).

Execution command:

```bash
ais_bench ais_bench/configs/performance_benchmark/performance_qwen2_7b_sharegpt.py
```

:::
:::{tab-item} Alternative: Command-Line Parameters

You can also use the preset configuration file for one-click evaluation (the service address needs to be configured in the model configuration file [vllm_api_stream_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py)):

```bash
ais_bench --models vllm_api_stream_chat --datasets sharegpt_gen -m perf
```

:::
:::

#### Specifying Custom Performance Dimensions

AISBench supports users in customizing the statistical items of performance reports. In the custom configuration file, you can customize the `summarizer` and modify the `stats_list` field of its calculator to control which statistical dimensions (e.g., `Average`, `Min`, `Max`, `Median`, `P75`, `P90`, `P99`) are computed for each performance parameter (E2EL, TTFT, TPOT, etc.) in the summary report.

Commonly used statistical items include:

| Statistic | Description |
| --- | --- |
| `Average` | Average value |
| `Min` | Minimum value |
| `Max` | Maximum value |
| `Median` | Median value |
| `P75` | 75th percentile |
| `P90` | 90th percentile |
| `P95` | 95th percentile |
| `P99` | 99th percentile |

The following is an example configuration that contains the most commonly used statistical items:

```python
from mmengine.config import read_base
from ais_bench.benchmark.summarizers import DefaultPerfSummarizer
from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator

with read_base():
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

# Key: customize the stats_list of the performance summarizer calculator
summarizer = dict(
    attr="performance",
    type=DefaultPerfSummarizer,
    calculator=dict(
        type=DefaultPerfMetricCalculator,
        stats_list=["Average", "Min", "Max", "Median", "P75", "P90", "P95", "P99"],
    ),
)

models = vllm_api_stream_chat
```

For a complete runnable example, refer to [performance_re_eval.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_re_eval.py).

### Multi-Task Performance Evaluation

Supports simultaneous configuration of multiple datasets or multiple sending parameter combinations (such as different `request_rate`s) for performance evaluation through a single command, facilitating the comparison of performance indicators of different sending strategies.

#### Description of Sub-task Combinations

In multi-task evaluation scenarios, the number of subtasks is the product of the number of tasks configured by `models` and the number of tasks configured by `datasets`—that is, one model configuration and one dataset configuration form a subtask.

The following example simultaneously evaluates 2 model tasks (`vllm_api_general_stream`, `vllm_api_stream_chat`) and 2 dataset tasks (`gsm8k_gen_4_shot_cot_str`, `aime2024_gen_0_shot_str`), and will execute the following 4 combined performance test tasks:

+ [vllm_api_general_stream](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py) Model Task + [gsm8k_gen_4_shot_cot_str](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/gsm8k/gsm8k_gen_4_shot_cot_str.py) Dataset Task
+ [vllm_api_general_stream](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py) Model Task + [aime2024_gen_0_shot_str](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/aime2024/aime2024_gen_0_shot_str) Dataset Task
+ [vllm_api_stream_chat](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py) Model Task + [gsm8k_gen_4_shot_cot_str](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/gsm8k/gsm8k_gen_4_shot_cot_str.py) Dataset Task
+ [vllm_api_stream_chat](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py) Model Task + [aime2024_gen_0_shot_str](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/aime2024/aime2024_gen_0_shot_str.py) Dataset Task

#### Custom Model-Dataset Pairings (Optional)

By default, the `models` list and `datasets` list in the configuration file are automatically combined as a Cartesian product, with the number of subtasks equal to the number of models × the number of datasets (in this example, 2 × 2 = 4). If you want to precisely control which models are paired with which datasets (e.g., letting some models only run on some datasets to avoid meaningless combinations), you can explicitly declare the pairing relationship in the configuration file via the `model_dataset_combinations` field:

```python
from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_str import gsm8k_datasets
    from ais_bench.benchmark.configs.datasets.aime2024.aime2024_gen_0_shot_str import aime2024_datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as vllm_api_general_stream
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

datasets = gsm8k_datasets + aime2024_datasets
models = vllm_api_general_stream + vllm_api_stream_chat

# Key: Precisely control pairings via model_dataset_combinations
# The following example generates only 2 subtasks (the Cartesian product would generate 4):
#   - vllm_api_general_stream + gsm8k_gen_4_shot_cot_str
#   - vllm_api_stream_chat + aime2024_gen_0_shot_str
model_dataset_combinations = [
    dict(models=[models[0]], datasets=[datasets[0]]),
    dict(models=[models[1]], datasets=[datasets[1]]),
]
```

> ⚠️ **Note**: The unique identifier for models and datasets is determined by the `abbr` field. In the same configuration file, repeated combinations of models or datasets with the same `abbr` will be treated as duplicate tasks and skipped. When reusing model/dataset configurations via methods such as `.copy()`, the `abbr` must be explicitly modified to ensure uniqueness. See 📚 [Custom Model and Dataset Combinations](../../advanced_tutorials/run_custom_config.md#custom-model-and-dataset-combinations) for details.

#### Multi-Task Parallel

Supports multi-task parallelism through the [`--max-num-workers`](../all_params/cli_args.md#common-parameters) command-line parameter. Different sub-tasks will be distributed to different processes for parallel execution.

#### Specifying Multiple Datasets for Performance Evaluation

:::{tab-set}
:::{tab-item} ⭐ Custom Configuration File

```python
from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_str import gsm8k_datasets
    from ais_bench.benchmark.configs.datasets.aime2024.aime2024_gen_0_shot_chat_prompt import aime2024_datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

datasets = gsm8k_datasets + aime2024_datasets

models = vllm_api_stream_chat
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8080
models[0]["max_out_len"] = 512
models[0]["batch_size"] = 1
models[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)

work_dir = "outputs/default/"
```

Execution command:

```bash
ais_bench ais_bench/configs/performance_benchmark/performance_multi_dataset.py
```

:::
:::{tab-item} Alternative: Command-Line Parameters

Use the `--datasets` parameter to specify multiple datasets:

```bash
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_4_shot_cot_str,aime2024_gen_0_shot_chat_prompt -m perf
```

:::
:::

#### Specifying Multiple Sending Rates for Performance Evaluation

The following configuration file example sends the `ShareGPT` dataset to the service at `request_rate=1, 2, 4, 8` respectively for performance evaluation.

:::{tab-set}
:::{tab-item} ⭐ Custom Configuration File

```python
import copy
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.sharegpt.sharegpt_gen import sharegpt_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as base_vllm_api_stream_chat

# In AISBench, `request_rate` is a field of the model configuration, so build one
# model configuration per rate via `copy.deepcopy` and combine them with one dataset.
models = []
for rate in [1, 2, 4, 8]:
    model_cfg = copy.deepcopy(base_vllm_api_stream_chat[0])
    model_cfg["abbr"] = f"vllm-api-stream-chat-rate-{rate}"
    model_cfg["host_ip"] = "localhost"
    model_cfg["host_port"] = 8080
    model_cfg["max_out_len"] = 1024
    model_cfg["batch_size"] = 50
    model_cfg["request_rate"] = rate
    model_cfg["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)
    models.append(model_cfg)

work_dir = "outputs/default/"
```

Execution command:

```bash
ais_bench ais_bench/configs/performance_benchmark/performance_multi_rate.py
```

:::
:::{tab-item} Alternative: Command-Line Parameters

It is not supported to specify multiple sending rates for one dataset in a single command. It is recommended to use a custom configuration file.

:::
:::

#### Specifying Multiple Models for Performance Evaluation

Supports simultaneous evaluation of multiple models on the same dataset, suitable for comparing the performance of different models.

:::{tab-set}
:::{tab-item} ⭐ Custom Configuration File

```python
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.sharegpt.sharegpt_gen import sharegpt_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as vllm_api_general_stream

# Rename the abbr of each model so that the results are distinguishable
vllm_api_stream_chat[0]["abbr"] = "vllm-qwen2.5-7b"
vllm_api_general_stream[0]["abbr"] = "vllm-qwen2.5-14b"

vllm_api_stream_chat[0]["host_ip"] = "localhost"
vllm_api_stream_chat[0]["host_port"] = 8080
vllm_api_stream_chat[0]["max_out_len"] = 1024
vllm_api_stream_chat[0]["batch_size"] = 50
vllm_api_stream_chat[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)

vllm_api_general_stream[0]["host_ip"] = "localhost"
vllm_api_general_stream[0]["host_port"] = 8081
vllm_api_general_stream[0]["max_out_len"] = 1024
vllm_api_general_stream[0]["batch_size"] = 50
vllm_api_general_stream[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)

models = vllm_api_stream_chat + vllm_api_general_stream

work_dir = "outputs/default/"
```

Execution command:

```bash
ais_bench ais_bench/configs/performance_benchmark/performance_multi_model.py
```

:::
:::{tab-item} Alternative: Command-Line Parameters

```bash
ais_bench --models vllm_api_stream_chat,vllm_api_general_stream --datasets sharegpt_gen -m perf
```

:::
:::

### Synthetic Dataset Multi-Task Combinations

In actual performance evaluation, it is sometimes necessary to simulate the input load in production environments, such as fixed-length inputs, Poisson-distributed request arrival, etc. AISBench supports users in defining custom performance evaluation datasets through the `SyntheticDataset`, and supports configuring the distribution of input sequence lengths, the distribution of output sequence lengths, the request arrival rate (QPS), etc. through parameters. The model-dataset sub-tasks generated by the synthetic dataset support combinations with each other.

The following configuration file example sends synthetic datasets of different input lengths to the service for performance evaluation at `request_rate=2`:

:::{tab-set}
:::{tab-item} ⭐ Custom Configuration File

```python
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import synthetic_datasets as base_synthetic_datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

# Define multiple sub-datasets with different input/output lengths via the `config` field
datasets = []
for input_len in [256, 512, 1024]:
    for output_len in [256, 512]:
        ds = dict(base_synthetic_datasets[0])
        ds["abbr"] = f"syn_in{input_len}_out{output_len}"
        ds["config"] = {
            "Type": "string",
            "RequestCount": 100,
            "TrustRemoteCode": False,
            "StringConfig": {
                "Input": {
                    "Method": "uniform",
                    "Params": {"MinValue": input_len, "MaxValue": input_len},
                },
                "Output": {
                    "Method": "uniform",
                    "Params": {"MinValue": output_len, "MaxValue": output_len},
                },
            },
        }
        datasets.append(ds)

models = vllm_api_stream_chat
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8080
models[0]["max_out_len"] = 512
models[0]["batch_size"] = 1
models[0]["request_rate"] = 2
models[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)

work_dir = "outputs/default/"
```

Execution command:

```bash
ais_bench ais_bench/configs/performance_benchmark/performance_synthetic.py
```

:::
:::{tab-item} Alternative: Command-Line Parameters

It is not supported to specify multiple synthetic datasets with different lengths in a single command. It is recommended to use a custom configuration file.

:::
:::

> 💡 For more configuration details of `SyntheticDataset`, please refer to 📚 [Datasets](../../get_started/datasets.md#randomly-synthesized-datasets).

### Custom Sequence Length Usage through Custom Config File Approach

:::{admonition} Why use a custom config file?
:class: tip

For the synthetic dataset scenario, in order to fully support the user's combination of multiple different input/output lengths, multiple different QPS sending rates, etc., it is **strongly recommended to use a custom configuration file**, because the command-line parameters can only support a single fixed length and a single QPS, and cannot satisfy the combinatorial requirements.
:::

For detailed instructions on writing custom configuration files, please refer to [Custom Configuration Files](../../advanced_tutorials/run_custom_config.md#4-synthetic-dataset-performance-evaluation).

### Custom Sequence Multi-Task Combinations

For multi-task combinations based on custom sequence lengths, the user can combine different models and datasets for evaluation through the `model_dataset_combinations` field.

:::{tab-set}
:::{tab-item} ⭐ Custom Configuration File

```python
from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import synthetic_datasets as base_synthetic_datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

datasets = []
for input_len in [256, 512]:
    for output_len in [256, 512]:
        ds = dict(base_synthetic_datasets[0])
        ds["abbr"] = f"syn_in{input_len}_out{output_len}"
        ds["config"] = {
            "Type": "string",
            "RequestCount": 100,
            "TrustRemoteCode": False,
            "StringConfig": {
                "Input": {
                    "Method": "uniform",
                    "Params": {"MinValue": input_len, "MaxValue": input_len},
                },
                "Output": {
                    "Method": "uniform",
                    "Params": {"MinValue": output_len, "MaxValue": output_len},
                },
            },
        }
        datasets.append(ds)

models = vllm_api_stream_chat
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8080
models[0]["max_out_len"] = 512
models[0]["batch_size"] = 1
models[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)

# Key: Only specify partial models for partial datasets
model_dataset_combinations = [
    dict(models=[models[0]], datasets=[datasets[0], datasets[1]]),
    dict(models=[models[0]], datasets=[datasets[2]]),
]

work_dir = "outputs/default/"
```

Execution command:

```bash
ais_bench ais_bench/configs/performance_benchmark/performance_seq_combinations.py
```

:::
:::{tab-item} Alternative: Command-Line Parameters

Not supported.

:::
::::

### Fixed Request Count Evaluation

When the dataset scale is too large and you only want to perform performance testing on a subset of samples, you can use either of the following two approaches to control the data reading range. They achieve the same goal, so just pick the one that fits your habit:

- **Basic approach**: Specify the number of data entries to read directly via the command-line parameter 📚 [`--num-prompts`](../all_params/cli_args.md#common-parameters). No configuration file modification is required, and it is the simplest to use.
- **Advanced approach (more powerful)**: Set the `reader_cfg.test_range` field of the dataset in the custom configuration file, which supports a more flexible sampling range (e.g., specifying a start index and custom step). For detailed usage, refer to 📚 [Custom Configuration Files](../../advanced_tutorials/run_custom_config.md).

Example as follows:

::::{tab-set}
:::{tab-item} ⭐ Recommended: Using a Custom Configuration File

**Method 1: Basic approach — Use `--num-prompts` to specify the number of entries to read**

For a complete example, refer to [performance_fixed_request.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_fixed_request.py):

```python
from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

models = vllm_api_stream_chat
# ...For other parameter configurations, please refer to the configuration file
```

Execute the command (specify reading only 1 sample via `--num-prompts 1`):

```bash
ais_bench ais_bench/configs/performance_benchmark/performance_fixed_request.py --mode perf --num-prompts 1
```

**Method 2: Advanced approach — Use `test_range` to flexibly specify the reading range**

If you need more flexible range control (e.g., specifying a start index and custom step), you can set the `reader_cfg.test_range` field of the dataset directly in the custom configuration file, without passing any command-line parameter. For a complete example, refer to [performance_fixed_request.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_fixed_request.py):

```python
from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

# Key: control the sampling range flexibly via reader_cfg.test_range
# For example, '[0:8]' reads the first 8 samples; '[10:20]' reads samples from index 10 to 20
datasets[0]['reader_cfg']['test_range'] = '[0:8]'

models = vllm_api_stream_chat
# ...For other parameter configurations, please refer to the configuration file
```

Execute the command (test_range has been specified in the configuration file, no need to pass `--num-prompts`):

```bash
ais_bench ais_bench/configs/performance_benchmark/performance_fixed_request.py --mode perf
```

:::
:::{tab-item} Alternative: Using Command-Line Parameters

```bash
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --mode perf --num-prompts 1
```
The above command only performs inference on the first entry in the sample dataset and only measures the performance of this one entry.

:::
::::

> ⚠️ Note: Currently, the dataset is read sequentially in the default queue order; random sampling or shuffling is not supported. When `reader_cfg.test_range` in the configuration file and the command-line `--num-prompts` are both specified, the command-line parameter `--num-prompts` takes precedence.

### Fixed Request Count Performance Evaluation

In some scenarios, the user wants to fix the total number of requests sent without limiting the sending rate, that is, to send requests at the maximum throughput. In this case, `request_rate` needs to be set to `-1`, indicating that requests are sent concurrently without rate limiting.

:::{tab-set}
:::{tab-item} ⭐ Custom Configuration File

```python
from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

models = vllm_api_stream_chat
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8080
models[0]["max_out_len"] = 512
models[0]["batch_size"] = 1
models[0]["request_rate"] = -1  # -1 indicates concurrent sending without rate limiting
models[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)

work_dir = "outputs/default/"
```

Execution command (you can append `--num-prompts 100` to fix the total number of requests):

```bash
ais_bench ais_bench/configs/performance_benchmark/performance_fixed_request.py --mode perf
```

:::
:::{tab-item} Alternative: Command-Line Parameters

```bash
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --mode perf --num-prompts 100
```

:::
:::

## Implementation via Custom Configuration Files

> 💡 All the above functional scenarios (multi-task evaluation, multi-task parallel, fixed request count, etc.) can be implemented through the [Custom Configuration File](../../advanced_tutorials/run_custom_config.md) approach. The configuration file is essentially a Python script, which supports all Python syntaxes such as loops, conditional judgments, and list comprehensions. Model, dataset, summarizer, and other configurations can be written into one file for one-time writing and multiple reuse.

All custom configuration file examples involved in this section are uniformly stored in the `ais_bench/configs/performance_benchmark/` directory for easy reference and reuse:

| File Name | Corresponding Scenario |
| --- | --- |
| [performance_qwen2_7b_sharegpt.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_qwen2_7b_sharegpt.py) | Single-Task Performance Evaluation |
| [performance_multi_dataset.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_multi_dataset.py) | Multi-Dataset Performance Evaluation |
| [performance_multi_rate.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_multi_rate.py) | Multi-Rate Performance Evaluation |
| [performance_multi_model.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_multi_model.py) | Multi-Model Performance Evaluation |
| [performance_synthetic.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_synthetic.py) | Synthetic Dataset Multi-Task Combinations |
| [performance_seq_combinations.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_seq_combinations.py) | Custom Sequence Multi-Task Combinations |
| [performance_fixed_request.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_fixed_request.py) | Fixed Request Count Performance Evaluation |
| [performance_re_eval.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_re_eval.py) | Performance Result Recalculation |

For details, refer to the "Service-Oriented Performance Evaluation" example in [Running AISBench via Custom Configuration Files](../../advanced_tutorials/run_custom_config.md#custom-configuration-file-examples-for-each-scenario).

## Other Functional Scenarios

### Speculative Decoding Metrics Collection

When running performance evaluation against a vLLM inference server with speculative decoding enabled, you can append `--spec-decode` to collect spec decode performance metrics (acceptance rate, acceptance length, etc.) from the server's Prometheus `/metrics` endpoint. The metrics are displayed alongside the standard performance results and saved to `spec_decode_*.json` under the `performances/` directory.

```shell
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --mode perf --spec-decode
```

For prerequisites, configuration details, and metric explanations, see 📚 [Speculative Decoding Metrics Collection](../../advanced_tutorials/spec_decode.md).

### Performance Result Recalculation

In the actual evaluation process, the user may want to update the performance summary based on the existing inference results, for example, after modifying the `stats_list` configuration, recalculate the summary report without re-running the inference.

AISBench supports recalculating performance summaries based on existing inference results through the `--mode perf` and `--reuse` parameters.

For a complete example, refer to [performance_re_eval.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/performance_benchmark/performance_re_eval.py):

```python
from mmengine.config import read_base
from ais_bench.benchmark.summarizers import DefaultPerfSummarizer
from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

# Key: customize the stats_list of the performance summarizer to control the
# statistical dimensions of the performance summary
summarizer = dict(
    attr="performance",
    type=DefaultPerfSummarizer,
    calculator=dict(
        type=DefaultPerfMetricCalculator,
        stats_list=["Average", "Min", "Max", "Median", "P75", "P90", "P95", "P99"],
    ),
)

models = vllm_api_stream_chat
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8080
models[0]["max_out_len"] = 512
models[0]["batch_size"] = 1
models[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)

work_dir = "outputs/default/"
```

Execution command (`--mode perf` and `--reuse` are common parameters, and can still be appended through the command line when using a custom configuration file):

```bash
ais_bench ais_bench/configs/performance_benchmark/performance_re_eval.py --mode perf --reuse 20250628_151326
```

## Specifications

The following specifications are required when using AISBench for performance evaluation:

| Item | Specification |
| --- | --- |
| Service Status | The service must be running normally, and the listening port is consistent with the `url` field in the configuration |
| `max-model-len` | Must be greater than or equal to `prompt length + output length`, otherwise the service will reject the request |
| Network | The evaluation machine needs to be able to access the service address normally |
| Concurrency | The number of concurrent evaluations should not exceed the service's processing capacity to avoid request timeout/failure |
| Output Directory | Each task generates a timestamp directory containing `configs/`, `logs/`, `predictions/`, `results/`, `summary/`, `performances/` |