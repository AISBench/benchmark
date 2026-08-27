# Quick Start

## Preparations Before Running the Command

- An inference service that supports the `v1/chat/completions` sub-service is required. You can refer to 🔗 [Launching an OpenAI-Compatible Server with VLLM](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html#openai-compatible-server) to start the inference service.
- The gsm8k dataset is required, which can be downloaded from 🔗 [the gsm8k dataset zip package provided by opencompass](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip). Deploy the unzipped `gsm8k/` folder to the `ais_bench/datasets` folder in the root directory of the AISBench evaluation tool.

## Start Evaluation (Choose One of Two Methods)

| ⭐ Recommended: Using a Custom Configuration File | Alternative: Using Command-Line Arguments (Original Quick Start Method) |
| :--- | :--- |
| Modify a single file to centrally manage all configurations, with configuration written at any path | Specify via `--models` `--datasets` parameters |
| Write once, reuse multiple times | Each run requires inputting the full command |
| Supports all Python syntax for flexible extension | Only supports Cartesian product combinations |

::::{tab-set}
:::{tab-item} ⭐ Recommended: Using a Custom Configuration File

AISBench provides a pre-built custom configuration file [model_api_test_en.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/model_api_test_en.py), which centralizes common service-oriented inference test configurations (model selection, service address, port, generation parameters, etc.) in a single file, eliminating the need to find and modify multiple configuration files separately. This file is essentially a Python script that supports all Python syntax, allowing you to freely extend it.

Open `ais_bench/configs/model_api_test_en.py` and modify the following configurations according to the actual situation (If you installed the tool via `pip3 install ais_bench_benchmark`, you can create `model_api_test_en.py` at any path and write the following configuration content into that file):

```python
from mmengine.config import read_base

with read_base():
# Model tasks, choose one of them. For other model tasks, refer to: https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/all_params/models.html to obtain more model tasks
    # vllm_api_general is a base model that only supports text generation
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general
    # vllm_api_general_chat is a chat model that supports dialogue
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat
    # vllm_api_stream_chat is a streaming chat model that supports streaming dialogue
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat
    # vllm_api_general_stream is a streaming model that supports streaming generation
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as vllm_api_general_stream

# Dataset tasks, refer to: https://ais-bench-benchmark-rf.readthedocs.io/en/latest/get_started/datasets.html to obtain more dataset tasks
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets

models = vllm_api_general_chat

models[0]["path"] = ""  # Specify the absolute path to the model serialized vocabulary file (generally not required for accuracy testing scenarios)
models[0]["model"] = "" # Specify the name of the model loaded on the server, configured according to the actual model name pulled by the VLLM inference service (configure as an empty string to automatically retrieve it)
models[0]["request_rate"] = 0 # Request sending frequency: send 1 request to the server every 1/request_rate seconds; if less than 0.001, all requests are sent at once
models[0]["api_key"] = "" # Custom API key, default is an empty string
models[0]["host_ip"] = "localhost" # Specify the IP of the inference service
models[0]["host_port"] = 8080 # Specify the port of the inference service
models[0]["url"] = "" # Custom URL path for accessing the inference service (needs to be configured when the base URL is not a combination of http://host_ip:host_port; after configuration, host_ip and host_port will be ignored)
models[0]["max_out_len"] = 512 # Maximum number of tokens output by the inference service
models[0]["batch_size"] = 1 # Maximum concurrency for sending requests
models[0]["trust_remote_code"] = False # Whether the tokenizer trusts remote code, default is False
models[0]["generation_kwargs"] = dict( # Model inference parameters, configured with reference to the VLLM documentation; the AISBench evaluation tool does not process these parameters and attaches them directly to the sent requests
    temperature=0.01,
    ignore_eos=False,
)

# datasets[0]["path"] = ais_bench/datasets/gsm8k # Specify the absolute path of the dataset directory (required for accuracy testing scenarios)

work_dir = 'outputs/default/'  # Specify the working directory for saving task results and logs (default is outputs/default/)

```

> 💡 The configuration file already pre-imports commonly used model types (`vllm_api_general`, `vllm_api_general_chat`, `vllm_api_stream_chat`, `vllm_api_general_stream`), just uncomment/modify the relevant lines to switch. For more usages of custom configuration files, please refer to 📚 [Running AISBench with a Custom Configuration File](../advanced_tutorials/run_custom_config.md).

The selection, preparation, and usage of dataset tasks are described in the following steps:

1. Select a dataset task from 📚 [Open Source Datasets](https://ais-bench-benchmark.readthedocs.io/en/latest/get_started/datasets.html#open-source-datasets).
2. Go to the 📚 [Detailed Introduction / Dataset Deployment](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/demo/README_en.md#dataset-deployment) for the dataset to prepare the dataset.
3. Refer to 📚 [Detailed Introduction / Available Dataset Tasks](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/demo/README_en.md#available-dataset-tasks) to select an available dataset task, and copy the corresponding task import method (e.g., `from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets`) into the custom configuration file.

After modifying the configuration file, run the following command to start the service-oriented accuracy evaluation:

```bash
ais_bench ais_bench/configs/model_api_test_en.py
```

:::
:::{tab-item} Alternative: Using Command-Line Arguments

If you prefer the command-line argument approach, AISBench also supports specifying tasks directly via the `--models`, `--datasets`, `--summarizer` parameters. The following is the command-line approach that has **exactly the same execution effect** as the above custom configuration file approach.

A single or multiple evaluation tasks executed by the AISBench command are defined by a combination of model tasks (single or multiple), dataset tasks (single or multiple), and result presentation tasks (single). Take the following AISBench command as an example:

```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer example
```

This command does not specify other command-line options, so it defaults to an accuracy evaluation scenario task, where:
- `--models` specifies the model task, i.e., the `vllm_api_general_chat` model task.
- `--datasets` specifies the dataset task, i.e., the `demo_gsm8k_gen_4_shot_cot_chat_prompt` dataset task.
- `--summarizer` specifies the result presentation task, i.e., the `example` result presentation task (if `--summarizer` is not specified, the `example` task is used by default in the accuracy evaluation scenario). It is generally recommended to use the default, so there is no need to specify it in the command line.

For multi-task evaluation, please refer to: 📚 [Multi-Task Evaluation](../base_tutorials/scenes_intro/accuracy_benchmark.md#multi-task-evaluation) for accuracy scenarios and 📚 [Multi-Task Evaluation](../base_tutorials/scenes_intro/performance_benchmark.md#multi-task-performance-evaluation) for performance scenarios.

For more flexible evaluation methods with self-combined tasks, you can refer to: 📚 [Running AISBench with a Custom Configuration File](../advanced_tutorials/run_custom_config.md#running-aisbench-with-a-custom-configuration-file).

The specific information (introduction, usage constraints, etc.) of the selected model task `vllm_api_general_chat`, dataset task `demo_gsm8k_gen_4_shot_cot_chat_prompt`, and result presentation task `example` can be queried from the following links respectively:

- `--models`: 📚 [Service-Oriented Inference Backend](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/all_params/models.html#service-oriented-inference-backend)
- `--datasets`: 📚 [Open Source Datasets](https://ais-bench-benchmark.readthedocs.io/en/latest/get_started/datasets.html#open-source-datasets) → 📚 [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/demo/README_en.md)
- `--summarizer`: 📚 [Result Summary Tasks](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/all_params/summarizer.html)

Each model task, dataset task, and result presentation task corresponds to a configuration file. You need to modify the content of these configuration files before running the command. The paths of these configuration files can be queried by adding `--search` to the original AISBench command. For example:

```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --search
```

> ⚠️ **Note**: Executing the command with the `search` option will print the absolute paths of the configuration files corresponding to the tasks.

Executing the query command will yield the following results:

```shell
╒══════════════╤═══════════════════════════════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
│ Task Type    │ Task Name                             │ Config File Path                                                                                                               │
╞══════════════╪═══════════════════════════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
│ --models     │ vllm_api_general_chat                 │ /your_workspace/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py                                 │
├──────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ --datasets   │ demo_gsm8k_gen_4_shot_cot_chat_prompt │ /your_workspace/benchmark/ais_bench/benchmark/configs/datasets/demo/demo_gsm8k_gen_4_shot_cot_chat_prompt.py                   │
╘══════════════╧═══════════════════════════════════════╧════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

```

- The dataset task configuration file `demo_gsm8k_gen_4_shot_cot_chat_prompt.py` in the quick start does not require additional modifications. For an introduction to the content of the dataset task configuration file, please refer to 📚 [Configuring Open Source Datasets](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/all_params/datasets.html#configuring-open-source-datasets).

The model configuration file `vllm_api_general_chat.py` contains configuration content related to model operation and needs to be modified according to the actual situation. The content that needs to be modified in the quick start is marked with comments.

> 💡 **Tip**: Some parameters in the model config above (e.g. `host_ip`, `host_port`, `model`, `url`, `max_out_len`, `generation_kwargs`, etc.) can be overridden directly on the command line without editing the config file. For example:
>
> ```bash
> ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --host-ip 127.0.0.1 --host-port 8000
> ```
>
> An explicitly specified parameter overrides the corresponding field in **all executed model configs**; only fields **already present** in the config are overridden, and unspecified parameters keep their config-file values. For the full overridable parameter list and coverage notes, refer to 📚 [User Configuration Parameters - API Model Common Override Parameters](../base_tutorials/all_params/cli_args.md#api-model-common-override-parameters).

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="",                    # Specify the absolute path of the model serialized vocabulary file (configuration is generally not required for accuracy testing scenarios).
        model="",        # Specify the name of the model loaded on the server, configured according to the actual model name pulled by the VLLM inference service (configure as an empty string to get it automatically)
        stream=False,
        request_rate=0,           # Request sending frequency: send 1 request to the server every 1/request_rate seconds; if less than 0.1, all requests are sent at once
        use_timestamp=False,      # Whether to schedule requests by dataset timestamp; used with timestamped datasets (e.g. Mooncake Trace)
        retry=2,                  # Maximum number of retries per request
        api_key="",               # Custom API key, default is an empty string
        host_ip="localhost",      # Specify the IP of the inference service
        host_port=8080,           # Specify the port of the inference service
        url="",                     # Custom access path for the inference service (required when the base URL is not http://host_ip:host_port; after configuration, host_ip and host_port will be ignored)
        max_out_len=512,          # Maximum number of tokens output by the inference service
        batch_size=1,               # Maximum concurrency for sending requests
        trust_remote_code=False,    # Whether to trust remote code in the tokenizer, default False;
        generation_kwargs=dict(   # Model inference parameters shall be configured with reference to the VLLM documentation. The AISBench evaluation tool does not process these parameters, which will be included in the sent request.
            temperature=0.01,
            ignore_eos=False,
        )
    )
]
```

After modifying the configuration file, run the following command to start the service-oriented accuracy evaluation:

```bash
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt
```

:::
::::

## View Task Execution Details

After executing the AISBench command, the task management dashboard will refresh in real time in the command line to show the task execution status (press the "P" key to pause/resume refreshing for copying dashboard information, and press "P" again to continue refreshing). The task management dashboard supports monitoring the detailed execution status of multiple tasks simultaneously, including task name, progress, time cost, status, log path, extended parameters, and other information. For example:

```
Base path of result&log : outputs/default/20250628_151326
Task Progress Table (Updated at: 2025-11-06 10:08:21)
Page: 1/1  Total 2 rows of data
Press Up/Down arrow to page,  'P' to PAUZE/RESUME screen refresh, 'Ctrl + C' to exit

+----------------------------------+-----------+-------------------------------------------------+-------------+-------------+-------------------------------------------------+------------------------------------------------+
| Task Name                        |   Process | Progress                                        | Time Cost   | Status      | Log Path                                        | Extend Parameters                              |
+==================================+===========+=================================================+=============+=============+================================================+================================================+
| vllm-api-general-chat/demo_gsm8k |    547141 | [###############               ] 4/8 [0.5 it/s] | 0:00:11     | inferencing | logs/infer/vllm-api-general-chat/demo_gsm8k.out | {'POST': 5, 'RECV': 4, 'FINISH': 4, 'FAIL': 0} |
+----------------------------------+-----------+-------------------------------------------------+-------------+-------------+-------------------------------------------------+------------------------------------------------+

```

Detailed logs of task execution are continuously written to the default output path, which is displayed on the real-time refreshing dashboard as `Log Path`. The `Log Path` (`logs/infer/vllm-api-general-chat/demo_gsm8k.out`) is located under the `Base path` (`outputs/default/20250628_151326`). Using the dashboard information above as an example, the path to the detailed task execution log is:

```shell
# {Base path}/{Log Path}
outputs/default/20250628_151326/logs/infer/vllm-api-general-chat/demo_gsm8k.out
```

> 💡 To print detailed logs directly during execution, add the `--debug` parameter to the command:
> `ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug`

The `Base path` (`outputs/default/20250628_151326`) contains all task execution details. After the command completes, the full execution details are structured as follows:

```shell
20250628_151326/
├── configs # Combined configuration file for model tasks, dataset tasks, and structure presentation tasks
│   └── 20250628_151326_29317.py
├── logs # Execution logs (no process logs will be written to disk if --debug is added to the command; all logs are printed directly)
│   ├── eval
│   │   └── vllm-api-general-chat
│   │       └── demo_gsm8k.out # Logs of the accuracy evaluation process based on inference results in the predictions/ folder
│   └── infer
│       └── vllm-api-general-chat
│           └── demo_gsm8k.out # Inference process logs
├── predictions
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # Inference results (all outputs returned by the inference service)
├── results
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # Raw scores calculated from accuracy evaluation
└── summary
    ├── summary_20250628_151326.csv # Final accuracy scores (table format)
    ├── summary_20250628_151326.md # Final accuracy scores (Markdown format)
    └── summary_20250628_151326.txt # Final accuracy scores (text format)
```

> ⚠️ **Note**: The content of task execution details written to disk varies across different evaluation scenarios. Please refer to the guide for the specific evaluation scenario.

### Output Results

Since there are only 8 data samples, the results will be generated quickly. Example output:

```bash
dataset                 version  metric   mode  vllm_api_general_chat
----------------------- -------- -------- ----- ----------------------
demo_gsm8k              401e4c   accuracy gen                   62.50
```