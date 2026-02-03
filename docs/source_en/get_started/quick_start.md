# Quick Start
## Command Meaning
A single or multiple evaluation tasks executed by the AISBench command are defined by a combination of model tasks (single or multiple), dataset tasks (single or multiple), and result presentation tasks (single). Other command-line options of AISBench specify the scenario of the evaluation task (e.g., accuracy evaluation scenario, performance evaluation scenario). Take the following AISBench command as an example:
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer example
```
This command does not specify other command-line options, so it defaults to an accuracy evaluation scenario task, where:
- `--models` specifies the model task, i.e., the `vllm_api_general_chat` model task.

- `--datasets` specifies the dataset task, i.e., the `demo_gsm8k_gen_4_shot_cot_chat_prompt` dataset task.

- `--summarizer` specifies the result presentation task, i.e., the `example` result presentation task (if `--summarizer` is not specified, the `example` task is used by default in the accuracy evaluation scenario). It is generally recommended to use the default, so there is no need to specify it in the command line, and subsequent commands will omit it.

## Task Meaning Query (Optional)
The specific information (introduction, usage constraints, etc.) of the selected model task `vllm_api_general_chat`, dataset task `demo_gsm8k_gen_4_shot_cot_chat_prompt`, and result presentation task `example` can be queried from the following links respectively:
- `--models`: 📚 [Service-Oriented Inference Backend](../base_tutorials/all_params/models.md#service-oriented-inference-backend)

- `--datasets`: 📚 [Open Source Datasets](../base_tutorials/all_params/datasets.md#open-source-datasets) → 📚 [Detailed Introduction](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/demo/README_en.md)

- `--summarizer`: 📚 [Result Summary Tasks](../base_tutorials/all_params/summarizer.md)

## Preparations Before Running the Command
- `--models`: To use the `vllm_api_general_chat` model task, you need to prepare an inference service that supports the `v1/chat/completions` sub-service. You can refer to 🔗 [Launching an OpenAI-Compatible Server with VLLM](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html#openai-compatible-server) to start the inference service.
- `--datasets`: To use the `demo_gsm8k_gen_4_shot_cot_chat_prompt` dataset task, you need to prepare the gsm8k dataset, which can be downloaded from 🔗 [the gsm8k dataset zip package provided by opencompass](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip). Deploy the unzipped `gsm8k/` folder to the `ais_bench/datasets` folder in the root directory of the AISBench evaluation tool.

## Modification of Configuration Files Corresponding to Tasks
Each model task, dataset task, and result presentation task corresponds to a configuration file. You need to modify the content of these configuration files before running the command. The paths of these configuration files can be queried by adding `--search` to the original AISBench command. For example:
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --search
```
> ⚠️ **Note**: Executing the command with the `search` option will print the absolute paths of the configuration files corresponding to the tasks.

Executing the query command will yield the following results:
```shell
06/28 11:52:25 - AISBench - INFO - Searching configs...
╒══════════════╤═══════════════════════════════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
│ Task Type    │ Task Name                             │ Config File Path                                                                                                               │
╞══════════════╪═══════════════════════════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
│ --models     │ vllm_api_general_chat                 │ /your_workspace/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py                                 │
├──────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ --datasets   │ demo_gsm8k_gen_4_shot_cot_chat_prompt │ /your_workspace/benchmark/ais_bench/benchmark/configs/datasets/demo/demo_gsm8k_gen_4_shot_cot_chat_prompt.py                   │
╘══════════════╧═══════════════════════════════════════╧════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

```

- The dataset task configuration file `demo_gsm8k_gen_4_shot_cot_chat_prompt.py` in the quick start does not require additional modifications. For an introduction to the content of the dataset task configuration file, please refer to 📚 [Configuring Open Source Datasets](../base_tutorials/all_params/datasets.md#configuring-open-source-datasets).

The model configuration file `vllm_api_general_chat.py` contains configuration content related to model operation and needs to be modified according to the actual situation. The content that needs to be modified in the quick start is marked with comments.
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="",                    # Specify the absolute path of the model serialized vocabulary file (configuration is generally not required for accuracy testing scenarios).
        model="DeepSeek-R1",        # Specify the name of the model loaded on the server, configured according to the actual model name pulled by the VLLM inference service (configure as an empty string to get it automatically)
        stream=False,
        request_rate=0,           # Request sending frequency: send 1 request to the server every 1/request_rate seconds; if less than 0.1, all requests are sent at once
        use_timestamp=False,      # Whether to schedule requests by dataset timestamp; used with timestamped datasets (e.g. Mooncake Trace)
        retry=2,                  # Maximum number of retries per request
        api_key="",               # Custom API key, default is an empty string
        host_ip="localhost",      # Specify the IP of the inference service
        host_port=8080,           # Specify the port of the inference service
        url="",                     # Custom access path for the inference service (required when the base URL is not http://host_ip:host_port, and will ignore host_ip and host_port)
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
## Execute Command
After modifying the configuration file, run the following command to start the service-oriented accuracy evaluation:
```bash
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt
```

## View Task Execution Details
After executing the AISBench command, the status of the running task will be displayed on a real-time refreshing dashboard in the command line (press the "P" key to pause refreshing for copying dashboard information, and press "P" again to resume refreshing). Example:
```
Base path of result&log : outputs/default/20250628_151326
Task Progress Table (Updated at: 2025-11-06 10:08:21)
Page: 1/1  Total 2 rows of data
Press Up/Down arrow to page,  'P' to PAUSE/RESUME screen refresh, 'Ctrl + C' to exit

+----------------------------------+-----------+-------------------------------------------------+-------------+-------------+-------------------------------------------------+------------------------------------------------+
| Task Name                        |   Process | Progress                                        | Time Cost   | Status      | Log Path                                        | Extend Parameters                              |
+==================================+===========+=================================================+=============+=============+=================================================+================================================+
| vllm-api-general-chat/demo_gsm8k |    547141 | [###############               ] 4/8 [0.5 it/s] | 0:00:11     | inferencing | logs/infer/vllm-api-general-chat/demo_gsm8k.out | {'POST': 5, 'RECV': 4, 'FINISH': 4, 'FAIL': 0} |
+----------------------------------+-----------+-------------------------------------------------+-------------+-------------+-------------------------------------------------+------------------------------------------------+
```

Detailed logs of task execution are continuously written to the default output path, which is displayed on the real-time refreshing dashboard as `Log Path`. The `Log Path` (`logs/infer/vllm-api-general-chat/demo_gsm8k.out`) is located under the `Base path` (`outputs/default/20250628_151326`). Using the dashboard information above as an example, the path to the detailed task execution log is:
```shell
# {Base path}/{Log Path}
outputs/default/20250628_151326/logs/infer/vllm-api-general-chat/demo_gsm8k.out
```

> 💡 To print detailed logs directly during execution, add the `--debug` parameter to the command:
`ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug`

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