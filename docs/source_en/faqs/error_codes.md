# Error Code Description
## TMAN-CMD-001
### Error Description
This error indicates that a required input parameter is missing when executing a command.
When launching the ais_bench evaluation tool via the command line, you must specify the model configuration and dataset configuration.

Examples of valid scenarios:
```bash
# When using an open-source dataset, you must specify the model task via `--models` and the dataset task via `--datasets`
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen
# When using a custom dataset, you must specify the model task via `--models` and the custom dataset path via `--custom_dataset_path`
ais_bench --models vllm_api_stream_chat --custom_dataset_path /path/to/custom/dataset
```

### Solution
Refer to the examples of valid scenarios to supplement the missing parameters.

## TMAN-CMD-002
### Error Description
This error indicates that the value of a command-line parameter is not within the valid range.

### Solution
Search this document for the specific command line that appears in the log, and find the constraints on parameter values specified in the command line description.<br>
For example, if this error occurs when executing `ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen --num-prompts -1 --mode perf`, search for `--num-prompts` in the document to find the constraints in the parameter description.

| Parameter | Description | Example |
| ---- | ---- | ---- |
| `--num-prompts` | Specifies the number of test cases to evaluate in the dataset. A positive integer must be entered. If the value exceeds the number of dataset cases or is not specified, the entire dataset will be evaluated. | `--num-prompts 500` |

The parameter description specifies that the value must be a positive integer (greater than 0).

## TMAN-CFG-001
### Error Description
There is a syntax error in the .py configuration file, causing parsing failure.

### Solution
Check the Python syntax errors in the configuration file printed in the log (all configurable files for the ais_bench evaluation tool follow Python syntax), such as missing quotation marks or mismatched parentheses, and correct them.

## TMAN-CFG-002
### Error Description
A required parameter is missing from the .py configuration file, causing parsing failure.
For example, the specific error log is: `Config file /path/to/vllm_api_stream_chat.py does not contain 'models' param!`, which indicates that the `models` parameter is missing from the configuration file.

A valid `vllm_api_stream_chat.py` file contains the `models` parameter:
```python
# ......
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-stream-chat",
        # ......
    )
]

```

### Solution
In the .py configuration file printed in the error log, add the parameter that the log indicates is missing.

## TMAN-CFG-003
### Error Description
A parameter in the .py configuration file has an incorrect type, causing parsing failure.
For example, the relevant configuration in the `vllm_api_stream_chat.py` configuration file is:
```python
# ......
models = dict(
    attr="service",
    type=VLLMCustomAPIChat,
    abbr="vllm-api-stream-chat",
    # ......
)
```

The specific error log is: `In config file /path/to/vllm_api_stream_chat.py, 'models' param must be a list!`, which indicates that the `models` parameter in the configuration file has an incorrect type. It should be a list type (but is actually a dictionary type).

### Solution
In the .py configuration file printed in the error log, correct the incorrect parameter type to the required type as indicated by the log.

## UTILS-MATCH-001
### Error Description
The task name specified via `--models`, `--datasets`, or `--summarizer` cannot be matched to a .py configuration file with the same name as the task.

### Solution
Check the task name that the log indicates cannot be matched. For example, if `xxxx` cannot be matched, the following log will be printed:
```
+------------------------+
| Not matched patterns   |
|------------------------|
| xxxx                   |
+------------------------+
```

#### Scenario 1: The configuration file folder path is not specified
First, execute `pip3 show ais_bench_benchmark | grep "Location:"` to check the installation path of the ais_bench evaluation tool. For example, the following information is obtained after execution:
```bash
Location: /usr/local/lib/python3.10/dist-packages
```

The configuration file path is then `/usr/local/lib/python3.10/dist-packages/ais_bench/benchmark/configs`. Navigate to this path and perform the following checks:
1. If the unmatchable task name is specified via `--models`, check whether there is a .py configuration file with the same name as the task in the `models/` path (including subdirectories).
2. If the unmatchable task name is specified via `--datasets`, check whether there is a .py configuration file with the same name as the task in the `datasets/` path (including subdirectories).
3. If the unmatchable task name is specified via `--summarizer`, check whether there is a .py configuration file with the same name as the task in the `summarizers/` path (including subdirectories).

#### Scenario 2: The configuration file folder path is specified
If you specified the configuration file folder path via `--config-dir` when executing the command, navigate to this path and perform the following checks:
1. If the unmatchable task name is specified via `--models`, check whether there is a .py configuration file with the same name as the task in the `models/` path (including subdirectories).
2. If the unmatchable task name is specified via `--datasets`, check whether there is a .py configuration file with the same name as the task in the `datasets/` path (including subdirectories).
3. If the unmatchable task name is specified via `--summarizer`, check whether there is a .py configuration file with the same name as the task in the `summarizers/` path (including subdirectories).

## UTILS-CFG-001
### Error Description
When using the [randomly synthesized dataset](../advanced_tutorials/synthetic_dataset.md) in the `tokenid` scenario, the model configuration file must specify the tokenizer path.

### Solution
Assume the ais_bench evaluation tool command is `ais_bench --models vllm_api_stream_chat --datasets synthetic_gen_tokenid --mode perf`. Then, all `path` parameters in the `models` section of the `vllm_api_stream_chat.py` configuration file (refer to [Modifying Configuration Files for Corresponding Tasks](../get_started/quick_start.md#Modifying Configuration Files for Corresponding Tasks) for the configuration file path retrieval method) must be set to the tokenizer path (usually the model weight folder path).

```python
# ......
models = dict(
    attr="service",
    type=VLLMCustomAPIChat,
    abbr="vllm-api-stream-chat",
    path="/path/to/tokenizer", # Enter the tokenizer path
    # ......
)
```

## UTILS-CFG-002
### Error Description
Initializing a model instance using parameters in the model configuration file failed due to invalid parameter content.

### Solution
Check the log for `build failed with the following errors:{error_content}`, and correct the parameters in the model configuration file according to the prompts in `error_content`.
For example, if the `batch_size` parameter value in the model configuration file is 100001, and `error_content` is `"batch_size must be an integer in the range (0, 100000]"`, this indicates that the `batch_size` parameter exceeds the valid range (0, 100000]. You need to correct the `batch_size` parameter value to 100000.

## UTILS-CFG-003
### Error Description
The value of a parameter in the model configuration file is outside the range limited by the tool.

### Solution
Configure the parameter value within the range limited by the tool according to the prompts in the detailed log. For example, if the configuration file content is:
```python
# In vllm_stream_api_chat.py
models = [
    dict(
        attr="service1",
        # ......
    )
]
```

The detailed error log is:
```bash
Model config contain illegal attr, 'attr' in model config is 'service1', only 'local' and 'service' are supported!
```

This indicates that the value of the `attr` parameter in the model configuration is `'service1'`, but the tool only supports the values `'local'` and `'service'`. You need to set the `attr` parameter to one of the valid values.

## UTILS-CFG-004
### Error Description
Some configuration items for model parameters must be consistent across all model configurations and cannot have different values.

### Solution
Unify the configuration values according to the prompts in the detailed log. For example, if the configuration file content is:
```python
# In vllm_stream_api_chat.py
models = [
    dict(
        attr="service",
        # ......
    ),
    dict(
        attr="local"
    )
]
```

The detailed error log is:
```bash
Cannot run local and service model together! Please check 'attr' parameter of models
```

Because the `models` configuration contains two parameter values: `'service'` and `'local'`, but the tool only supports a unified configuration of one value. Therefore, you need to set the `attr` parameter in the `models` configuration to either `'service'` or `'local'`.

## UTILS-CFG-008
### Error Description
The loaded multimodal dataset contains invalid content.

### Solution
1. If the error log is `Invalid dataset: /path/to/non-mm-dataset , please check whether the dataset is a MM-style dataset!`, it means the specified dataset `/path/to/non-mm-dataset` is not a valid multimodal dataset. Each piece of data in a valid dataset must contain at least a `type` or `path` field. If it contains a `type` field, the value of the `type` field must be one of `["image", "video", "audio"]`.
2. If the error log is `Param 'mm_type' does not match the data type of dataset: /path/to/mm-dataset , please check it!`, it means the specified dataset `/path/to/mm-dataset` is a valid multimodal dataset, but the value of the `mm_type` field in the prompt engineering configuration of the dataset configuration file is invalid. The valid values for the `mm_type` field must be one of `["image", "video", "audio"]`.

## UTILS-DEPENDENCY-001
### Error Description
A required dependency module is missing during execution.

### Solution
If the detailed error log is `Failed to import required modules. Please install the necessary packages: pip install math_verify`, follow the guidance in the detailed log and execute `pip install math_verify` to install the dependent library.

## UTILS-TYPE-001
### Error Description
No direct solution is available yet.

### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## UTILS-TYPE-002
### Error Description
No direct solution is available yet.

### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## UTILS-TYPE-003
### Error Description
No direct solution is available yet.

### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## UTILS-TYPE-004
### Error Description
No direct solution is available yet.

### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## UTILS-TYPE-005
### Error Description
No direct solution is available yet.

### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## UTILS-TYPE-006
### Error Description
No direct solution is available yet.
### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## UTILS-TYPE-007
### Error Description
No direct solution is available yet.
### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## UTILS-TYPE-008
### Error Description
The command-line parameter value is too large.
### Solution
If the error log shows `'--max-num-workers' must be <= 8, but got 9 ......`, it indicates that the value of the command-line parameter `--max-num-workers` is 9. However, the tool only supports a maximum of 8 concurrent workers. Therefore, you need to adjust the value of `--max-num-workers` to be ≤ 8.

## UTILS-TYPE-009
### Error Description
The command-line parameter value is not an integer type.
### Solution
If the error log shows `'--max-num-workers' must be an integer, but got '9' ......`, it indicates that the value of the command-line parameter `--max-num-workers` is the string '9'. However, the tool only supports integer-type values. Therefore, you need to correct the value of `--max-num-workers` to an integer type.

## UTILS-TYPE-010
### Error Description
The command-line parameter value is too small.
### Solution
If the error log shows `'--max-num-workers' must be >= 1, but got 0 ......`, it indicates that the value of the command-line parameter `--max-num-workers` is 0. However, the tool only supports at least 1 concurrent worker. Therefore, you need to adjust the value of `--max-num-workers` to be ≥ 1.

## UTILS-PARAM-001
### Error Description
No direct solution is available yet.
### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## UTILS-PARAM-002
### Error Description
In the custom dataset scenario, the `request_count` parameter in the configuration file `*.meta.json` is outside the valid range.
### Solution
If the error message is `Please make sure that the value of parameter 'request_count' can be converted to int(greater than 0).`, it means the `request_count` parameter in `*.meta.json` needs to be set to > 0.

## UTILS-PARAM-003
### Error Description
In the custom dataset scenario, the `min_value` parameter is greater than the `max_value` parameter in the configuration file `*.meta.json`.
### Solution
If the error message is `When the uniform distribution is set, parameter 'min_value' must be less than or equal to parameter 'max_value'.`, it means the `min_value` parameter in `*.meta.json` needs to be set to ≤ the `max_value` parameter. You need to correct it to `min_value` ≤ `max_value`.

## UTILS-PARAM-004
### Error Description
In the custom dataset scenario, the `min_value` and `max_value` parameters in the configuration file `*.meta.json` are outside the valid range.
### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## UTILS-PARAM-005
### Error Description
In the custom dataset scenario, the configuration file `*.meta.json` lacks required parameters.
### Solution
For example, if the error message is `When the uniform distribution is set, parameter 'min_value' and 'max_value' must be provided.`, it means that in the uniform distribution scenario, both the `min_value` and `max_value` parameters need to be set in `*.meta.json`.

## UTILS-PARAM-006
### Error Description
In the custom dataset scenario, the `percentage_distribute` parameter in the configuration file `*.meta.json` is invalid.
### Solution
The valid value range for the `percentage_distribute` parameter is described in the detailed log as follows:
```
 Ensure the configuration data follows the format [max_tokens, percentage], where:
    - 'max_tokens' must be a positive number (greater than 0).
    - 'percentage' must be a float between 0 and 1 (greater than 0 and inclusive 1).
    - The sum of all 'percentage' values must equal exactly 1.
    Example valid format: [[1000, 0.5],[500,0.5]] or [[2000, 1.0]]
    Example invalid formats: [[0, 0.5]] (max_tokens <= 0), [[1000, 1.5]] (percentage > 1), [[1000, 0.3], [500,0.2]] (sum not 1)
```

## UTILS-PARAM-007
### Error Description
In the custom dataset scenario, the value of the `method` parameter (which defines the data distribution method) in the configuration file `*.meta.json` is outside the valid range.
### Solution
If the error message is `Type of data distribution(method): uniform1 not supported, legal methods chosen from ['uniform', 'percentage'].`, it means the value `uniform1` of the `method` parameter in `*.meta.json` is outside the valid range. You need to correct the `method` parameter value to either `uniform` or `percentage`.

## UTILS-PARAM-008
### Error Description
In the custom dataset scenario, the configuration file `*.meta.json` contains invalid fields.
### Solution
If the specific error message is `There are illegal keys: xxxxxx,yyyyyy`, it means the `*.meta.json` file contains the two invalid fields `xxxxxx` and `yyyyyy`. You need to delete these two fields from `*.meta.json`.

## UTILS-FILE-002
### Error Description
The tokenizer path specified by the `path` parameter in the model configuration file does not exist.
### Solution
If the content of the model configuration file is as follows:
```python
# In vllm_stream_api_chat.py
models = [
    dict(
        # ......
        path="/path/to/invalid",
        # ......
    ),
]
```
And the specific error log is `Tokenizer path '/path/to/invalid' does not exist`, it indicates that the tokenizer path `/path/to/invalid` specified by the `path` parameter in the model configuration file does not exist (an empty path is also considered non-existent). You need to correct it to an existing tokenizer path.

## UTILS-FILE-003
### Error Description
Failed to load the tokenizer file.
### Solution
If the error message is `Failed to load tokenizer from /path/to/tokenizer: ExceptionName: XXXXXX`, first confirm whether the tokenizer file under the path `/path/to/tokenizer` is compatible with the `transformers` version of the current runtime environment. If compatible, perform further troubleshooting based on the specific error information represented by `XXXXXX`.

## UTILS-FILE-004
### Error Description
No direct solution is available yet.
### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## PARTI-FILE-001
### Error Description
Insufficient permissions for the output path file; the tool cannot write results to it.
### Solution
For example, if the error log is:
```bash
Current user can't modify /path/to/workspace/outputs/default/20250628_151326/predictions/vllm-api-stream-chat/gsm8k.json, reuse will not enable.
```
Execute `ls -l /path/to/workspace/outputs/default/20250628_151326/predictions/vllm-api-stream-chat/gsm8k.json` to check the owner and permissions of this path. If the current user does not have write permission for the file, you need to add write permission for the current user (for example, execute `chmod u+w /path/to/workspace/outputs/default/20250628_151326/predictions/vllm-api-stream-chat/gsm8k.json` to add write permission for the current user).

## CALC-MTRC-001
### Error Description
The performance result data is invalid, and metrics cannot be calculated.
### Solution
#### Scenario 1: The original performance result data is empty
If you specified recalculation of performance results via `--mode perf_viz` when executing the command, and the base output path is `outputs/default/20250628_151326` (find `Current exp folder: ` in the console output), check whether all `*_details.jsonl` files in the `performances/` folder under this path are empty. If they are empty, you need to run the evaluation once first to generate performance result data.

#### Scenario 2: The original performance result data contains no valid values
If you specified recalculation of performance results via `--mode perf_viz` when executing the command, and the base output path is `outputs/default/20250628_151326` (find `Current exp folder: ` in the console output), check whether the `*_details.jsonl` files in the `performances/` folder under this path contain no valid fields (they may have been tampered with). If so, you need to re-run the performance evaluation to generate new data.

## CALC-FILE-001
### Error Description
Failed to save performance result data to disk.
### Solution
If the detailed error log is:
```bash
Failed to write request level performance metrics to csv file '{/path/to/workspace/outputs/default/20250628_151326/performances/vllm-api-stream-chat/gsm8k.csv': XXXXXX
```
Where `XXXXXX` is the specific reason for the disk-saving failure. For example, `Permission denied` means the file already exists and the current user does not have write permission. You can either delete the file or add write permission for the current user to the existing file.

## CALC-DATA-001
### Error Description
No valid performance metric data was obtained for all completed inference requests, and metrics cannot be calculated.
### Solution
If the specific log is:
```bash
All requests failed, cannot calculate performance results. Please check the error logs from responses!
```
This indicates that all requests during the inference process failed. You need to further check the logs of failed requests to identify the cause of the failure.
1. If the command includes `--debug`, the logs of failed requests will be printed directly to the console, and you can view them in the console records.
2. If the command does not include `--debug`, the console records will contain logs similar to `[ERROR] [RUNNER-TASK-001]task failed. OpenICLApiInfervllm-api-stream-chat/synthetic failed with code 1, see outputs/default/20251125_160128/logs/infer/vllm-api-stream-chat/synthetic.out`. You can view the specific cause of the request failure in `outputs/default/20251125_160128/logs/infer/vllm-api-stream-chat/synthetic.out`.

## CALC-DATA-002
### Error Description
When calculating steady-state performance metrics, no requests belonging to the steady state were found among all request information, and steady-state metrics cannot be calculated.
### Solution
You can check the concurrency graph of inference requests (reference document: https://ais-bench-benchmark-rf.readthedocs.io/en/latest/base_tutorials/results_intro/performance_visualization.html) to confirm whether the `Request Concurrency Count` in the concurrency step graph reaches the concurrency number set in the model configuration file (the `batch_size` parameter) **and at least two requests reach the maximum concurrency number**.

If the above conditions are not met, you can try the following methods to achieve a steady state:

#### Scenario A: `Request Concurrency Count` in the concurrency step graph increases continuously and then decreases continuously
1. Reduce the concurrency number of inference requests (the `batch_size` parameter in the model configuration file).
2. Increase the total number of inference requests.

#### Scenario B: `Request Concurrency Count` in the concurrency step graph increases continuously, fluctuates for a period of time, and then decreases continuously
1. Reduce the concurrency number of inference requests (the `batch_size` parameter in the model configuration file).
2. Increase the frequency of sending inference requests (the `request_rate` parameter in the model configuration file).

## SUMM-TYPE-001
### Error Description
The `abbr` parameter configurations of all dataset tasks are mixed (i.e., use different types).
### Solution
For example, if the error log is:
```bash
mixed dataset_abbr type is not supported, dataset_abbr type only support (list, tuple) or str.
```
This indicates that in the `datasets` configuration, the `abbr` parameter configurations of all dataset tasks use different types (e.g., `list` and `str`). You need to unify the `abbr` parameter configurations of all dataset tasks to use the same type (e.g., `list` or `str`).

## SUMM-FILE-001
### Error Description
There are no performance data files (`*_details.jsonl`) in the output working path.
### Solution
1. Confirm whether you incorrectly specified recalculation of performance results via `--mode perf_viz` when executing the evaluation. If you want to run a complete performance test, specify `--mode perf`.
2. Confirm whether the base output path is correct (e.g., `outputs/default/20250628_151326`; find `Current exp folder: ` in the console output).
3. Confirm whether there are `*_details.jsonl` files in the `performances/` folder under this path. If not, check other error information in the previous console logs to confirm whether other errors caused the performance data files to not be generated, and perform further troubleshooting based on the guidance of other error logs.

## SUMM-MTRC-001
### Error Description
The number of valid fields is inconsistent across requests in the detailed performance data.
### Solution
Check whether the number of valid fields is consistent across all requests in the `*_details.jsonl` files under the base output path (e.g., `outputs/default/20250628_151326`; find `Current exp folder: ` in the console output). If inconsistent, check whether there are other errors in the historical console logs that caused the performance data files to not be generated, and perform further troubleshooting based on the guidance of other error logs.

## RUNNER-TASK-001
### Error Description
The evaluation task failed to execute.
### Solution
For example, if the specific error message is `[ERROR] [RUNNER-TASK-001]task failed. OpenICLApiInfervllm-api-stream-chat/synthetic failed with code 1, see outputs/default/20251125_160128/logs/infer/vllm-api-stream-chat/synthetic.out`, please view the specific error information in `outputs/default/20251125_160128/logs/infer/vllm-api-stream-chat/synthetic.out` to identify the cause of the failure.

## TINFER-PARAM-001
### Error Description
The maximum concurrency value `batch_size` in the model configuration file is outside the valid range.
### Solution
If the error log shows `Concurrency must be greater than 0 and <= 100000, but got -1`, it means the maximum concurrency of the model is configured as -1. You need to set the `batch_size` parameter in the model configuration file to an integer greater than 0 and less than or equal to 100000.

Example:
```python
# In vllm_stream_api_chat.py
models = [
    dict(
        attr="service",
        # ......
        batch_size=100,
        # ......
    ),
]
```

## TINFER-PARAM-002
### Error Description
The `num_return_sequences` parameter (number of returned sequences) of the `generation_kwargs` parameter in the model configuration file is outside the valid range.
### Solution
If the error log shows `num_return sequences must be a positive integer, but got {0}`, it means the number of returned sequences of the model is configured as 0. You need to set the `num_return_sequences` parameter in the model configuration file to an integer greater than 0.

Example:
```python
# In vllm_stream_api_chat.py
models = [
    dict(
        attr="service",
        # ......
        generation_kwargs=dict(
            num_return_sequences=1,
        ),
        # ......
    ),
]
```

## TINFER-PARAM-004
### Error Description
The `ramp_up_strategy` parameter (ramp-up strategy) of the `traffic_cfg` parameter in the model configuration file is outside the valid range.
### Solution
If the error log shows `Invalid ramp_up_strategy: {constant} only support 'linear' and 'exponential'`, it means the request sending strategy of the model is configured to a value not in `['exponential', 'linear']`. You need to set the `ramp_up_strategy` parameter in the model configuration file to `'exponential'` or `'linear'`.

Example:
```python
# In vllm_stream_api_chat.py
models = [
    dict(
        attr="service",
        # ......
        traffic_cfg=dict(
            ramp_up_strategy="linear",
        ),
        # ......
    ),
]
```

## TINFER-PARAM-005
### Error Description
Excessively high virtual memory usage when the tool runs inference.
### Solution
If the specific error log is:
```bash
Virtual memory usage too high: 90% > 80% (Total memory: 50 GB "Used: 45 GB, Available: 5 GB, Dataset needed memory size: 3000 MB)
```
It indicates that the current system memory is 50GB, with 45GB used and 5GB available, while the dataset requires 3000MB of memory, thus triggering this error. Solutions are divided into two cases:
1. If the total system memory is insufficient, increase the system memory.
2. If the total system memory is sufficient but the memory required by the dataset is greater than the available memory, clear the occupied memory or cache on the current server.

## TINFER-IMPL-001
### Error Description
When executing a service-oriented inference task, a process fails to start while multiple processes are launched within the inference task.
### Solution
If the error log is:
```bash
Failed to start worker x: XXXXXX, total workers to launch: 4
```
Where `x` is the ID of the failed process, `XXXXXX` is the specific reason for the failure, and `4` is the total number of processes.

Solutions:
1. If the number of occurrences of this error log is equal to the total number of processes, it means all processes failed to start. Check the specific failure reason, take corresponding measures, and retry.
2. If the number of occurrences of this error log is less than the total number of processes, it means some processes failed to start. Partial process startup failures do not affect the execution of the evaluation task but will impact the actual maximum concurrency `batch_size`. Decide whether to manually interrupt to locate the specific failure reason based on actual circumstances.

## TINFER-RUNTIME-001
### Error Description
All requests fail during the warm-up phase when evaluating inference serviceization.
### Solution
If the error log is `Exit task because all warmup requests failed, failed reasons: XXXXXX`, locate the problem based on the specific failure reason `XXXXXX` (**error information from the service**), take corresponding measures, and retry.

## TEVAL-PARAM-001
### Error Description
Invalid values for the number of candidate solutions generated by inference `n` and the number of samples collected from them `k`.
### Solution
If the error log is:
```bash
k and n must be greater than 0 and k <= n, but got k: 16, n: 8
```
It means `k` is greater than `n`. You need to configure `k` to an integer less than or equal to `n`.

Examples:
1. If both `n` and `k` parameters are configured in the dataset configuration file, set their values to the valid range in the configuration file:
```python
# In aime2024_gen_0_shot_str.py, the k parameter corresponds to `k`
aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        # ......
        k=4,
        n=8,
    )
]
```
2. If the `n` parameter is not configured in the dataset configuration file, the value of the `num_return_sequences` parameter in the model configuration file will be used as the value of `n`. You need to configure `k` in the dataset configuration file to an integer less than or equal to `num_return_sequences` in the model configuration file.

```python
# In vllm_stream_api_chat.py, the num_return_sequences parameter corresponds to `n`
models = [
    dict(
        attr="service",
        # ......
        generation_kwargs=dict(
            num_return_sequences=8,
        ),
        # ......
    ),
]

# In aime2024_gen_0_shot_str.py, the k parameter corresponds to `k`
aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        # ......
        k=4,
    )
]
```

## ICLI-PARAM-001
### Error Description
The type parameter of the retriever parameter for constructing prompt engineering in the dataset configuration file is not a subclass of BaseRetriever or a list of subclasses of BaseRetriever.
### Solution
1. If you want to use a custom retriever class `CustomedRetriever`, ensure that `CustomedRetriever` is a subclass of `BaseRetriever`.
2. If you want to use multiple custom retriever classes `CustomedRetriever1, CustomedRetriever2`, configure the `retriever` parameter in the dataset configuration file as `[CustomedRetriever1, CustomedRetriever2]`, and each class in the list must inherit from `BaseRetriever`.

## ICLI-PARAM-002
### Error Description
The value of the infer_mode parameter in the inferencer configuration in the multi-turn dialogue dataset configuration file is outside the valid range.
### Solution
Taking the mtbench configuration file as an example, if the configuration of mtbench_gen.py is as follows:
```python
mtbench_infer_cfg = dict(
    # ......
    inferencer=dict(type=MultiTurnGenInferencer, infer_mode="every1")
)
```
The log error is:
```bash
Multiturn dialogue infer model only supports every、last or every_with_gt, but got every1
```
The correct configuration should set the infer_mode parameter to one of `every`, `last`, or `every_with_gt`.

## ICLI-PARAM-003
### Error Description
When specifying `--mode perf --pressure` in the command line for performance stress testing, the batch_size parameter is not specified in the model configuration file.
### Solution
Taking the `vllm_stream_api_chat.py` configuration file as an example:
```python
# In vllm_stream_api_chat.py
models = [
    dict(
        attr="service",
        # ......
        batch_size=16,
        # ......
    ),
]
```

## ICLI-PARAM-004
### Error Description
The maximum concurrency value `batch_size` in the model configuration file is outside the valid range.
### Solution
If the error log shows `The range of batch_size is [1, 100000], but got -1. Please set it in datasets config`, it means the maximum concurrency of the model is configured as -1. You need to set the `batch_size` parameter in the model configuration file to an integer greater than 0 and less than or equal to 100000.

Example:
```python
# In vllm_stream_api_chat.py
models = [
    dict(
        attr="service",
        # ......
        batch_size=100,
        # ......
    ),
]
```
## ICLI-PARAM-006
### Error Description
PPL-type datasets do not support performance testing.
### Solution
Check the used dataset configuration file, for example:
```python
# In ARC_c_ppl_0_shot_str.py
ARC_c_infer_cfg = dict(
    # ......
    inferencer=dict(type=PPLInferencer))
```
The type of `inferencer` is `PPLInferencer`. Such dataset configuration files do not support performance testing, so you need to replace them with other dataset configuration files or specify `--mode all` to execute accuracy evaluation.

## ICLI-PARAM-007
### Error Description
PPL-type datasets do not support inference using streaming model configurations.
### Solution
Check the used dataset configuration file, for example:
```python
# In ARC_c_ppl_0_shot_str.py
ARC_c_infer_cfg = dict(
    # ......
    inferencer=dict(type=PPLInferencer))
```
The type of `inferencer` is `PPLInferencer`. Such dataset configuration files do not support inference using streaming model configurations, so you need to replace them with other dataset configuration files, or specify a non-streaming model configuration file via `--models`, such as `--models vllm_api_general_chat`.

## ICLI-IMPL-004
### Error Description
BFCL datasets do not support performance testing.
### Solution
1. If you want to use the BFCL dataset task for accuracy testing but mistakenly specify `--mode perf` in the command line (which triggers performance testing), change the command line to `--mode all` to specify accuracy testing.
2. If you want to use the BFCL dataset task for performance testing, it is not supported currently.

## ICLI-IMPL-006
### Error Description
Model tasks with streaming interfaces do not support accuracy evaluation using BFCL datasets.
### Solution
Refer to [Model Configuration Instructions](../base_tutorials/all_params/models.md) and select model tasks with text interfaces (e.g., `vllm_api_general_chat`) for inference.

## ICLI-IMPL-008
### Error Description
The model backend corresponding to the current model configuration file has not implemented the methods required for PPL inference.
### Solution
Refer to the documentation (not yet available) to check which model configurations support PPL inference, such as `vllm_api_general_chat`.

## ICLI-IMPL-010
### Error Description
No token IDs in the result of a PPL inference, leading to failure in loss calculation.
### Solution
Verify whether the tested inference object (inference service) supports PPL inference and can normally return valid `prompt_logprobs` required for PPL inference.

## ICLI-RUNTIME-001
### Error Description
Failed to obtain inference results when accessing the inference service during warm-up.
### Solution
If the log shows `Get result from cache queue failed: XXXXXX` (where `XXXXXX` is the specific reason for the failure to obtain inference results), take corresponding measures based on the specific reason (e.g., if it is a timeout-related exception, confirm whether the timeout setting of the inference service is reasonable or check if the current configuration can access the inference service normally).

## ICLI-FILE-001
### Error Description
Failed to write inference result files to disk.
### Solution
1. If the log shows `Failed to write results to /path/to/outputs/default/20250628_151326/*/*/*.json: XXXXXX`, it means the inference results failed to be written to disk in the accuracy scenario. Troubleshoot and resolve the issue based on the specific saving reason indicated by `XXXXXX` (e.g., permission issues, insufficient disk space, etc.).
2. If the log shows `Failed to write results to /path/to/outputs/default/20250628_151326/*/*/*.jsonl: XXXXXX`, it means the inference results failed to be written to disk in the performance scenario. Troubleshoot and resolve the issue based on the specific saving reason indicated by `XXXXXX` (e.g., permission issues, insufficient disk space, etc.).

## ICLI-FILE-002
### Error Description
Failed to save numpy-format data (e.g., ITL data for each request) to the database.
### Solution
If the log shows `Failed to save numpy array to database: XXXXXX`, it means the numpy-format data failed to be saved to the database. Troubleshoot and resolve the issue based on the specific saving reason indicated by `XXXXXX` (e.g., database connection issues, non-existent database tables, etc.).

## ICLE-DATA-002
### Error Description
The configured number of candidate solutions generated by inference `n` is inconsistent with the actual number of returned candidate solutions.
### Solution
1. If `--mode all` is specified in the command line or `--mode` is not specified (indicating execution of infer + evaluate), triggering this exception means there is a bug in the tool itself. You can provide feedback in the [issue](https://github.com/AISBench/benchmark/issues).
2. If `--mode eval` is specified in the command line (evaluation based on previous inference results), and the exception error is:
`Replication length mismatch, len of replications: 4 != n: 8`, then set the parameter `n` in the configuration file corresponding to the dataset task to the number of replications `4`:
```python
# In aime2024_gen_0_shot_str.py, the k parameter corresponds to `k`
aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        # ......
        n=4,
    )
]
```

## ICLR-TYPE-001
### Error Description
In the dataset configuration file, the type of the prompt template is incorrect. Only `str` or `dict` types are supported currently.
### Solution
Ensure that the type of the prompt template in the inference configuration of the dataset configuration file is `str` or `dict`, for example:
```python
# In aime2024_gen_0_shot_str.py
aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.' # str type
    ),
    # ......
)

# In aime2024_gen_0_shot_chat_prompt.py
aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict( # dict type
            round=[
                dict(
                    role="HUMAN",
                    prompt="{question}\nPlease reason step by step, and put your final answer within \\boxed{}.",
                ),
            ],
        ),
    ),
    # ......
)
```
If the type of the value of the `template` parameter is incorrect, correct it to `str` or `dict` type.

## ICLR-TYPE-002
### Error Description
In the dataset configuration file, when the type of the prompt template is `dict`, the value type of all key-value pairs in it is incorrect. Currently, the supported value types are only `str`, `list`, and `dict`.
### Solution
Ensure that in the dataset configuration file, the value type of all key-value pairs in the prompt template under the inference configuration is `str`, `list`, or `dict`. For example:
```python
# In aime2024_gen_0_shot_chat_prompt.py
aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict( # dict type
            round=[
                dict(
                    role="HUMAN", # str type
                    prompt="{question}\nPlease reason step by step, and put your final answer within \\boxed{}.", # str type
                ),
            ],
        ),
    ),
    # ......
)
```

## ICLR-PARAM-001
### Error Description
In the dataset configuration file, when the `ice_token` parameter is configured in the prompt template, the value of the `template` parameter does not contain the value of the `ice_token` parameter.
### Solution
1. When the type of the `template` parameter is `str`, ensure that the string value of `template` contains the value of the `ice_token` parameter. For example:
```python
# In ceval_gen_5_shot_str.py
ceval_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=f'Below are single-choice questions from the {_ch_name} exam in China. Please select the correct answer.\n</E>{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: {{answer}}', # The string contains '</E>', the value of ice_token
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=GenInferencer),
)

```

2. When the type of the `template` parameter is `dict`, ensure that the value of at least one key-value pair in the dictionary of the `template` value contains the value of the `ice_token` parameter. For example:
```python
# In aime2024_gen_0_shot_chat_prompt.py
cmmlu_infer_cfg = dict(
    # ......
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>', # Same as the value of ice_token
            round=[
                dict(role='HUMAN', prompt=prompt_prefix+QUERY_TEMPLATE),
            ],
        ),
        ice_token='</E>',
    ),
    # ......
)
```

## ICLR-PARAM-002
### Error Description
The `ice_template` parameter is not specified when the dataset configuration file needs to construct few-shots based on the training set.
### Solution
Take `cmmlu_gen_5_shot_cot_chat_prompt.py` as an example. This configuration specifies `retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),` to construct few-shots, so the `ice_template` parameter must be specified. You can modify it with reference to the following content:
```python
cmmlu_infer_cfg = dict(
    ice_template=dict( # ice_template must be configured
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=prompt_prefix+QUERY_TEMPLATE,
            ),
            dict(role='BOT', prompt="{answer}\n",)
        ]),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt=prompt_prefix+QUERY_TEMPLATE),
            ],
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]), # 5-shots specified
    inferencer=dict(type=GenInferencer),
)

```

## ICLR-PARAM-003
### Error Description
In the multimodal dataset configuration file, the key value of the `prompt_mm` parameter in the prompt template is not one of ["text", "image", "video", "audio"].
### Solution
Take `textvqa_gen_base64.py` as an example. In this configuration, the key value of the `prompt_mm` parameter in the prompt template is one of "text", "image", "video", "audio". You can modify it with reference to the following content:
```python
textvqa_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={ # The key value of the prompt_mm parameter is one of "text", "image", "video", "audio"
                    "text": {"type": "text", "text": "{question} Answer the question using a single word or phrase."},
                    "image": {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
                    "video": {"type": "video_url", "video_url": {"url": "data:video/jpeg;base64,{video}"}},
                    "audio": {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,{audio}"}},
                })
            ]
            )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)
```

## ICLR-PARAM-004
### Error Description
The id values in `fix_id_list` for constructing few-shots in the dataset configuration file exceed the range of selectable ids in the training set.
### Solution
If the configuration for constructing few-shots in the dataset configuration file is as follows:
```python
retriever=dict(type=FixKRetriever, fix_id_list=[1,2,5,8]),
```
The detailed error log is `Fix-K retriever index 8 is out of range of [0, 8)`, indicating that the id value 8 in `fix_id_list` exceeds the range [0, 8) of selectable ids in the training set and needs to be corrected to a value within this range.

## ICLR-IMPL-002
### Error Description
The `ice_token` parameter is not configured in the prompt template of the dataset configuration file.
### Solution
1. If both the `prompt_template` parameter and the `ice_template` parameter exist, and the log error is `ice_token of prompt_template is not provided`, then the `ice_token` parameter must exist in the `prompt_template` parameter. For example:
```python
cmmlu_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=prompt_prefix+QUERY_TEMPLATE,
            ),
            dict(role='BOT', prompt="{answer}\n",)
        ]),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt=prompt_prefix+QUERY_TEMPLATE),
            ],
        ),
        ice_token='</E>', # Must be set
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]), # 5-shots specified
    inferencer=dict(type=GenInferencer),
)
```
2. If only the `ice_template` parameter exists, and the log error is `ice_token of ice_template is not provided`, then the `ice_token` parameter must exist in the `ice_template` parameter. For example:
```python
ceval_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=f'Below are single-choice questions from the {_ch_name} exam in China. Please select the correct answer.\n</E>{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: {{answer}}',
        ice_token='</E>', # Must exist
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=GenInferencer),
)
```

## ICLR-IMPL-003
### Error Description
Necessary template fields are missing in the dataset configuration file.
### Solution
If the error log is `Leaving prompt as empty is not supported`, it means that at least one of the `prompt_template` parameter and the `ice_template` parameter must exist in the dataset configuration file.
For example:
```python
cmmlu_infer_cfg = dict( # At least one of ice_template and prompt_template must exist
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt=prompt_prefix+QUERY_TEMPLATE,
            ),
            dict(role='BOT', prompt="{answer}\n",)
        ]),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt=prompt_prefix+QUERY_TEMPLATE),
            ],
        ),
        ice_token='</E>', # Must be set
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]), # 5-shots specified
    inferencer=dict(type=GenInferencer),
)
```

## MODEL-IMPL-001
### Error Description
When implementing a new class based on the `BaseAPIModel` class, the `parse_text_response` method is not implemented, making it impossible to test the inference service through the text interface.
### Solution
(For developers) When implementing a subclass based on the `BaseAPIModel` class, if you want to test the inference service through the text interface, you need to implement the `parse_text_response` method, which is used to parse the text response returned by the model and convert it into the output format of the model inference service.

## MODEL-IMPL-002
### Error Description
When implementing a new class based on the `BaseAPIModel` class, the `parse_stream_response` method is not implemented, making it impossible to test the inference service through the streaming interface.
### Solution
(For developers) When implementing a subclass based on the `BaseAPIModel` class, if you want to test the inference service through the streaming interface, you need to implement the `parse_stream_response` method, which is used to parse the streaming response returned by the model and convert it into the output format of the model inference service.

## MODEL-PARAM-002
### Error Description
In the dataset configuration file, the chat-type prompt template does not contain the `role` or `fallback_role` field.
### Solution
Refer to the following configuration file content:
```python
cmmlu_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN', # Contains the 'role' field
                prompt=prompt_prefix+QUERY_TEMPLATE,
            ),
            dict(role='BOT', prompt="{answer}\n",)
        ]),
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt=prompt_prefix+QUERY_TEMPLATE),
            ],
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]), # 5-shots specified
    inferencer=dict(type=GenInferencer),
)
```

## MODEL-PARAM-003
### Error Description
In the dataset configuration file, the value of the `role` parameter in the chat template of prompt engineering is not within the legal range.
### Solution
If the chat template-related configuration in the dataset configuration file is as follows:
```python
# Take aime2024_gen_0_shot_chat_prompt.py as an example
aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN1",
                    prompt="{question}\nPlease reason step by step, and put your final answer within \\boxed{}.",
                ),
            ],
        ),
    ),
    # ......
)
```
The error log is `Unknown role HUMAN1 in chat template, legal role chosen from ['HUMAN', 'BOT', 'SYSTEM'].`, indicating that the value of the role parameter in the chat template is HUMAN1, while the legal role values are HUMAN, BOT, and SYSTEM. Therefore, the value of the role parameter needs to be corrected to one of HUMAN, BOT, or SYSTEM.

## MODEL-PARAM-004
### Error Description
There is no direct solution available yet.
### Solution
If you need to resolve this issue, [please raise an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## MODEL-PARAM-005
### Error Description
There is no direct solution available yet.
### Solution
If you need to resolve this issue, [please raise an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## MODEL-TYPE-001
### Error Description
In the dataset configuration file, a set of strings is not supported in the prompt engineering template.
### Solution
If the prompt template-related configuration in the dataset configuration file is as follows:
```python
# Take aime2024_gen_0_shot_chat_prompt.py as an example
aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[ # The list contains multiple strings
                "{question}\nPlease reason step by step, and put your final answer within \\boxed{}.",
                "{question}\nPlease reason step by step, and put your final answer within \\boxed{}."
            ],
        ),
    ),
    # ......
)
```
An error will occur: `Mixing str without explicit role is not allowed in API models!`. Please modify `round` to a legal chat template, for example:
```python
round=[
    dict(
        role="HUMAN1",
        prompt="{question}\nPlease reason step by step, and put your final answer within \\boxed{}.",
    ),
],
```

## MODEL-TYPE-002
### Error Description
No direct solution is available at this time.
### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## MODEL-TYPE-003
### Error Description
No direct solution is available at this time.
### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## MODEL-TYPE-004
### Error Description
No direct solution is available at this time.
### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## MODEL-DATA-001
### Error Description
The model task failed to retrieve model name information from the tested inference service.
### Solution
If the error message is `Failed to get service model path from http://url-to-infer-service. Error: XXXXXX`, it indicates a failure to access `http://url-to-infer-service/v1/models`. You need to check if the tested inference service is running properly and if the `/v1/models` sub-service is enabled. You can also locate the cause of the access failure to `http://url-to-infer-service/v1/models` based on the specific error `XXXXXX`. If the URL `http://url-to-infer-service/` does not support the `v1/models` sub-service, you can configure the model name in the `model` parameter of the model configuration file. For example:
```python
# In vllm_api_stream_chat.py
models = [
    dict(
        # ......
        model="name_of_model",
        # ......
    )
]
```

## MODEL-DATA-002
### Error Description
The dataset configuration file lacks required parameters.
### Solution
If the error message is `Invalid prompt content: without 'prompt' or 'prompt_mm' param!`, it means the dataset configuration file does not contain either the `prompt` or `prompt_mm` parameter. You need to add one of these two parameters to the dataset configuration file. For example:
```python
# Take aime2024_gen_0_shot_chat_prompt.py as an example
aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict( # Must contain either the 'prompt' or 'prompt_mm' field
                    role="HUMAN",
                    prompt="{question}\nPlease reason step by step, and put your final answer within \\boxed{}.",
                ),
            ],
        ),
    ),
    # ......
)
```

## MODEL-DATA-003
### Error Description
Failed to parse the returned result of the request in JSON format.
### Solution
If the error message is `Unexpected response format. Please check 'error_info' in {dataset_abbr}_failed.jsonl for more information.`, you need to check the specific error information (content of the `error_info` field) in the `{dataset_abbr}_failed.jsonl` file under the current inference task's output path (e.g., `outputs/default/20250628_151326/performances/vllm-api-stream-chat/`) and further explore solutions.

## MODEL-CFG-001
### Error Description
The `max_seq_len` parameter is not configured in the local model configuration file.
### Solution
If the error message is `max_seq_len is not provided and cannot be inferred from the model config.`, it means you need to add the `max_seq_len` parameter to the local model configuration file. For example:
```python
# In hf_chat_model.py
models = [
    dict(
        attr="local",
        # ......
        max_seq_len=2048,
        # ......
    )
]
```

## MODEL-MOD-001
### Error Description
Special dependencies required for model execution are not installed.
### Solution
If the error message is `fastchat module not found. Please install with\npip install "fschat[model_worker,webui]"`, it indicates that the `fastchat` dependency is missing. You can install it by executing `pip install "fschat[model_worker,webui]"`.

## DSET-CFG-001
### Error Description
The dataset configuration file lacks the `path` field to specify the dataset path.
### Solution
If the error message is `The 'path' argument is required to load the dataset.`, it means the dataset configuration file does not contain the `path` field. You need to add the `path` field to the dataset configuration file. For example:
```python
# In aime2024_gen_0_shot_chat_prompt.py
aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        path='ais_bench/datasets/aime/aime.jsonl', # Required field to configure the dataset path
        # ......
    )
]
```

## DSET-FILE-001
### Error Description
The dataset file does not exist.
### Solution
1. If the error message is `Path is not a directory or Parquet file: /path/to/dataset.jsonl`, it means `/path/to/dataset.jsonl` is not a dataset in the required `.parquet` format. Please confirm that the dataset format meets expectations.
2. If the error message is `No Parquet file found in /path/to/dataset/.`, it means no `.parquet` format dataset is found in the path `/path/to/dataset/`. Please confirm that the dataset format meets expectations.
3. If the error message is `"Dataset file not found: /path/to/dataset/`, it means the dataset path `/path/to/dataset/` itself does not exist. Please confirm that the dataset path matches the expected input path.

## DSET-DATA-002
### Error Description
The content structure of the dataset is invalid.
### Solution
Please check for format issues in the dataset content based on the detailed error message.

## DSET-DATA-005
### Error Description
No direct solution is available at this time.
### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## DSET-DATA-006
### Error Description
No direct solution is available at this time.
### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## DSET-PARAM-002
### Error Description
Invalid values for `n` (number of candidate solutions generated by inference) and `k` (number of samples collected from candidates).
### Solution
If the error log shows:
```bash
Maximum value of `k` 4 must be less than or equal to `n` 8
```
It means `k` is greater than `n`. You need to configure `k` as an integer less than or equal to `n`.
For example:
1. If both `n` and `k` parameters are configured in the dataset configuration file, set their values within the valid range in the configuration file:
```python
# In aime2024_gen_0_shot_str.py, the k parameter corresponds to `k`
aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        # ......
        k=4,
        n=8,
    )
]
```

## DSET-PARAM-004
### Error Description
Invalid parameters in the dataset configuration file.
### Solution
Please check for invalid parameter value issues in the dataset configuration file based on the detailed error message.

## DSET-DEPENDENCY-002
### Error Description
Missing dependencies required for the dataset task evaluation.
### Solution
If the error message is:
```bash
Please install human_eval use following steps:
git clone git@github.com:open-compass/human-eval.git
cd human-eval && pip install -e .
```
Execute `git clone git@github.com:open-compass/human-eval.git` and `cd human-eval && pip install -e .` in sequence according to the error log content to install the `human-eval` library.

## DSET-MTRC-001
### Error Description
No direct solution is available at this time.
### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.

## DSET-MTRC-003
### Error Description
No direct solution is available at this time.
### Solution
If you need to resolve this issue, [please submit an issue](https://github.com/AISBench/benchmark/issues) and include this error code in the issue description.