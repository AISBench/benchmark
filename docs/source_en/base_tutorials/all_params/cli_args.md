# User Configuration Parameters
AISBench Benchmark supports customizing the inference mode and evaluation process through two methods: [**Command Line Interface (CLI) Parameters**](#command-line-parameters) and [**Configuration Constant File**](#configuration-constant-file-parameters).


## Command Line Parameters

The basic calling format for command line parameters `[OPTIONS]` is as follows:
```bash
ais_bench [OPTIONS]
```

### Parameter Description
Based on the execution scenario, command line parameters are divided into three categories:
- Common Parameters
- Accuracy Evaluation Parameters (effective only when `--mode` is set to `all`, `infer`, `eval`, or `viz`)
- Performance Evaluation Parameters (effective only when `--mode` is set to `perf` or `perf_viz`)

`Accuracy Evaluation Parameters` take effect only when the `--mode` parameter is specified as `"all", "infer", "eval", "viz"`. `Performance Evaluation Parameters` take effect only when the `--mode` parameter is specified as `"perf", "perf_viz"`. `Common Parameters` are not restricted by the task execution mode and can be specified in all modes.

# ### Common Parameters
Applicable to all modes and can be used in combination with accuracy or performance parameters.

| Parameter               | Description                                                                                                                                                                                                 | Example                          |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `--models` | Specifies the name of the model inference backend task (corresponding to a pre-implemented default model configuration file under the path `ais_bench/benchmark/configs/models`). Multiple task names are supported. For details, refer to 📚 [Supported Models](./models.md) | `--models vllm_api_general`  |
| `--datasets` | Specifies the name of the dataset task (corresponding to a pre-implemented default dataset configuration file under the path `ais_bench/benchmark/configs/datasets`). Multiple dataset names are supported. For details, refer to 📚 [Supported Dataset Types](./datasets.md) | `--datasets gsm8k_gen`    |
| `--summarizer` | Specifies the name of the result summary task (corresponding to a pre-implemented default configuration file under the path `ais_bench/benchmark/configs/summarizers`). For details, refer to 📚 [Supported Result Summary Tasks](./summarizer.md) | `--summarizer medium`|
| `--mode` or `-m` | Running mode, optional values: `all`, `infer`, `eval`, `viz`, `perf`, `perf_viz`; default value is `all`.<br>For details, refer to 📚 [Running Mode Description](./mode.md). | `--mode infer`<br>`-m all`|
| `--reuse` or `-r`       | Specifies the timestamp in an existing working directory to continue execution and overwrite original results. Used in conjunction with the `--mode` parameter, it can resume interrupted inference, or perform accuracy calculation/visualization result printing based on existing inference results. If no parameter is added, the latest timestamp in the `--work-dir` is automatically selected. | `--reuse 20250126_144254`<br>`-r 20250126_144254` |
| `--work-dir` or `-w`    | Specifies the evaluation working directory for saving output results. Default path: `outputs/default`.                                                                                                       | `--work-dir /path/to/work`<br>`-w /path/to/work` |
| `--config-dir`          | Path to the folder where configuration files for `models`, `datasets`, and `summarizers` are stored. Default path: `ais_bench/benchmark/configs`.                                                          | `--config-dir /xxx/xxx`          |
| `--debug`               | Enables Debug mode. The mode is enabled if this parameter is configured, and disabled if not; disabled by default. In Debug mode, all logs are printed directly to the terminal. (In Debug mode, the `--max-num-workers` parameter is forced to 1, tasks are executed serially, and only single-core execution is used, which limits concurrency capabilities.)                              | `--debug`                        |
| `--dry-run`             | Enables Dry Run mode (prints logs to the screen without actually running tasks). The mode is enabled if this parameter is configured, and disabled if not; disabled by default.                              | `--dry-run`                      |
| `--max-workers-per-gpu` | Reserved parameter; not currently supported.                                                                                                                                                               | `--max-workers-per-gpu 1`        |
| `--merge-ds`            | Enables merged inference for datasets of the same type (runs multiple datasets for the same task together).                                                                                                 | `--merge-ds`                     |
| `--num-prompts`         | Specifies the number of test cases for the dataset (selected in dataset order). A positive integer must be passed. If the number exceeds the total number of cases in the dataset or no value is specified, the entire dataset is used for testing. | `--num-prompts 500`              |
| `--max-num-workers`     | Number of parallel tasks, range: `[1, number of CPU cores]`; default value: `1`. Invalid when `--debug` is specified; all tasks are executed serially.                                                                          | `--max-num-workers 2`            |
| `--num-warmups`         | Number of warm-up runs before sending requests. Data is selected in dataset order for testing. When `num-warmups` exceeds the number of dataset entries, data from the dataset will be sent in a loop. Default value: `1`; set to `0` to disable warm-up. If all requests fail during the warmup phase, subsequent inference tasks will not be executed.                                                                                                          | `--num-warmups 10`               |


# ### Accuracy Evaluation Parameters
Valid only when the mode is `all`, `infer`, `eval`, or `viz`.

| Parameter               | Description                                                                 | Example              |
| ----------------------- | --------------------------------------------------------------------------- | -------------------- |
| `--dump-eval-details`   | Toggle to dump details of the evaluation process. Enabled if configured, disabled if not; disabled by default. | `--dump-eval-details`|
| `--dump-extract-rate`   | Toggle to dump evaluation speed data. Enabled if configured, disabled if not; disabled by default.             | `--dump-extract-rate`|


# ### Performance Evaluation Parameters
Valid only when the mode is `perf` or `perf_viz`.

| Parameter               | Description                                                                                                                                                                                                 | Example              |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| `--pressure` | Switch to enable performance pressure testing mode. Effective only when `--mode perf` is set. Enabled if this parameter is configured, disabled if not; disabled by default. For details on pressure testing, refer to 📚 [Enabling Steady-State Testing with Stress Testing](../../advanced_tutorials/stable_stage.md#enabling-steady-state-testing-with-stress-testing). | `--pressure`|
| `--pressure-time`       | Duration of pressure testing. Only takes effect when `--pressure` mode is specified. Unit: seconds; default value: 15 seconds; value range: `[1, 86400]` (i.e., 1 second to 24 hours).                     | `--pressure-time 30` |


## Configuration Constant File Parameters
Some global constants are not restricted by task type, and it is recommended to keep their default values. If customization is required, edit the constant file: [`global_consts.py`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/global_consts.py) for configuration.

The currently supported parameter configurations are as follows:

| Parameter Name | Description | Value Range / Requirements |
| ----------- | ----------- | ----------- |
| `WORKERS_NUM` | Number of processes used for sending requests. The default value is 0, which means automatic allocation based on the maximum number of concurrent requests configured by the user. (Invalid when the command-line parameter `--debug` is specified; single-core execution is used for sending requests, which limits concurrency capabilities.) | [0, number of CPU cores] |
| `MAX_CHUNK_SIZE` | Maximum cache size for a single chunk returned by the streaming inference model backend. The default value is 65535 bytes (64KB). | `(0, 16777216]` (Unit: Byte) |
| `REQUEST_TIME_OUT` | Timeout period for the client to wait for a response after sending a request. The default value is None, meaning infinite waiting (always waiting for the model to return results). | `None` or `>0` (Unit: seconds) |
| `LOG_LEVEL` | Log level, optional values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Default value: `INFO`. | `[DEBUG, INFO, WARNING, ERROR, CRITICAL]` |