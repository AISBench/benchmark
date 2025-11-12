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


### Common Parameters
Applicable to all modes and can be used in combination with accuracy or performance parameters.

| Parameter | Description | Example |
| ---- | ---- | ---- |
| `--models` | Specifies the name of the model inference backend task (corresponding to a pre-implemented default model configuration file under the path `ais_bench/benchmark/configs/models`). Multiple task names are supported; this parameter is mutually exclusive with the `config` parameter. For details, refer to 📚 [Supported Models](./models.md) | `--models vllm_api_general`  |
| `--datasets` | Specifies the name of the dataset task (corresponding to a pre-implemented default dataset configuration file under the path `ais_bench/benchmark/configs/datasets`). Multiple dataset names are supported; this parameter is mutually exclusive with the `config` parameter. For details, refer to 📚 [Supported Dataset Types](./datasets.md) | `--datasets gsm8k_gen`    |
| `--summarizer` | Specifies the name of the result summary task (corresponding to a pre-implemented default configuration file under the path `ais_bench/benchmark/configs/summarizers`). For details, refer to 📚 [Supported Result Summary Tasks](./summarizer.md) | `--summarizer medium`|
| `--mode` or `-m` | Running mode, optional values: `all`, `infer`, `eval`, `viz`, `perf`, `perf_viz`; default value is `all`.<br>For details, refer to 📚 [Running Mode Description](./mode.md). | `--mode infer`<br>`-m all`|
| `--reuse` or `-r` | Specifies the timestamp in an existing working directory to continue execution and overwrite original results. Combined with the value of the `--mode` parameter, it can be used to resume interrupted inference, or perform accuracy calculation and visualization result printing based on existing inference results. If no parameter is added, the latest timestamp under `--work-dir` is automatically selected. | `--reuse 20250126_144254`<br>`-r 20250126_144254` |
| `--work-dir` or `-w` | Specifies the evaluation working directory for saving output results. The default path is `outputs/default`. | `--work-dir /path/to/work`<br>`-w /path/to/work` |
| `--config-dir` | The folder path where the configuration files for `models`, `datasets`, and `summarizers` are stored. The default path is `ais_bench/benchmark/configs`. | `--config-dir /xxx/xxx`   |
| `--debug` | Enables Debug mode. This parameter is enabled if configured, and disabled if not configured; it is disabled by default. In Debug mode, all logs are printed directly to the terminal. | `--debug`   |
| `--dry-run` | Enables Dry Run mode (only prints logs to the screen without actually running the task). This parameter is enabled if configured, and disabled if not configured; it is disabled by default. | `--dry-run` |
| `--max-workers-per-gpu` | Reserved parameter, not supported temporarily. | `--max-workers-per-gpu 1` |
| `--merge-ds` | Enables merged inference for datasets of the same type (running multiple datasets for the same task together). | `--merge-ds`|


### Accuracy Evaluation Parameters
Effective only when the mode is `all`, `infer`, `eval`, or `viz`.

| Parameter | Description | Example |
| ---- | ---- | ---- |
| `--max-num-workers` | Number of parallel tasks, range `[1, number of CPU cores]`, default value is `1`. Invalid in Continuous Batch or performance mode. | `--max-num-workers 2` |
| `--dump-eval-details` | Switch to enable dumping details of the evaluation process. Enabled if this parameter is configured, disabled if not; disabled by default. | `--dump-eval-details` |
| `--dump-extract-rate` | Switch to enable dumping evaluation speed data. Enabled if this parameter is configured, disabled if not; disabled by default. | `--dump-extract-rate` |
| `--disable-cb` | Disables Continuous Batch inference (effective only for service-oriented API-type models). Disabled if this parameter is configured, enabled if not; enabled by default. When CB is enabled, multiple processes run concurrently, with a maximum concurrency limit of 500 per process.<br>After disabling, single-process mode is restored, and `--max-num-workers` takes effect. | `--disable-cb`  |


### Performance Evaluation Parameters
Effective only when the mode is `perf` or `perf_viz`.

| Parameter | Description | Example |
| ---- | ---- | ---- |
| `--num-prompts` | Specifies the number of data samples for dataset evaluation. A positive integer must be entered. If the value exceeds the total number of dataset samples or no value is specified, the entire dataset is used for evaluation. | `--num-prompts 500` |
| `--pressure` | Switch to enable performance pressure testing mode. Effective only when `--mode perf` is set. Enabled if this parameter is configured, disabled if not; disabled by default. For details on pressure testing, refer to 📚 [Enabling Steady-State Testing with Stress Testing](../../advanced_tutorials/stable_stage.md#enabling-steady-state-testing-with-stress-testing). | `--pressure`|


## Configuration Constant File Parameters
Some global constants are not restricted by task type, and it is recommended to keep their default values. If customization is required, edit the constant file: [`global_consts.py`](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/global_consts.py) for configuration.

The currently supported parameter configurations are as follows:

| Parameter Name | Description | Value Range / Requirements |
| ----------- | ----------- | ----------- |
| `WORKERS_NUM` | Number of processes used for sending requests. The default value is 0, which means automatic allocation based on the maximum number of concurrent requests configured by the user. | [0, number of CPU cores] |
| `CUSTOM_PACKAGE_DIR` | Specifies the directory path of custom Python packages. The Benchmark tool will load user-defined packages from this directory. | Must be a local path accessible to the user, pointing to the folder containing custom packages |
| `PRESSURE_TIME` | Duration of pressure testing, effective only when `--pressure` mode is specified. Unit: seconds. | `[1, 86400]` (i.e., 1 second to 24 hours) |
| `CONNECTION_ADD_RATE` | Concurrent thread creation rate. Represents the number of new concurrent threads added per second until the maximum concurrency limit is reached. Effective only when `--pressure` mode is specified. | `> 0.1` (Unit: threads per second) |
| `MAX_CHUNK_SIZE` | Maximum cache size for a single chunk returned by the streaming inference model backend. The default value is 65535 bytes (64KB). | `(0, 16777216]` (Unit: Byte) |
| `REQUEST_TIME_OUT` | Timeout period for the client to wait for a response after sending a request. The default value is None, meaning infinite waiting (always waiting for the model to return results). | `None` or `>0` (Unit: seconds) |