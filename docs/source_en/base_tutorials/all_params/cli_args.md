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

| Parameter               | Description                                                                                                                                                                                                 | Example                          |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `config` | Specifies the path to a custom configuration file. | `ais_bench /path/to/custom_config.py {other optional arguments}` |
| `--models` | Specifies the name of the model inference backend task (corresponding to a pre-implemented default model configuration file under the path `ais_bench/benchmark/configs/models`). Multiple task names are supported. For details, refer to 📚 [Supported Models](./models.md).<br> ⚠️ **Note**: This parameter is invalid when a custom configuration file path is specified. | `--models vllm_api_general`  |
| `--datasets` | Specifies the name of the dataset task (corresponding to a pre-implemented default dataset configuration file under the path `ais_bench/benchmark/configs/datasets`). Multiple dataset names are supported. For details, refer to 📚 [Supported Dataset Types](../../get_started/datasets.md).<br> ⚠️ **Note**: This parameter is invalid when a custom configuration file path is specified. | `--datasets gsm8k_gen`    |
| `--summarizer` | Specifies the name of the result summary task (corresponding to a pre-implemented default configuration file under the path `ais_bench/benchmark/configs/summarizers`). For details, refer to 📚 [Supported Result Summary Tasks](./summarizer.md).<br> ⚠️ **Note**: This parameter is invalid when a custom configuration file path is specified. | `--summarizer medium`|
| `--mode` or `-m` | Running mode, optional values: `all`, `infer`, `eval`, `viz`, `perf`, `perf_viz`; default value is `all`.<br>For details, refer to 📚 [Running Mode Description](./mode.md). | `--mode infer`<br>`-m all`|
| `--reuse` or `-r`       | Specifies the timestamp in an existing working directory to continue execution and overwrite original results. Used in conjunction with the `--mode` parameter, it can resume interrupted inference, or perform accuracy calculation/visualization result printing based on existing inference results. If no parameter is added, the latest timestamp in the `--work-dir` is automatically selected. | `--reuse 20250126_144254`<br>`-r 20250126_144254` |
| `--work-dir` or `-w`    | Specifies the evaluation working directory for saving output results. Default path: `outputs/default`.                                                                                                       | `--work-dir /path/to/work`<br>`-w /path/to/work` |
| `--config-dir`          | Path to the folder where configuration files for `models`, `datasets`, and `summarizers` are stored. Default path: `ais_bench/benchmark/configs`.                                                          | `--config-dir /xxx/xxx`          |
| `--debug`               | Enables Debug mode. The mode is enabled if this parameter is configured, and disabled if not; disabled by default. In Debug mode, all logs are printed directly to the terminal. (In Debug mode, the `--max-num-workers` parameter is forced to 1, tasks are executed serially, and only single-core execution is used, which limits concurrency capabilities.)                              | `--debug`                        |
| `--dry-run`             | Enables Dry Run mode (prints logs to the screen without actually running tasks). The mode is enabled if this parameter is configured, and disabled if not; disabled by default.                              | `--dry-run`                      |
| `--max-workers-per-gpu` | Reserved parameter; not currently supported.                                                                                                                                                               | `--max-workers-per-gpu 1`        |
| `--merge-ds`            | Enables merged inference for datasets of the same type (runs multiple datasets for the same task together).                                                                                                 | `--merge-ds`                     |
| `--num-prompts`         | Specifies the number of test cases for the dataset (selected in dataset order). A positive integer must be passed. If the number exceeds the total number of cases in the dataset or no value is specified, the entire dataset is used for testing. | `--num-prompts 500`              |
| `--max-num-workers`     | Number of parallel tasks, range: `[1, number of CPU cores]`; default value: `1`. Invalid when `--debug` is specified; all tasks are executed serially. Note: In performance evaluation scenarios, an excessively high concurrency may cause resource contention among different processes, leading to inaccurate test results. | `--max-num-workers 2`            |
| `--num-warmups`         | Number of warm-up runs before sending requests. Data is selected in dataset order for testing. When `num-warmups` exceeds the number of dataset entries, data from the dataset will be sent in a loop. Default value: `1`; set to `0` to disable warm-up. If all requests fail during the warmup phase, subsequent inference tasks will not be executed.                                                                                                          | `--num-warmups 10`               |
| `--response-anomaly` / `--no-response-anomaly` | Enables or disables msProbe response anomaly detection. The command-line value overrides `response_anomaly.enabled` in the config file. Detection is serially bound to the inference stage: after inference finishes, the workflow starts detection and waits for it to complete (the dedicated status board prints the final result) before entering the subsequent Judge / Eval / Summary stages; requires the service to return token ids and top-k logprobs. Only supported in `all`, `infer`, and `infer_judge` generation chains; performance mode and Agent evaluation modes are unsupported. | `--response-anomaly` |
| `--response-anomaly-payload-retention` | Payload retention mode after anomaly detection: `all` keeps everything, `anomalies` keeps anomalous and detection-failed/unavailable Cases, `none` keeps nothing. The command-line value overrides the config file; defaults to `anomalies`. | `--response-anomaly-payload-retention anomalies` |


### Accuracy Evaluation Parameters
Valid only when the mode is `all`, `infer`, `eval`, or `viz`.

| Parameter               | Description                                                                 | Example              |
| ----------------------- | --------------------------------------------------------------------------- | -------------------- |
| `--dump-eval-details`   | Toggle to dump details of the evaluation process. Enabled if configured, disabled if not; disabled by default. | `--dump-eval-details`|
| `--dump-extract-rate`   | Toggle to dump evaluation speed data. Enabled if configured, disabled if not; disabled by default.             | `--dump-extract-rate`|


### Performance Evaluation Parameters
Valid only when the mode is `perf` or `perf_viz`.

| Parameter               | Description                                                                                                                                                                                                 | Example              |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| `--pressure` | Switch to enable performance pressure testing mode. Effective only when `--mode perf` is set. Enabled if this parameter is configured, disabled if not; disabled by default. For details on pressure testing, refer to 📚 [Enabling Steady-State Testing with Stress Testing](../../advanced_tutorials/stable_stage.md#enabling-steady-state-testing-with-stress-testing). | `--pressure`|
| `--pressure-time`       | Duration of pressure testing. Only takes effect when `--pressure` mode is specified. Unit: seconds; default value: 15 seconds; value range: `[1, 86400]` (i.e., 1 second to 24 hours).                     | `--pressure-time 30` |
| `--spec-decode`         | Enable speculative decoding metrics collection from the inference server's Prometheus `/metrics` endpoint. Only effective in `--mode perf`. For detailed usage, see 📚 [Speculative Decoding Metrics Collection](../../advanced_tutorials/spec_decode.md). | `--spec-decode` |


## Configuration Constant File Parameters
Some global constants are not restricted by task type, and it is recommended to keep their default values. If customization is required, edit the constant file: [`global_consts.py`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/global_consts.py) for configuration.

The currently supported parameter configurations are as follows:

| Parameter Name | Description | Value Range / Requirements |
| ----------- | ----------- | ----------- |
| `WORKERS_NUM` | Number of processes used for sending requests. The default value is 0, which means automatic allocation based on the maximum number of concurrent requests configured by the user. (Invalid when the command-line parameter `--debug` is specified; single-core execution is used for sending requests, which limits concurrency capabilities.) | [0, number of CPU cores] |
| `MAX_CHUNK_SIZE` | Maximum cache size for a single chunk returned by the streaming inference model backend. The default value is 65535 bytes (64KB). | `(0, 16777216]` (Unit: Byte) |
| `REQUEST_TIME_OUT` | Timeout period for the client to wait for a response after sending a request. The default value is None, meaning infinite waiting (always waiting for the model to return results). | `None` or `>0` (Unit: seconds) |
| `LOG_LEVEL` | Log level, optional values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Default value: `INFO`. | `[DEBUG, INFO, WARNING, ERROR, CRITICAL]` |

## Response Anomaly Detection Configuration

Response anomaly detection currently supports only the vLLM Chat API model configurations `vllm_api_general_chat`, `vllm_api_stream_chat`, and `vllm_api_stream_chat_multiturn`. Other model backends are not supported yet.

Add a `response_anomaly` entry to the top-level config file to enable detection; it can also be overridden with `--response-anomaly`:

```python
response_anomaly = dict(
    enabled=True,
)
```

Model-specific msProbe configuration goes into the model config:

```python
models = [
    dict(
        abbr='qwen3-30b',
        attr='service',
        response_anomaly=dict(
            model_name="",   # Model name, for example Qwen3-30B-A3B
            model_path="",   # Local model directory, for example /home/Qwen3-30B-A3B; optional, used to auto-generate configs
            msprobe_config_path='',  # Optional; msProbe algorithm-threshold config.yaml path for manual threshold tuning
            msprobe_mtype_path='/path/to/mtype_config.json',
            msprobe_token2category_dir='/path/to/token2category/',
        ),
    ),
]
```

When `msprobe_mtype_path` / `msprobe_token2category_dir` are not provided, the default files inside the msProbe package are used; when `model_path` is configured, configs are auto-generated into `<work_dir>/response_anomaly_config/<model abbr>/` (auto-generation never overwrites an existing `config.yaml`, so manually tuned thresholds are preserved). They can also be generated manually:

```bash
ais_bench-gen-response-anomaly-config \
  --model-path /home/Qwen3-30B-A3B \
  --model-name Qwen3-30B-A3B \
  --output-dir ./msprobe_configs
```

When `model_name` is not configured explicitly, it falls back to the model `abbr` with a warning (msProbe model matching may be degraded; explicit configuration is recommended).

When enabled, AISBench adds `logprobs=True` and a fixed `top_logprobs=20` to the service inference requests; the value is constrained by the detection algorithm and cannot be configured externally. For vLLM backends, `return_token_ids=True` and `return_tokens_as_token_ids=True` are also appended to obtain token ids; if the server version is too old to support these parameters, requests may fail — upgrade vLLM in that case. During inference, the full payload is written directly to `response_anomaly/<model>/payload_staging/<dataset>/*.jsonl.zst`, and predictions only keep lightweight results from the start. After inference finishes, the detection thread streams and decompresses the staging data and calls msProbe; detection results are written to `response_anomaly/<model>/<dataset>.jsonl`. Each Case contains `id`, `uuid`, `is_anomaly`, `anomaly_type` (0: normal, 1: rare character, 2: garbled, 3: repetition, 4: NaN value), `anomaly_type_name` (the type-name string such as `normal`/`garbled`/`repetition`, which is more convenient for statistics), and `detection_status`. After detection, the staging data is retained or cleaned according to `payload_retention`. The status panel shows the config preparation, detector loading, streaming detection, and archive finalization stages.

The `detection_status` values in the detection results are:

| Status | Meaning | Troubleshooting |
| --- | --- | --- |
| `completed` | msProbe was invoked and returned a detection result | Nothing to do |
| `skipped` | The inference response did not carry token ids or top-k logprobs | Check whether the service supports and returns `logprobs` / `top_logprobs` / token id fields |
| `unavailable` | `mindstudio-probe` (the response_anomaly extra) is not installed | Install the optional dependency per the [Installation Guide](../../get_started/install.md) and re-run |
| `failed` | An exception occurred during invocation or input conversion | Check the `reason` field of the Case result (error type and summary) and the detection logs |

**Anomaly detection results do not affect the original evaluation metrics**: anomalous Cases are not rewritten as inference failures; accuracy/performance metrics are computed as usual, and anomaly information is an independent audit result. Detection-specific logs are located at `<work_dir>/logs/response_anomaly/<model>/<dataset>.out`; detection progress and per-type statistics can also be found in the `<work_dir>/status_tmp/tmp_ResponseAnomaly.json` status file.

```python
response_anomaly = dict(
    enabled=True,
    payload_retention='anomalies',  # all | anomalies | none
    payload_storage=dict(
        format='jsonl',
        compression='zstd',
        compression_level=3,
        rows_per_shard=2000,
    ),
)
```

`all` keeps every payload and atomically promotes the staging data to the official archive after detection without re-compressing; `anomalies` keeps only detected anomalies plus detection-failed/unavailable Cases; `none` keeps no payload. All three modes keep the standalone detection results. `--reuse` must keep the retention policy of the original work directory. Results are written to disk in batches, and the status is refreshed at most once per second; msProbe token-category maps are cached per model and EOS token to avoid re-parsing large JSON files for every Case.

Files produced by detection are laid out under `<work_dir>` as follows:

```text
<work_dir>/
├── predictions/<model abbr>/<dataset abbr>.jsonl        # Inference results (lightweight, no token/logprob payload)
├── response_anomaly/
│   └── <model abbr>/
│       ├── <dataset abbr>.jsonl                          # Detection results, one Case per line
│       ├── payload_staging/<dataset abbr>/               # Transient staging during inference; cleaned after detection
│       │   └── part-*.jsonl.zst
│       └── payload/<dataset abbr>/                       # Payload archive; absent when payload_retention is none
│           ├── payload_manifest.json                     # Archive manifest (per-shard rows, sizes, sha256)
│           └── part-*.jsonl.zst                          # Compressed payload shards (at most rows_per_shard Cases each)
├── response_anomaly_config/                              # Present only when auto-generated via model_path
│   └── <model abbr>/
│       ├── configs/
│       │   ├── config.yaml                               # Detection algorithm thresholds (never overwritten once present)
│       │   └── mtype_config.json                         # Model name to BOS/EOS token id mapping
│       └── token2category/
│           └── <model name>_<vocab size>.json            # Token id to character-category mapping
└── logs/
    └── response_anomaly/<model abbr>/<dataset abbr>.out  # Detection-specific log
```

Path-by-path notes:

- **Detection results** `response_anomaly/<model abbr>/<dataset abbr>.jsonl`: one Case per line; field semantics are described in the `detection_status` table and the Case field notes above.
- **Payload archive** `response_anomaly/<model abbr>/payload/<dataset abbr>/`: `all` keeps every Case, `anomalies` keeps only anomalous plus detection-failed/unavailable Cases, and `none` keeps nothing (the directory does not exist). To read the data, decompress the `part-*.jsonl.zst` shards with zstandard and parse each line as JSON; `payload_manifest.json` records per-shard row counts, sizes, and sha256 checksums for integrity verification. Note: under `anomalies`, even when there is nothing to retain, an archive directory containing only an empty manifest is still published to indicate the archiving flow completed successfully — it is not a leftover file.
- **Transient files**: `payload_staging/` receives payload records during inference and is cleaned automatically after detection; `.<dataset>.payload-build-*` build directories left by an interrupted detection are cleaned automatically when the next detection starts; `status_tmp/tmp_ResponseAnomaly.json` is a runtime status file (detection progress and per-type statistics) removed together with the status directory when the workflow ends.
- **Auto-generated msProbe configs** `response_anomaly_config/<model abbr>/`: generated only when `model_path` is configured and explicit mtype/token2category paths are absent. An existing `config.yaml` is never overwritten (manually tuned thresholds are preserved), and `mtype_config.json` supports multi-model merging across repeated generations.
- **Detection log** `logs/response_anomaly/<model abbr>/<dataset abbr>.out`: records the detection run for the model/dataset group, including detector initialization failures and per-Case failure reasons.

Detection runs through msProbe's `ILLDetector(config_path, mtype_path, tk2cat_path).run(...)`; all three file paths can be configured in AISBench. Install the optional dependencies first: `pip install 'ais-bench-benchmark[response_anomaly]'`. During installation, pip downloads and builds the pinned msProbe source from GitCode, so the environment needs Git and network access. The service response must contain `token_ids` (or `tokens`) and `topk_logprobs`; Cases missing these fields are recorded with a `skipped` status. `model_name` must be consistent with msProbe's `mtype_config.json` and the token-category mapping. When using `--reuse`, existing detection results are inherited by matching both the Case `id` and `uuid` (a changed `uuid` means the Case was re-inferred, so it never gets a stale result); Cases with `completed` status are not re-detected, while Cases with `skipped` / `failed` / `unavailable` status are re-detected on resume.
