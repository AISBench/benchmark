# 用户配置参数
AISBench Benchmark 支持通过 [**命令行参数（CLI）**](#命令行参数
) 和 [**配置常量文件**](#配置常量文件参数) 两种方式，自定义推理模式和评测流程。

## 命令行参数

命令行参数 `[OPTIONS]` 的基本调用格式：
```bash
ais_bench [OPTIONS]
```
### 参数说明
根据执行场景，命令行参数分为三大类：

- 公共参数
- 精度测评参数（仅在 `--mode` 为 `all、infer、eval` 或 `viz` 时生效）
- 性能测评参数（仅在 `--mode` 为 `perf` 或 `perf_viz` 时生效）

`精度测评参数`只有在`--mode`参数指定为`"all", "infer", "eval", "viz"`时生效，`性能测评参数`只有在`--mode`参数指定为`"perf", "perf_viz"`时生效，`公共参数`则不区分任务执行模式，在所有模式下均可指定。

### 公共参数
适用于所有模式，可同时与精度或性能参数联合使用。
| 参数| 说明| 示例|
| ---- | ---- | ----|
| `--models`| 指定模型推理后端任务名称（对应 `ais_bench/benchmark/configs/models` 路径下一个已经实现的默认模型配置文件），支持传入多个任务名称。详情参考📚 [支持的模型](./models.md)| `--models vllm_api_general`  |
| `--datasets`   | 指定数据集任务名称（对应 `ais_bench/benchmark/configs/datasets` 路径下一个已经实现的默认数据集配置文件），可传入多个。详情参考📚 [支持的数据集类型](./datasets.md)| `--datasets gsm8k_gen`    |
| `--summarizer` | 指定结果总结任务名称（对应 `ais_bench/benchmark/configs/summarizers` 路径下一个已经实现的默认模型配置文件）。详情参考📚 [支持的结果汇总任务](./summarizer.md) | `--summarizer medium`|
| `--mode` 或 `-m`| 运行模式，可选：`all`、`infer`、`eval`、`viz`、`perf`、`perf_viz`；默认 `all`。<br>详细请见 📚 [运行模式说明](./mode.md)。 | `--mode infer`<br>`-m all`|
| `--reuse` 或 `-r`| 指定已有工作目录下的时间戳，继续执行并覆盖原有结果。结合`--mode`参数值，可用于推理中断续推，或基于已有推理结果执行精度计算、可视化结果打印。若不加参，则自动选取 `--work-dir` 下最新时间戳。| `--reuse 20250126_144254`<br>`-r 20250126_144254` |
| `--work-dir` 或 `-w`     | 指定评测工作目录，用于保存输出结果。默认 `outputs/default`。| `--work-dir /path/to/work`<br>`-w /path/to/work` |
| `--config-dir` | `models`，`datasets`和`summarizers`配置文件所在的文件夹路径，默认 `ais_bench/benchmark/configs`。    | `--config-dir /xxx/xxx`   |
| `--debug` | 开启 Debug 模式，配置该参数表示开启，未配置表示关闭，默认未配置。debug模式下所有日志将直接打印在终端。(debug模式下`--max-num-workers`参数将强制设置为1，串行执行每个任务，且只会调用单核执行任务，并发能力受限)    | `--debug`   |
| `--dry-run`    | 开启 Dry Run 模式（只打屏不实际跑任务）开关，配置该参数表示开启，未配置表示关闭，默认未配置。  | `--dry-run` |
| `--max-workers-per-gpu` | 预留参数，暂不支持。 | `--max-workers-per-gpu 1` |
| `--merge-ds`   | 开启同类数据集合并推理（同一任务多数据集一起跑）。| `--merge-ds`|
| `--num-prompts` | 指定数据集测评条数（按照数据集顺序选取），需传入正整数，超过数据集条数或默认情况下表示对全量数据集进行测评。 | `--num-prompts 500` |
| `--max-num-workers`   | 并行任务数，范围 `[1, CPU 核数]`，默认 `1`。在指定`--debug`时配置无效，所有任务串行执行。注意：性能测评场景下，并发数过高可能会导致不同进程出现资源抢占，导致测试结果失真。  | `--max-num-workers 2` |
|`--num-warmups`|发送请求前预热次数，按照数据集顺序选取数据进行测试，大概num-warmups大于数据集条数时，会循环发送数据集中数据。默认 `1`；若设为0，则不预热。如果warmup阶段所有请求失败，后续推理任务将不会执行。| `--num-warmups 10` |
| `--response-anomaly` / `--no-response-anomaly` | 开启或关闭 msProbe 推理响应异常检测。命令行配置优先于配置文件中的 `response_anomaly.enabled`。检测在线程中与 Eval 并行运行；需服务返回 token id 与 top-k logprobs。仅支持 `all`、`infer`、`infer_judge` 普通生成链路，不支持性能模式与 Agent 测评模式。 | `--response-anomaly` |
| `--response-anomaly-payload-retention` | 异常检测完成后的 payload 保存模式：`all` 保存全部，`anomalies` 保存异常及检测失败/不可用 Case，`none` 不保存。命令行配置优先于配置文件，默认 `anomalies`。 | `--response-anomaly-payload-retention anomalies` |

### 精度测评参数
仅在模式为 `all、infer、eval` 或 `viz` 时有效。
| 参数| 说明  | 示例|
| ---- | ---- | ---- |
| `--dump-eval-details` | 是否dump出评测过程细节的开关，配置该参数表示开启，未配置表示关闭，默认未配置。  | `--dump-eval-details` |
| `--dump-extract-rate` | 是否dump出评测速度的开关，配置该参数表示开启，未配置表示关闭，默认未配置。    | `--dump-extract-rate` |

### 性能测评参数
仅在模式为 `perf` 或 `perf_viz` 时有效。
| 参数| 说明| 示例 |
| ---- | ---- | ---- |
| `--pressure`   | 	是否开启性能压测方式的开关，仅当 `--mode perf` 时有效，配置该参数表示开启，未配置表示关闭，默认未配置。压力测试详情可参考:📚 [压力测试使能稳态测试](../../advanced_tutorials/stable_stage.md#压力测试使能稳态测试)。| `--pressure`|
|`--pressure-time`|压测持续时间，仅在指定 `--pressure` 模式时生效。单位为秒，默认15秒，取值范围为 `[1, 86400]`（即 1 秒 至 24 小时）。| `--pressure-time 30`|
|`--spec-decode`|启用投机推理（Speculative Decoding）指标采集，从推理服务的 Prometheus `/metrics` 端点拉取指标。仅在 `--mode perf` 时有效。详细用法见 📚 [投机推理指标采集](../../advanced_tutorials/spec_decode.md)。| `--spec-decode` |

## 配置常量文件参数

## 推理响应异常检测配置

当前响应异常检测仅支持基于 vLLM Chat API 的 `vllm_api_general_chat`、`vllm_api_stream_chat` 和 `vllm_api_stream_chat_multiturn` 模型配置，其他模型后端暂不支持。

在总配置文件中增加 `response_anomaly` 可启用检测，也可通过 `--response-anomaly` 覆盖：

```python
response_anomaly = dict(
	enabled=True,
)
```

模型相关的 msProbe 配置放在模型配置中：

```python
models = [
	dict(
		abbr='qwen3-30b',
		attr='service',
		response_anomaly=dict(
			model_name="",   # 填写模型名称，如 Qwen3-30B-A3B
			model_path="",   # 填写本地模型目录，如 /home/Qwen3-30B-A3B；可选，用于自动生成配置
			msprobe_mtype_path='/path/to/mtype_config.json',
			msprobe_token2category_dir='/path/to/token2category/',
		),
	),
]
```

未提供 `msprobe_mtype_path` / `msprobe_token2category_dir` 时回退到 msProbe 包内默认文件；配置了 `model_path` 时会自动生成到 `<work_dir>/response_anomaly_config/<模型 abbr>/`。也可手动生成：

```bash
ais_bench-gen-response-anomaly-config \
  --model-path /home/Qwen3-30B-A3B \
  --model-name Qwen3-30B-A3B \
  --output-dir ./msprobe_configs
```

启用后，AISBench 会在服务推理请求中补充 `logprobs=True` 与固定的 `top_logprobs=20`；该值由检测算法约束，不支持外部配置。推理阶段将完整 payload 直接写入 `response_anomaly/<模型>/payload_staging/<数据集>/*.jsonl.zst`，prediction 从一开始只保存轻量结果。推理结束后，检测线程流式解压 staging 数据并调用 msProbe，检测结果写入 `response_anomaly/<模型>/<数据集>.jsonl`；每个 Case 包含 `is_anomaly`、`anomaly_type`（0：正常，1：生僻字，2：乱码，3：重复，4：NaN Value）和 `detection_status`。检测完成后按 `payload_retention` 保留或清理 staging。状态面板会显示配置准备、检测器加载、流式检测和归档收尾阶段。

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

`all` 保存全部 payload，检测后直接将 staging 原子转为正式归档，不会二次压缩；`anomalies` 只保存已检出异常以及检测失败/不可用 Case；`none` 不保存 payload。三种模式都保留独立检测结果。`--reuse` 必须沿用原工作目录的保留策略。检测结果按批写盘，状态最多每秒刷新一次；msProbe token 分类映射按模型和 EOS token 缓存，避免每个 Case 重复解析大 JSON 文件。

检测通过 msProbe 的 `ILLDetector(config_path, mtype_path, tk2cat_path).run(...)` 完成，三个文件路径均可由 AISBench 配置。请先安装 AISBench 的可选依赖：`pip install 'ais-bench-benchmark[response_anomaly]'`。安装过程中 pip 会从 GitCode 下载并构建已固定提交的 msProbe 源码，因此安装环境需要 Git 和网络访问。服务响应必须包含 `token_ids`（或 `tokens`）和 `topk_logprobs`；缺少这些字段的 Case 会以 `skipped` 状态落盘。`model_name` 需与 msProbe 的 `mtype_config.json` 配置以及 token 分类映射保持一致。使用 `--reuse` 时，已有检测结果按 Case id 继承，已完成 Case 不会重复检测。

部分全局常量不区分任务类型，推荐保持默认；如需自定义，可编辑常量文件：[`global_consts.py`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/global_consts.py)配置。
当前支持的参数配置如下：
| 参数名| 说明| 取值范围 / 要求 |
| ----------- | ----------- | ----------- |
|`WORKERS_NUM`|请求发送所用的进程数。 默认为0， 根据用户配置的请求最大并发数自动分配。（在指定命令行参数`--debug`时配置无效，调用单核发送请求，并发能力存在限制）|[0, cpu核数]|
| `MAX_CHUNK_SIZE` | 流式推理模型后端返回的单个 chunk 最大缓存大小。默认值为 65535 字节（64KB）。 | `(0, 16777216]`（单位：Byte） |
| `REQUEST_TIME_OUT` | Client 端请求发送后等待返回的超时时间。默认为 None，即无限等待，始终等待模型返回结果。 | `None` 或 `>0`（单位：秒）|
|`LOG_LEVEL`|日志级别，可选：`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`。默认 `INFO`。|`[DEBUG, INFO, WARNING, ERROR, CRITICAL]`|
| `PRESSURE_TIME`| 压测持续时间，仅在指定 `--pressure` 模式时生效。单位为秒。(该参数将在未来版本中废弃，请使用 `--pressure-time` 参数代替)| `[1, 86400]`（即 1 秒 至 24 小时） |
| `CONNECTION_ADD_RATE`| 并发线程创建速率。表示每秒新增的并发线程数，直至达到最大并发限制。仅在指定 `--pressure` 模式时生效。(该参数将在未来版本中废弃，请在模型配置文件中修改 `request_rate` 参数代替) | `> 0.1`（单位：线程数 / 秒） |
