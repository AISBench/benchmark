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
|`config`|指定自定义配置文件路径|`ais_bench /path/to/custom_config.py {other optional arguments}`|
| `--models`| 指定模型推理后端任务名称（对应 `ais_bench/benchmark/configs/models` 路径下一个已经实现的默认模型配置文件），支持传入多个任务名称。详情参考📚 [支持的模型](./models.md)。<br> ⚠️注意：指定了自定义配置文件路径后此参数无效| `--models vllm_api_general`  |
| `--datasets`   | 指定数据集任务名称（对应 `ais_bench/benchmark/configs/datasets` 路径下一个已经实现的默认数据集配置文件），可传入多个。详情参考📚 [支持的数据集类型](../../get_started/datasets.md)。<br> ⚠️注意：指定了自定义配置文件路径后此参数无效| `--datasets gsm8k_gen`    |
| `--summarizer` | 指定结果总结任务名称（对应 `ais_bench/benchmark/configs/summarizers` 路径下一个已经实现的默认模型配置文件）。详情参考📚 [支持的结果汇总任务](./summarizer.md) 。<br> ⚠️注意：指定了自定义配置文件路径后此参数无效| `--summarizer medium`|
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
| `--response-anomaly` | 开启 msProbe 推理响应异常检测。该开关**仅支持命令行配置**：命令行增加 `--response-anomaly` 即开启，不增加默认不开启。检测串行绑定在推理阶段内：推理结束后启动检测并等待其完成（专属状态面板打印最终结果后）才进入后续 Judge / Eval / 汇总流程；需服务返回 token id 与 top-k logprobs。仅支持 `all`、`infer`、`infer_judge` 普通生成链路，不支持性能模式与 Agent 测评模式。详细用法见 📚 [推理响应异常检测](../../advanced_tutorials/response_anomaly_detection.md)。 | `--response-anomaly` |
| `--response-anomaly-payload-retention` | 异常检测完成后的 payload 保存模式：`all` 保存全部，`anomalies` 保存异常及检测失败/不可用 Case，`none` 不保存。命令行配置优先于配置文件，默认 `anomalies`。 | `--response-anomaly-payload-retention anomalies` |

### API 模型通用覆盖参数

适用于服务化推理后端（API 模型，如 vLLM、Triton、MindIE、TGI 等），用于在不修改模型配置文件的前提下，通过命令行直接覆盖模型配置中的常用字段。

> ⚠️ **覆盖范围说明**：
> - 仅覆盖模型配置中**已存在的字段**，不新增字段（避免向不支持某字段的模型类传入多余参数，保证向后兼容）。
> - 命令行显式指定的参数会覆盖**执行的所有模型配置**中对应的字段（同一命令下多个模型任务均生效）。
> - 未显式指定的参数不生效（默认 `None`），配置文件中保持原值。
> - `model` / `model_name` 字段按模型 `type` 的构造签名自动选择写入目标：VLLM 系列写入 `model`，Triton 写入 `model_name`；模型类型两者都不接收（如 MindIE、TGI）时打印 warning 并跳过。

| 参数 | 说明 | 示例 |
| ---- | ---- | ---- |
| `--path` | 覆盖模型配置的 `path` 字段（Tokenizer/模型序列化词表文件路径） | `--path /weight/Qwen` |
| `--model-name` | 覆盖模型名。按模型 `type` 自动写入 `model` 或 `model_name` 字段（VLLM→`model`，Triton→`model_name`；MindIE/TGI 不接收时 warning 跳过） | `--model-name Qwen` |
| `--request-rate` | 覆盖 `request_rate`（请求发送速率） | `--request-rate 10` |
| `--retry` | 覆盖 `retry`（每个请求最大重试次数） | `--retry 3` |
| `--api-key` | 覆盖 `api_key`（自定义 API key） | `--api-key sk-xxx` |
| `--host-ip` | 覆盖 `host_ip`（推理服务 IP） | `--host-ip 127.0.0.1` |
| `--host-port` | 覆盖 `host_port`（推理服务端口） | `--host-port 8000` |
| `--url` | 覆盖 `url`（自定义访问推理服务的 URL 路径） | `--url http://x.x.x.x:8000/v1` |
| `--max-out-len` | 覆盖 `max_out_len`（推理输出最大 token 数） | `--max-out-len 1024` |
| `--batch-size` | 覆盖 `batch_size`（请求最大并发数） | `--batch-size 4` |
| `--trust-remote-code` | 覆盖 `trust_remote_code`（tokenizer 是否信任远程代码），支持 `--trust-remote-code` / `--no-trust-remote-code` 两种形态 | `--no-trust-remote-code` |
| `--generation-kwargs` | 覆盖 `generation_kwargs`（推理生成参数），以 JSON 对象形式传入，**整体替换**配置中的 dict | `--generation-kwargs '{"temperature": 0.5}'` |

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
