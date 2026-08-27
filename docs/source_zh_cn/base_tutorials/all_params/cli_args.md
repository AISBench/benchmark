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
| `--response-anomaly` | 开启 msProbe 推理响应异常检测。该开关**仅支持命令行配置**：命令行增加 `--response-anomaly` 即开启，不增加则关闭。检测串行绑定在推理阶段内：推理结束后启动检测并等待其完成（专属状态面板打印最终结果后）才进入后续 Judge / Eval / 汇总流程；需服务返回 token id 与 top-k logprobs。仅支持 `all`、`infer`、`infer_judge` 普通生成链路，不支持性能模式与 Agent 测评模式。 | `--response-anomaly` |
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

## 推理响应异常检测配置

当前响应异常检测仅支持基于 vLLM Chat API 的 `vllm_api_general_chat`、`vllm_api_stream_chat` 和 `vllm_api_stream_chat_multiturn` 模型配置，其他模型后端暂不支持。

异常检测的**功能开关仅支持命令行**：在命令中增加 `--response-anomaly` 开启（不增加即关闭）。配置文件中的 `response_anomaly` 仅用于非开关类配置（`payload_retention`、`payload_storage` 等）：

```python
response_anomaly = dict(
    payload_retention='anomalies',  # all | anomalies | none
    payload_storage=dict(
        format='jsonl',
        compression='zstd',
        compression_level=3,
        rows_per_shard=2000,
    ),
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
			msprobe_config_path="",  # 可选，msProbe 算法阈值配置 config.yaml 路径，用于手工调优检测阈值
			msprobe_mtype_path="",  # 可选，msProbe 模型名与 BOS/EOS token id 映射文件 mtype_config.json 路径
			msprobe_token2category_dir="",  # 可选，msProbe token2category 目录，存放各模型的 token id 到字符类别映射
		),
	),
]
```

未提供 `msprobe_mtype_path` / `msprobe_token2category_dir` 时回退到 msProbe 包内默认文件；配置了 `model_path` 时会自动生成到 `<work_dir>/response_anomaly_config/<模型 abbr>/`（自动生成不会覆盖已存在的 `config.yaml`，便于保留手工调优的阈值）。也可手动生成：

```bash
ais_bench-gen-response-anomaly-config \
  --model-path /home/Qwen3-30B-A3B \
  --model-name Qwen3-30B-A3B \
  --output-dir ./msprobe_configs
```

`model_name` 未显式配置时自动取模型路径（`model_path` 或模型 `path` 字段）中的**模型名称**（如 `/home/Qwen3-30B-A3B` → `Qwen3-30B-A3B`），与配置生成工具的默认取值一致。当既未配置 `model_name`、也没有可用的模型路径（如仅显式配置 msprobe 三件套路径）时，任务启动即报错，要求显式配置 `model_name`，避免以错误的模型名静默运行导致检测失效。

启用后，AISBench 会在服务推理请求中补充 `logprobs=True` 与固定的 `top_logprobs=20`；该值由检测算法约束，不支持外部配置。对 vLLM 后端还会追加 `return_token_ids=True` 与 `return_tokens_as_token_ids=True` 以获取 token id，服务端版本过低不支持这些参数时请求可能失败，需升级 vLLM。推理阶段将完整 payload 直接写入 `response_anomaly/<模型>/payload_staging/<数据集>/*.jsonl.zst`，prediction 从一开始只保存轻量结果。推理结束后，检测线程流式解压 staging 数据并调用 msProbe，检测结果写入 `response_anomaly/<模型>/<数据集>.jsonl`；每个 Case 包含 `id`、`uuid`、`is_anomaly`、`anomaly_type`（0：正常，1：生僻字，2：乱码，3：重复，4：NaN Value）、`anomaly_type_name`（类型名字符串，如 `normal`/`garbled`/`repetition`，统计时更常用）和 `detection_status`。检测完成后按 `payload_retention` 保留或清理 staging。状态面板会显示配置准备、检测器加载、流式检测和归档收尾阶段。

检测结果中的 `detection_status` 取值如下：

| 状态 | 含义 | 排查建议 |
| --- | --- | --- |
| `completed` | 已调用 msProbe 并得到检测结果 | 无需处理 |
| `skipped` | 推理响应未携带 token id 或 top-k logprobs | 检查服务端是否支持并返回 `logprobs` / `top_logprobs` / token id 字段 |
| `unavailable` | 未安装 `mindstudio-probe`（response_anomaly extra） | 参考[安装指南](../../get_started/install.md)安装可选依赖后重跑 |
| `failed` | 调用或输入转换发生异常 | 查看该 Case 结果中的 `reason` 字段（保存错误类型与摘要）及检测日志 |

**异常检测结果不影响原有评测指标**：异常 Case 不会被改写为推理失败，精度/性能指标照常计算，异常信息是独立的审计结果。检测专属日志位于 `<work_dir>/logs/response_anomaly/<模型>/<数据集>.out`，检测进度与类型统计也可在 `<work_dir>/status_tmp/tmp_ResponseAnomaly.json` 状态文件中查看。

```python
response_anomaly = dict(
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

检测相关文件在 `<work_dir>` 下的落盘结构如下：

```text
<work_dir>/
├── predictions/<模型 abbr>/<数据集 abbr>.jsonl          # 推理结果（轻量，不含 token/logprobs payload）
├── response_anomaly/
│   └── <模型 abbr>/
│       ├── <数据集 abbr>.jsonl                          # 检测结果，每行一个 Case
│       ├── payload_staging/<数据集 abbr>/               # 推理期间的临时存放区，检测完成后自动清理
│       │   └── part-*.jsonl.zst
│       └── payload/<数据集 abbr>/                       # payload 归档；payload_retention 为 none 时不存在
│           ├── payload_manifest.json                    # 归档清单（分片行数、大小、sha256）
│           └── part-*.jsonl.zst                         # 压缩 payload 分片（每片最多 rows_per_shard 条 Case）
├── response_anomaly_config/                             # 仅配置 model_path 自动生成时存在
│   └── <模型 abbr>/
│       ├── configs/
│       │   ├── config.yaml                              # 检测算法阈值配置（已存在时不覆盖）
│       │   └── mtype_config.json                        # 模型名与 BOS/EOS token id 映射
│       └── token2category/
│           └── <模型名>_<词表大小>.json                  # token id 到字符类别映射
└── logs/
    └── response_anomaly/<模型 abbr>/<数据集 abbr>.out   # 检测专属日志
```

各路径说明：

- **检测结果** `response_anomaly/<模型 abbr>/<数据集 abbr>.jsonl`：每行一个 Case 的检测结果，字段含义见上文 `detection_status` 表与 Case 字段说明。
- **payload 归档** `response_anomaly/<模型 abbr>/payload/<数据集 abbr>/`：`all` 保留全部 Case，`anomalies` 只保留异常及检测失败/不可用 Case，`none` 不保留（目录不存在）。读取时用 zstandard 解压 `part-*.jsonl.zst` 分片后逐行解析 JSON；`payload_manifest.json` 记录每个分片的行数、大小与 sha256 校验值，可用于完整性校验。注意：`anomalies` 模式下即使无任何需保留的 Case，仍会发布一个仅含空 manifest 的归档目录，表示归档流程已成功完成，不是残留文件。
- **临时文件**：`payload_staging/` 在推理期间逐条接收 payload 写入，检测完成后自动清理；检测中断后残留的 `.<数据集>.payload-build-*` 构建目录会在下次检测启动时自动清理；`status_tmp/tmp_ResponseAnomaly.json` 为运行期状态文件（检测进度与类型统计），工作流结束后随状态目录一并清理。
- **自动生成的 msProbe 配置** `response_anomaly_config/<模型 abbr>/`：仅当配置了 `model_path` 且未显式提供 mtype/token2category 路径时生成。`config.yaml` 已存在时不会被覆盖（保留手工调优的阈值）；`mtype_config.json` 支持多模型合并，多次生成不互相覆盖。
- **检测日志** `logs/response_anomaly/<模型 abbr>/<数据集 abbr>.out`：记录对应模型/数据集组的检测过程，含检测器初始化失败与单 Case 失败的具体原因。

检测通过 msProbe 的 `ILLDetector(config_path, mtype_path, tk2cat_path).run(...)` 完成，三个文件路径均可由 AISBench 配置。请先安装 AISBench 的可选依赖：`pip install 'ais-bench-benchmark[response_anomaly]'`。安装过程中 pip 会从 GitCode 下载并构建已固定提交的 msProbe 源码，因此安装环境需要 Git 和网络访问。服务响应必须包含 `token_ids`（或 `tokens`）和 `topk_logprobs`；缺少这些字段的 Case 会以 `skipped` 状态落盘。`model_name` 需与 msProbe 的 `mtype_config.json` 配置以及 token 分类映射保持一致。使用 `--reuse` 时，已有检测结果按 Case 的 `id` + `uuid` 双键匹配继承（`uuid` 变化说明该 Case 已重新推理，不会错挂旧结果），`completed` 状态的 Case 不会重复检测；`skipped` / `failed` / `unavailable` 状态的 Case 会在续跑中重新检测。

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
