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
| `--models`| 指定模型推理后端任务名称（对应 `ais_bench/benchmark/configs/models` 路径下一个已经实现的默认模型配置文件），支持传入多个任务名称，；与 `config` 参数二选一。详情参考📚 [支持的模型](./models.md)| `--models vllm_api_general`  |
| `--datasets`   | 指定数据集任务名称（对应 `ais_bench/benchmark/configs/datasets` 路径下一个已经实现的默认数据集配置文件），可传入多个；与 `config` 参数二选一。详情参考📚 [支持的数据集类型](./datasets.md)| `--datasets gsm8k_gen`    |
| `--summarizer` | 指定结果总结任务名称（对应 `ais_bench/benchmark/configs/summarizers` 路径下一个已经实现的默认模型配置文件）。详情参考📚 [支持的结果汇总任务](./summarizer.md) | `--summarizer medium`|
| `--mode` 或 `-m`| 运行模式，可选：`all`、`infer`、`eval`、`viz`、`perf`、`perf_viz`；默认 `all`。<br>详细请见 📚 [运行模式说明](./mode.md)。 | `--mode infer`<br>`-m all`|
| `--reuse` 或 `-r`| 指定已有工作目录下的时间戳，继续执行并覆盖原有结果。结合`--mode`参数值，可用于推理中断续推，或基于已有推理结果执行精度计算、可视化结果打印。若不加参，则自动选取 `--work-dir` 下最新时间戳。| `--reuse 20250126_144254`<br>`-r 20250126_144254` |
| `--work-dir` 或 `-w`     | 指定评测工作目录，用于保存输出结果。默认 `outputs/default`。| `--work-dir /path/to/work`<br>`-w /path/to/work` |
| `--config-dir` | `models`，`datasets`和`summarizers`配置文件所在的文件夹路径，默认 `ais_bench/benchmark/configs`。    | `--config-dir /xxx/xxx`   |
| `--debug` | 开启 Debug 模式，配置该参数表示开启，未配置表示关闭，默认未配置。debug模式下所有日志将会直接打印在终端，日志会打印到终端。    | `--debug`   |
| `--dry-run`    | 开启 Dry Run 模式（只打屏不实际跑任务）开关，配置该参数表示开启，未配置表示关闭，默认未配置。  | `--dry-run` |
| `--max-workers-per-gpu` | 预留参数，暂不支持。 | `--max-workers-per-gpu 1` |
| `--merge-ds`   | 开启同类数据集合并推理（同一任务多数据集一起跑）。| `--merge-ds`|
| `--num-prompts` | 指定数据集测评条数，需传入正整数，超过数据集条数或默认情况下表示对全量数据集进行测评。 | `--num-prompts 500` |
| `--max-num-workers`   | 并行任务数，范围 `[1, CPU 核数]`，默认 `1`。在 Continuous Batch 或性能模式下无效。  | `--max-num-workers 2` |
|`--num-warmups`|发送请求前预热次数，默认 `1`；若设为0，则不预热。| `--num-warmups 10` |

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

## 配置常量文件参数

部分全局常量不区分任务类型，推荐保持默认；如需自定义，可编辑常量文件：[`global_consts.py`](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/global_consts.py)配置。
当前支持的参数配置如下：
| 参数名| 说明| 取值范围 / 要求 |
| ----------- | ----------- | ----------- |
|`WORKERS_NUM`|请求发送所用的进程数。 默认为0， 根据用户配置的请求最大并发数自动分配。|[0, cpu核数]|
| `MAX_CHUNK_SIZE` | 流式推理模型后端返回的单个 chunk 最大缓存大小。默认值为 65535 字节（64KB）。 | `(0, 16777216]`（单位：Byte） |
| `REQUEST_TIME_OUT` | Client 端请求发送后等待返回的超时时间。默认为 None，即无限等待，始终等待模型返回结果。 | `None` 或 `>0`（单位：秒）|
|`LOG_LEVEL`|日志级别，可选：`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`。默认 `INFO`。|`[DEBUG, INFO, WARNING, ERROR, CRITICAL]`|
