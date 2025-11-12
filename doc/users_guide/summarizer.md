# 支持的结果汇总任务
| 任务名称 | 简介 | 配置文件路径 |
| -------------- | -------------- | -------------- |
| `example` | 简化版精度评测结果汇总模板，覆盖当前支持的所有数据集，是默认使用的模板。   | [example.py](../../ais_bench/benchmark/configs/summarizers/example.py)  |
| `medium`  | 通用精度评测结果汇总模板，适用于多个基础数据集。| [medium.py](../../ais_bench/benchmark/configs/summarizers/medium.py)    |
| `default_perf` | 全量性能评测结果汇总模板，汇总所有请求的性能数据。支持通过 `default_perf.py` 手动配置性能统计指标。 | [default\_perf.py](../../ais_bench/benchmark/configs/summarizers/perf/default_perf.py) |
| `stable_stage` | 稳定阶段性能评测结果汇总模板，仅汇总系统达到配置最大并发时的请求数据。支持通过 `stable_stage.py` 手动配置性能统计指标。 | [stable\_stage.py](../../ais_bench/benchmark/configs/summarizers/perf/stable_stage.py) |
