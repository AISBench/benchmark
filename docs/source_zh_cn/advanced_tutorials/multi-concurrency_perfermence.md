# 多档并发场景服务化性能测试

## 一. 简介

多档并发场景是常见的多任务组合性能压测场景，旨在通过分析不同并发度的性能测试结果，找到模型的性能瓶颈。
本文档提供 AISBench 支持的命令行使用样例。具体而言，一次命令即可把并发数（`--batch-size`）与请求条数（`--num-prompts`）展开为多组任务，跑完后统一汇总为 CSV 与 HTML 报告，用于对比不同并发档下的吞吐与时延表现：

- **吞吐维度**：请求吞吐量、输出 Token 吞吐量随并发上升是否仍在增长
- **时延维度**：E2EL、TTFT、TPOT、ITL 的 Average 与 P99 随并发上升的劣化速度
- **瓶颈定位**：吞吐进入平台期、而 P99 时延开始陡增的并发档，即为该服务配置下的性能拐点


## 二. 快速入门

### 2.1 命令样例

前提条件：

1. 已完成服务化部署，且 `/v1/models` 返回的 model id 与模型配置中的 `model` 字段一致
2. 使用随机合成数据集，数据集配置参考 [随机合成数据集使用指南](synthetic_dataset.md)

样例：

```bash
ais_bench \
  --models vllm_api_stream_chat \    # 使用流式对话模型
  --datasets synthetic_gen_tokenid \ # 使用随机token id数据集
  --mode perf \                      # 性能测试场景
  --tokenid-config-request-size 8192 \ # 单请求token数量（输入长度）
  --tokenid-config-prefix-len 0 \ # 请求的公共前缀token数量（前缀长度）
  --batch-size 1 8 16 32 40 50 64 \ # 请求发送的最大并发数
  --num-prompts 1 8 16 32 40 50 64 \ # 请求发送总数
  --request-count 64 \ # 随机合成数据集生成的请求总数（大于等于请求发送总数）
  --path /mnt/weight/Qwen3.8-27B-w8a8 \ # Tokenizer/模型序列化词表文件路径
  --model-name qwen3.8 \ # 服务端已加载模型名称
  --max-out-len 1024 \ # 推理服务输出最大token数（输出长度）
  --num-warmups 16 \ # 发送请求前预热次数
  --generation-kwargs '{"ignore_eos":true}' # 推理服务输出忽略eos
```

说明：

- `--batch-size` 传入多个值时按并发档展开；`--num-prompts` 与 `--batch-size` 取值个数一致，按位置一一对应
- 每个并发档生成独立任务，模型 `abbr` 自动追加 `-bs{N}` 后缀（如 `vllm_api_stream_chat-bs8`），落盘文件互不覆盖
- `--request-count` 指定每档生成的请求总数；若未指定 `--num-prompts`，实际发送请求数取 `--request-count`
- 档间默认串行执行，并留有 5s 冷却时间，避免上一档的排队请求污染下一档结果

多组输入输出 + 多档并发的样例，详见 [aisbench_vllm_performance_zh_cn.sh](../../../ais_bench/configs/performance_benchmark/aisbench_vllm_performance_zh_cn.sh)。

### 2.2 执行结果

**执行过程**

命令行看板会按任务逐行刷新，每个并发档是一个独立任务：

```
Base path of result&log : outputs/default/20251102_103326
+---------------------------------------+-----------+-------------+-----------+-----------+
| Task Name                             |  Process  | Progress    | Time Cost | Status    |
+---------------------------------------+-----------+-------------+-----------+-----------+
| vllm_api_stream_chat-bs1/synthetic                 |   1024    | [#########] | 0:01:12   | finished  |
| vllm_api_stream_chat-bs8/synthetic                 |   1024    | [######    ] | 0:02:05   | inferencing |
+---------------------------------------+-----------+-------------+-----------+-----------+
```

**产物结构**

```
outputs/default/20251102_103326/
├── combos_manifest.json          # 组合参数快照（并发档与请求条数的对应关系）
├── batch_perf_summary.csv        # 多组合汇总指标（可透视/排序）
├── batch_perf_report.html        # 多组合可视化报告（单文件，可离线打开）
├── performances/
│   ├── vllm_api_stream_chat-bs1/
│   │   ├── synthetic.csv         # 单任务指标统计
│   │   ├── synthetic.json        # 单任务 common 指标
│   │   ├── synthetic_details.jsonl  # 逐请求明细
│   │   └── synthetic.html        # 单任务报告
│   └── vllm_api_stream_chat-bs8/
└── logs/infer/                   # 各任务执行日志
```

**汇总报告**

`batch_perf_report.html` 按以下章节组织，支持中英文切换与深浅色主题：

| 章节 | 内容 |
| --- | --- |
| 报告概览 | 运行摘要、全任务指标对比宽表、最佳性能配置、结论与建议 |
| 指标统计 | 每个任务的指标统计表（与 `synthetic.csv` 内容一致） |
| 百分位曲线 | E2EL、TTFT/TPOT/ITL、输入/输出 Token 数、Token 吞吐的百分位曲线 |
| 时延分析 | 时延、首 Token 时延、每 Token 时延随并发变化曲线 |
| 吞吐分析 | 请求吞吐量、Token 吞吐量、成功率随并发变化曲线 |
| 逐任务详情 | 每个任务的运行摘要、指标统计表、逐请求明细曲线 |
| 口径与说明 | 指标定义、缺失说明与运行期告警 |

具体指标含义可参考📚 [性能测评结果说明](../base_tutorials/results_intro/performance_metric.md)。

## 三. 注意事项

1. **请求条数不少于并发档**：当某一档的 `--batch-size` 大于该档实际发送的请求数时，真实并发受请求数限制，命令行会给出并发窗口告警，报告中的观测值将不代表该档的目标并发

2. **启动前校验**：`--batch-size` 与 `--num-prompts` 必须一一对应，两侧取值个数不一致（如 `--batch-size` 多值而 `--num-prompts` 只给一个值）会在启动前直接报错退出，不会产生半途任务

3. **非流式无时延拆分**：`--no-stream` 时无法测得首 Token 时延与逐 Token 时延，汇总 CSV 中对应统计值留空，报告中不绘制相关曲线

4. **单档不生成汇总**：只传一个 `--batch-size` 时按单任务流程执行，不生成 `combos_manifest.json` 与汇总报告

5. **失败隔离**：某一档任务失败不影响后续档执行，失败档在 CSV 中标记为 `fail`，其余档正常汇总
