# 标准 AISBench CLI 适配

[返回快速开始](../README.md)

测试容器内预先安装可运行服务化性能测试的 AISBench 及其常规依赖，不要求额外插件。默认使用外部数据；明确授权的GSM8K生成流程见[数据生成](generated-data.md)。不打包数据、模型或 PD 代理。

## 数据集绑定

用户提供服务器上的 JSONL 或 CSV 文件绝对路径。Agent 须在 bindings 的 mounts 中声明覆盖数据集和 generation_manifest 的挂载，并核验容器可读；生成器仅自动挂载任务与模型目录，不自动补数据挂载。数据不随 Skill 源码或 Git 仓库打包。外部数据通常由挂载读取；如需发布，须明确列为最小运行时的授权输入并记录传输凭据，不自动复制数据集。

在节点 `bindings.benchmark.datasets` 中声明数据集，以 `case.dataset_id` 引用：

```json
{
  "benchmark": {
    "datasets": {
      "default": {
        "path": "/data/benchmark/dataset.jsonl",
        "input_column": "question",
        "output_column": "answer"
      }
    }
  }
}
```

JSONL 每行一个对象，或 CSV 首行为列名。默认输入字段为 `question`，参考答案字段为 `answer`；性能测试时参考答案可为空字符串。可通过绑定改为实际字段名。使用 AISBench `CustomDataset` 原生格式；相邻元数据文件及逐行输出长度字段仍遵循 AISBench 的规则，应与本次测试目标一致。

外部数据集内容决定输入长度、前缀和顺序；默认不改写输入。授权生成的数据须绑定generation_manifest并核验摘要，不承诺前缀命中率。`input_tokens` 为可选的严格长度校验值；变长数据集应省略。`output_tokens` 表示单请求最大输出长度，允许提前结束。

不同数据集添加不同绑定键，在各 case 中明确选择；不再接受旧版 `prefix_ratio`、`prefix_num`、`seed` 数据生成参数。

## 标准命令

每个 attempt 生成一个仅引用已安装 AISBench 类与外部数据集路径的配置文件，然后执行：

```sh
ais_bench /work/attempt/aisbench-config.py \
  --mode perf \
  --work-dir /work/attempt/outputs \
  --num-prompts 100 \
  --num-warmups 1
```

配置使用 `VLLMCustomAPI` 流式 completions 接口、`CustomDataset` 和原生性能统计器。并发、请求速率、模型名、tokenizer 路径和服务地址来自冻结配置。`steady_state=true` 选择 `stable_stage` 统计器，不自动开启改变请求数量的压力模式。

默认请求数为并发×4，默认原生预热 1 次；`num_warmups` 可覆盖，`warmup=false` 禁用预热。缓存清理默认关闭；明确设置 `reset_cache=true` 时才使用已核验服务的 reset 接口，接口不支持则阻塞，不能假报清理成功。重试保留独立结果；未启用清理时不保证各次尝试缓存状态一致。

模型部署仍需模型运行环境、NPU 与权重。PD 部署需用户提供 `bindings.paths.proxy_program`，不属于 AISBench 测试客户端依赖。

## 日志落盘与查看

使用标准模式，不加 `--debug`，让 AISBench 将 worker 日志原生落盘到：

```text
scripts/tests/SUBTASK.config.py   # 冻结的原生配置
scripts/tests/SUBTASK.sh          # 包含实际 AISBench 命令
results/SUBTASK-tryNN/
├── aisbench.log                 # AISBench 主进程 stdout/stderr
├── execution.json              # CLI 退出码（执行进入 after 阶段后）
├── outputs/<timestamp>/
│   ├── configs/
│   ├── logs/infer/<model>/<dataset>.out
│   ├── performances/
│   └── summary/
├── validated.json              # 验证通过后生成
├── prefix-evidence.json         # 包含预热的缓存观测
└── cache-reset.json             # 仅显式启用 reset 时
```

逐实例 metrics 原文也保存在对应结果目录。失败可能发生在产物生成之前，缺失项须结合失败阶段解释；资源释放结论位于任务 status.json 和 .runtime/finalization.json，不以结果目录存在证明释放成功。

成功和失败都保留日志。验证以退出码、原生请求明细、请求数量、错误与 token 长度以及原生日志为依据，不把进程正常退出直接当成测试通过。

通过已冻结 Skill 的只读观察器跟踪当前及后续 attempt 新建的日志：

```sh
python3 /path/to/skill/scripts/watch_logs.py --run-dir /work/JOB --kind test
```

观察器跟踪当前 `results/*/aisbench.log`、原生 outputs 下的 `.out`/`.log`，兼容历史任务的 benchmark/runner 日志。每轮按文件大小快照分块读完积压，新文件每 5 秒发现一次；service 模式读取 `logs/`。Ctrl+C 只关闭观察，不停止任务。多节点按日志实际所在节点运行；任务数据与日志保存在仓库外。

当前 FULL 的生成配置由 validate_case.py 验证，validated.json 包含 formal 请求数、token 范围、零输出数和日志路径，不含 config 字段。case 提供 aisbench_config 时走原生分组校验，结果包含 config=native 和 request_groups；两种结果结构不同，应结合 job_spec 与冻结配置识别输入来源。数据集摘要采用分块 SHA256，不将整文件读入内存；行字段检查仍仅读取请求所需切片。


生成配置仅使用 CLI `--work-dir "$OUT/outputs"` 指定运行目录，不再重复写 config.work_dir；原生配置保留原文，CLI 覆盖其 work_dir。每个 case/attempt 的目录独立且拒绝复用。单模型、单数据集的生成配置要求恰好一份正式 details；原生多模型/数据集按已解析分组核验。原生预热和正式请求经同一次 CLI 执行，V1–V6 已有实际验证，不代表任意 AISBench 版本具有相同布局。

成功请求允许 `output_tokens=0`（如立即结束），仍要求 success=true、无 error_info、合法整数长度和正确请求数；生成配置的 validated.formal.zero_output_requests 单独计数。零输出不能据此解释为有有效输出吞吐或 TPOT，失败请求、负数和超过上限的输出仍拒绝。

缓存重置统一核验 cache-reset.json 总状态及各实例状态，不再依据原始 reset 文件名字典序选择“最新”。
