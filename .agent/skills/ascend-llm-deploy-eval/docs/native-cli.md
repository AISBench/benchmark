# 标准 AISBench CLI 适配

[返回快速开始](../README.md)

测试容器内只需预先安装可运行服务化性能测试的 AISBench 及其常规依赖，不要求 AISBench 源码目录或额外插件。Skill 不安装软件、不下载或生成数据集，不打包数据、模型或 PD 代理。

## 数据集绑定

用户提供服务器上的 JSONL 或 CSV 文件绝对路径。文件通过同路径挂载供测试容器读取；已被现有挂载覆盖时复用该挂载，否则增加只读挂载。文件不会复制到 Skill、冻结任务包或 Git 仓库。

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

数据集内容决定输入长度、前缀和顺序。Skill 不改写、截断、拼接或随机生成输入，也不承诺前缀命中率。`input_tokens` 为可选的严格长度校验值；变长数据集应省略。`output_tokens` 表示单请求最大输出长度，允许提前结束；数据集内逐行输出长度不应超过此值。

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
attempt-NN/
├── aisbench-config.py
├── dataset.json                # 仅路径和字段映射，不包含数据内容
├── benchmark-command.json
├── execution.json              # CLI 退出码
├── benchmark.log               # 主进程 stdout/stderr
├── outputs/<timestamp>/
│   ├── configs/
│   ├── logs/infer/<model>/<dataset>.out
│   ├── performances/
│   └── summary/
├── log-index.json              # 原生日志、主进程日志及工作目录索引
├── validated.json
└── cleanup.json
```

成功和失败都保留日志。验证以退出码、原生请求明细、请求数量、错误与 token 长度以及原生日志为依据，不把进程正常退出直接当成测试通过。

通过已冻结 Skill 的只读观察器跟踪当前及后续 attempt 新建的日志：

```sh
python3 /work/JOB/common/skill-snapshot/scripts/watch_logs.py --run-dir /work/JOB --kind test
```

观察器同时跟踪 `benchmark.log`、`runner.log` 和 AISBench 原生 `.out`/`.log` 文件。Ctrl+C 只关闭观察，不停止任务。多节点按日志实际所在节点运行；任务数据与日志保存在仓库外。
