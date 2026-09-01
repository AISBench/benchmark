# Scenario 配置参数说明

本文逐字段解释 [scenario.example.json](scenario.example.json)，并补充示例中没有展开的可选参数和模式。

## 1. 基本规则

- 配置是严格 JSON，不能写注释、尾随逗号或未支持的字段；
- 相对路径以 Scenario JSON 所在目录为基准；
- 比例使用 `0.0～1.0`，例如 `0.6` 表示 60%；
- token 长度以 `tokenizer.path` 加载出的 tokenizer 编码结果为准；
- 示例中的 `{}` 和 `[]` 只是 JSON 对象与列表，不是参数。
- 所有示例字段都可省略；源码默认值与当前 `scenario.example.json` 一致。未知字段仍会被严格拒绝。

## 2. 顶层字段

| 字段 | 必填 | 作用 |
|---|---:|---|
| `schema_version` | 否 | 配置契约版本，默认 `"1.0"`，当前只能为该值。 |
| `run` | 否 | 运行标识、随机种子和产物目录。 |
| `tokenizer` | 否 | token 计算和 Block 对齐。 |
| `corpus` | 否 | GSM8K 来源及样本选择方式。 |
| `requests` | 否 | 正式请求数量和输入/输出长度。 |
| `output` | 否 | 控制最小 `requests.jsonl` 是否输出第三个长度字段。 |
| `prefix_cache` | 否 | 缓存模式、目标命中率、组和顺序。 |
| `service` | 否 | vLLM 推理、指标、reset、多 DP 路由与超时配置；离线阶段也使用 `dp_size` 建模。 |
| `validation` | 否 | 偏差告警阈值。 |
| `aisbench` | 否 | `run` 使用的 AISBench 配置、工作目录和额外命令行参数。 |

嵌套对象同样采用严格字段白名单：

- `corpus.selection`：`mode`、`values`、`indices`、`question_sha256`；
- `requests.input_length`：`mode`、`value`、`values`、`ranges`、`min`、`max`、`mean`、`std`、`path`；每个 range 项只允许 `min`、`max`、`count`；
- `requests.output_length`：`mode`、`value`、`min`、`max`、`mean`、`std`、`path`；
- `output`：`output_key`；
- `prefix_cache.groups`：`count`、`assignment`、`overrides`；assignment 内允许 `mode`、`exponent`、`weights`；
- `prefix_cache.order`（即 `order` 对象）：`strategy`；
- `aisbench`：`config`、`work_dir`、`extra_args`、`dataset`、`model`；
- `aisbench.dataset`：`abbr`、`input_columns`、`output_column`、`prompt_template`、`pred_role`；
- `aisbench.model`：`abbr`、`stream`、`max_out_len`、`retry`、`batch_size`、`generation_kwargs`；
- 其余对象的允许字段由下文对应字段表完整列出。

## 3. `schema_version`

```json
"schema_version": "1.0"
```

用于防止插件按错误结构解释配置。当前其他版本会直接失败。

## 4. `run`

```json
"run": {
  "run_id": "gsm8k-prefix-cache-60",
  "random_seed": 42,
  "output_dir": "./outputs/gsm8k-prefix-cache-60"
}
```

| 字段 | 必填 | 默认值 | 作用 |
|---|---:|---|---|
| `run_id` | 否 | `"gsm8k-prefix-cache-60"` | 基础运行 ID。执行时追加时间戳，并作为四类产物的文件名前缀。prepare/run 可复用最近一次有效任务时间戳。 |
| `random_seed` | 否 | `42` | 控制 GSM8K 随机选择、长度采样、组分配、顺序和唯一 seed。相同输入与配置应生成相同内容。 |
| `output_dir` | 否 | `"./outputs/gsm8k-prefix-cache-60"` | 基础产物目录。执行时在最后一级目录名后追加与 run ID 相同的时间戳；prepare/run 可复用匹配 Manifest 所在目录。 |
| `overwrite` | 否 | `false` | 兼容保留字段。`prepare` 默认拒绝覆盖同名产物，重建使用 `prepare --overwrite`。 |

假设执行时间戳为 `20260825_123456`，示例会生成：

```text
outputs/gsm8k-prefix-cache-60_20260825_123456/
├── log/gsm8k-prefix-cache-60_20260825_123456.prepare.log
└── result/
    ├── gsm8k-prefix-cache-60_20260825_123456.full.jsonl
    ├── gsm8k-prefix-cache-60_20260825_123456.requests.jsonl
    ├── gsm8k-prefix-cache-60_20260825_123456.manifest.json
    └── gsm8k-prefix-cache-60_20260825_123456.analysis.json
```

时间戳采用 `_YYYYMMDD_HHMMSS`。`inspect` 每次创建新时间戳并在时间戳目录的 `result/` 下写轻量 Manifest，不再创建 `<output_dir>.inspect.json`。prepare/run 会扫描时间戳 Manifest，并校验状态、时间戳化 run/output 和 Scenario SHA-256；匹配时复用，否则创建新时间戳。

## 5. `tokenizer`

```json
"tokenizer": {
  "path": "/home/weights/Qwen3.6-27B",
  "block_size": 16,
  "trust_remote_code": false
}
```

| 字段 | 必填 | 默认值 | 作用 |
|---|---:|---|---|
| `path` | 否 | `"/home/weights/Qwen3.6-27B"` | 传给 `AutoTokenizer.from_pretrained` 的本地目录或 Hugging Face 标识。必须与 vLLM 服务端 tokenizer 一致。 |
| `block_size` | 否 | `16` | Prefix Cache Block 的 token 数。公共前缀和 seed 按它对齐，必须与服务端实际值一致。 |
| `revision` | 否 | `null` | tokenizer 的分支、tag 或 commit，用于固定版本。 |
| `trust_remote_code` | 否 | `false` | 是否执行模型仓库的自定义 tokenizer 代码，只应对可信仓库启用。 |

若 `block_size=16`、`seed_blocks=1`，每条请求会在公共前缀和自然后缀之间插入 16-token 唯一 seed。

## 6. `corpus`

```json
"corpus": {
  "path": "./GSM8K.jsonl",
  "field": "question",
  "selection": {"mode": "random"}
}
```

| 字段 | 必填 | 默认值 | 作用 |
|---|---:|---|---|
| `path` | 否 | `"./GSM8K.jsonl"` | GSM8K JSONL 路径，每个非空行必须是 JSON 对象。 |
| `field` | 否 | `"question"` | 读取自然语言问题的字段，只使用该字段，不拼接标准答案。 |
| `selection` | 否 | `{"mode":"random"}` | 为 canonical 前缀和自然后缀选择样本。 |

问题文本会先去除首尾空白，并把连续空白折叠成一个空格。`question_sha256` 基于规范化后的 UTF-8 文本。

### 6.1 `selection.mode=random`

```json
"selection": {"mode": "random"}
```

按 `random_seed` 确定性打乱。所需数量超过语料行数时开始新的打乱周期。

### 6.2 `selection.mode=indices`

```json
"selection": {"mode": "indices", "values": [0, 15, 72]}
```

- 使用零基行号，`0` 是第一行；
- `values` 也可写成 `indices`；
- 列表不足时循环复用；
- 任一行号不存在会失败。

### 6.3 `selection.mode=question_sha256`

```json
"selection": {
  "mode": "question_sha256",
  "values": ["规范化问题文本的64位SHA-256"]
}
```

`values` 也可写成 `question_sha256`。每个哈希必须唯一匹配一条语料；零匹配或多匹配都会失败。

### 6.4 `selection.mode=mixed`

```json
"selection": {
  "mode": "mixed",
  "indices": [0, 15],
  "question_sha256": ["某个问题的SHA-256"]
}
```

先加入行号样本，再加入哈希样本，适合同时固定位置和内容身份。
如果两类列表合计样本数小于实际需要数量，插件会按合并后的顺序循环复用；如需避免复用，请提供足够多的指定样本。
`indices` 与 `question_sha256` 不能同时为空，否则报 `specified GSM8K selection is empty`。

## 7. `requests`

```json
"requests": {
  "count": 100,
  "input_length": {"mode": "fixed", "value": 1024},
  "output_length": {"mode": "fixed", "value": 32}
}
```

### 7.1 `count`

正式请求总数，默认 `100`，必须是正整数。warmup 请求不计入该数量，也不写入 requests JSONL。

### 7.2 `input_length`

定义每条正式请求的目标输入 token 总数：

```text
公共前缀 + 全局唯一 seed + GSM8K 自然后缀
```

整个字段省略时默认 `{"mode":"fixed","value":1024}`；fixed 模式省略 `value` 时也默认 1024。

#### 固定长度

```json
"input_length": {"mode": "fixed", "value": 1024}
```

所有请求都是 1024 token，`value` 必须为正整数。

#### 闭区间采样

```json
"input_length": {
  "mode": "range",
  "ranges": [
    {"min": 512, "max": 1024, "count": 80},
    {"min": 2048, "max": 4096, "count": 20}
  ]
}
```

- `min`、`max` 均包含；
- 每个 `count` 表示该区间生成的请求数；
- 所有 `count` 之和必须等于 `requests.count`；
- 采样由 `random_seed` 决定。

#### 显式长度列表

```json
"input_length": {"mode": "explicit", "values": [512, 768, 1024, 2048]}
```

`values` 必须全部是正整数，元素个数必须等于对应范围内的请求数。全局配置时等于 `requests.count`；组级覆盖时等于该组实际请求数。

#### 截断正态分布

```json
"input_length": {
  "mode": "truncated_normal",
  "min": 512,
  "max": 2048,
  "mean": 1024,
  "std": 256
}
```

只接受 `[min,max]` 内的整数采样；`mean` 默认取区间中点，`std` 默认按区间宽度推导且显式值必须大于 0。相同 `random_seed` 产生相同长度序列。

#### CSV 指定

```json
"input_length": {"mode": "csv", "path": "./input_lengths.csv"}
```

CSV 行数必须等于 `requests.count`，并包含以下任一正整数列：

- `input_prompt_tokens`；
- `content_tokens`；
- `input_tokens`。

### 7.3 `output_length`

该值写入 requests JSONL 的 `max_tokens`。

整个字段省略时默认 `{"mode":"fixed","value":32}`；fixed 模式省略 `value` 时也默认 32。

#### 固定值

```json
"output_length": {"mode": "fixed", "value": 32}
```

`value` 必须为正整数。

#### 均匀分布

```json
"output_length": {"mode": "uniform", "min": 16, "max": 64}
```

`min`、`max` 必须是正整数且 `max >= min`；在包含上下界的整数区间均匀采样。

#### 截断正态分布

```json
"output_length": {
  "mode": "truncated_normal",
  "min": 16,
  "max": 128,
  "mean": 64,
  "std": 16
}
```

- 只保留 `[min,max]` 内的整数；
- `min`、`max` 必须是正整数且 `max >= min`；
- `mean` 省略时取区间中点；
- `std` 省略时按区间宽度推导，显式值必须大于 0；
- `min=max` 时直接返回固定值。

#### CSV 指定

```json
"output_length": {"mode": "csv", "path": "./output_lengths.csv"}
```

CSV 必须包含正整数 `output_tokens` 列，行数等于 `requests.count`。

### 7.3 顶层 `output`

```json
"output": {"output_key": null}
```

| 字段 | 必填 | 默认值 | 作用 |
|---|---:|---|---|
| `output_key` | 否 | `null` | 控制 `requests.jsonl` 的可选第三字段。允许 `null`、`"max_tokens"`、`"output_tokens"`。 |

- `null`：只写 `question`、`answer`；
- `"max_tokens"`：第三字段名为 `max_tokens`；
- `"output_tokens"`：第三字段名为 `output_tokens`，值仍取本请求的最大输出 token 数。

该语义与参考脚本 `extract_qa.py --output-key` 一致。`full.jsonl.max_tokens` 永远保留，AISBench 在线执行也从 full 审计行读取输出长度，所以省略第三字段不会影响压测。

## 8. `prefix_cache`

```json
"prefix_cache": {
  "mode": "warmup",
  "target_hit_rate": 0.6,
  "seed_blocks": 1,
  "minimum_non_shared_length": 16,
  "groups": {
    "count": 1,
    "assignment": {"mode": "uniform"}
  },
  "order": {"strategy": "interleave"}
}
```

### 8.1 `mode`

- `cold`：正式请求按 `(Prefix Group, DP rank)` lane 路由，理论命中率按 lane 从零水位模拟；
- `warmup`：为每个 `Prefix Group × DP rank` 生成预热计划（写入 Manifest 的 `warmup.plan`），正式请求本身不固定 DP。

`prepare` 只生成数据与预热计划；`run` 对每个 Prefix Group、每个 DP rank 执行计划。warmup 请求在正式 baseline 之前完成，不进入正式请求数、AISBench 性能数据、理论分母或实际指标增量。
省略时默认 `warmup`。

### 8.2 `target_hit_rate`

期望的全局 token 加权命中率，范围 `[0,1]`。它是求解器的主目标，不等于简单地把每条请求的固定百分比设成前缀。
省略时默认 `0.6`。

求解会考虑 Block 对齐、请求顺序、Prefix Group、水位和 cold DP 路由。最终命中 token 总量优先匹配最近可达目标；在总量相同的解中，优先让累计理论命中率低超调、少回落并逐步贴近目标。warmup 会按累计输入比例均衡分配，cold 会按 `(Prefix Group, DP rank)` 独立水位搜索。目标不可精确达到时，采用最接近的可达值并记录 requested/effective/theoretical 及原因。

累计命中率严格单调不是配置契约：后置 lane 的首次 cold miss、请求容量不足或 Block 离散性可能造成不可避免的小幅波动。求解器会最小化这些波动，而不会牺牲最终 target-driven 精度。

### 8.3 `seed_blocks`

唯一 seed 的 Block 数，默认 `1`，必须为正整数：

```text
seed token 数 = seed_blocks × tokenizer.block_size
```

seed 位于公共前缀和自然后缀之间，所有正式请求全局唯一，防止请求在公共前缀之后继续意外共享。输入长度必须能容纳 seed。

插件在加载 Scenario 时就会检查 fixed/explicit/range/truncated_normal/CSV 输入长度的最小值是否能容纳非共享区；不足时会在生成数据前直接报配置错误。

### 8.4 `minimum_non_shared_length`

每条正式请求至少预留多少个非共享 token，默认等于 `seed_blocks × block_size`，并且不能小于唯一 seed 长度。

```text
公共前缀最大长度 = 按 Block 向下对齐(input_length - minimum_non_shared_length)
```

当该值大于 seed 长度时，多出的空间由 GSM8K 自然后缀填充。它用于保证公共前缀之后不仅有全局唯一 seed，还能保留指定规模的自然差异内容。

### 8.5 `groups.count`

Prefix Group 数量，默认 `1`。插件生成 `group-0`、`group-1` 等 ID。每个组独立生成 canonical 前缀、维护水位、统计理论命中率，并在 warmup 时逐 DP 预热。

### 8.6 `groups.assignment`

整个 `assignment` 省略时默认 `{"mode":"uniform"}`。

均匀分配：

```json
"assignment": {"mode": "uniform"}
```

请求尽量平均分组，余数按稳定组序分配。

Zipf 分配：

```json
"assignment": {"mode": "zipf", "exponent": 1.0}
```

热度与 `1/rank^exponent` 成正比；`exponent` 必须大于 0，越大越集中于热点组。

显式权重：

```json
"assignment": {
  "mode": "weights",
  "weights": [0.5, 0.3, 0.15, 0.05]
}
```

权重数量必须等于 `groups.count`，不能为负且总和大于 0；无需预先归一化。

### 8.7 `groups.overrides`

可按组覆盖全局设置；省略时默认 `{}`：

```json
"groups": {
  "count": 4,
  "assignment": {"mode": "uniform"},
  "overrides": {
    "group-0": {
      "input_length": {"mode": "fixed", "value": 2048},
      "output_length": {"mode": "fixed", "value": 64},
      "corpus_selection": {"mode": "indices", "values": [0, 1, 2]}
    }
  }
}
```

- ID 必须是有效的 `group-0` 到 `group-(count-1)`；
- `input_length`、`output_length` 支持对应的全部全局模式；
- `corpus_selection` 支持 random/indices/question_sha256/mixed；
- 组级 range/CSV 生成数量必须等于实际分到该组的请求数。

### 8.8 `order.strategy`

- `sequential`：保持目标分配阶段顺序；
- `within_group_shuffle`：各组内部打乱，再按组输出；
- `interleave`：各组轮转交错，默认，适合多租户流量；
- `global_shuffle`：全局确定性打乱。
- `input_len_asc`：每个 Prefix Group 内按输入长度从短到长排列，再按组轮转交错；相同长度保持原始稳定顺序。

理论水位总是按最终发送顺序重新模拟。若要模拟无预热时 Cache 从短请求到长请求逐步建立，配置 `prefix_cache.mode="cold"` 与 `order.strategy="input_len_asc"`。prepare 会按该顺序生成并落盘，run 阶段即使存在并发任务，也会按每个 `(group_id, dp_rank)` lane 的 `lane_sequence` 严格串行发送；前一条完成后才放行下一条。不同 Group/DP 的 Cache 独立，可彼此并行。

## 9. `service`

```json
"service": {
  "inference_url": "http://127.0.0.1:8000/v1/completions",
  "metrics_url": "http://127.0.0.1:8000/metrics",
  "reset_url": "http://127.0.0.1:8000/reset_prefix_cache",
  "model": "model-name",
  "dp_size": 2,
  "assume_empty_cache": false
}
```

| 字段 | 必填 | 默认值 | 作用 |
|---|---:|---|---|
| `inference_url` | 否 | `"http://127.0.0.1:8000/v1/completions"` | `run` 的 vLLM Completions API；probe、warmup 和正式请求均使用。 |
| `metrics_url` | 否 | `"http://127.0.0.1:8000/metrics"` | `run` 采集 baseline/after 的 Prometheus 地址。 |
| `reset_url` | 否 | `"http://127.0.0.1:8000/reset_prefix_cache"` | 正式统计前清空 Prefix Cache；为空或失败时仅在显式启用 `assume_empty_cache` 后继续。 |
| `model` | 否 | `"model-name"` | completion 请求体中的模型名。不会写入最小 `requests.jsonl`。 |
| `dp_size` | 否 | `2` | 单入口内部 DP rank 数。离线用于 cold 路由/warmup 计划；在线用于逐 DP 探活、预热和指标完整性校验。 |
| `assume_empty_cache` | 否 | `false` | reset 不可用时是否由用户显式保证缓存为空；启用后记录 `ASSUME_EMPTY_CACHE`。 |
| `engine_label_map` | 否 | `{}` | Prometheus `engine` 标签到 DP rank 的显式映射；未配置时尝试解析标签尾部数字。 |
| `timeout_seconds` | 否 | `30` | probe、reset、warmup、metrics HTTP 请求超时秒数。 |
| `api_key` | 否 | `""` | 推理 API Bearer Token。Manifest 不保存明文，只记录是否配置；Scenario 文件本身仍需限制权限。 |

> `inspect`、`prepare`、`validate` 不访问服务；`run` 消费全部在线字段。当前支持一个 HTTP 入口及其内部多个 DP，不支持多个独立 vLLM 实例。

## 10. `validation`

```json
"validation": {
  "target_warning_pp": 1.0,
  "actual_warning_pp": 5.0
}
```

| 字段 | 默认值 | 作用 |
|---|---:|---|
| `target_warning_pp` | `1.0` | 理论值与请求目标相差超过多少百分点时记录 `TARGET_DEVIATION`。 |
| `actual_warning_pp` | `5.0` | `run`/`analyze` 的实际值与理论值相差超过多少百分点时记录 `ACTUAL_DEVIATION`。 |

单位是百分点（pp），不是相对百分比。例如 60% 与 58.5% 相差 1.5 pp。两种偏差始终只 warning，不改变原本成功的退出码。

分析产物同时记录带符号偏差、绝对偏差、目标是否在全局可达范围内，以及 `PASS`/`PASS_WITH_WARNING` 展示状态。该状态只用于展示，不控制退出码。

## 11. `aisbench`

```json
"aisbench": {
  "config": "./plugins/prefix_cache/config_examples/prefix_cache_perf.py",
  "work_dir": "./outputs/aisbench-prefix-cache-60",
  "extra_args": [],
  "dataset": {
    "abbr": null,
    "input_columns": ["question", "max_out_len"],
    "output_column": "answer",
    "prompt_template": "{question}",
    "pred_role": "BOT"
  },
  "model": {
    "abbr": null,
    "stream": true,
    "max_out_len": 1,
    "retry": 2,
    "batch_size": 1,
    "generation_kwargs": {
      "temperature": 0,
      "ignore_eos": true
    }
  }
}
```

| 字段 | 必填 | 默认值 | 当前用途 |
|---|---:|---|---|
| `config` | 否 | `"./plugins/prefix_cache/config_examples/prefix_cache_perf.py"` | `run` 加载并渲染的 AISBench Python 配置；该默认值适用于按 README 把 Scenario 复制到仓库根目录的工作流，可用 CLI `--config` 临时覆盖。 |
| `work_dir` | 否 | `"./outputs/aisbench-prefix-cache-60"` | AISBench 正式压测结果目录，相对 Scenario 所在目录解析。 |
| `extra_args` | 否 | `[]` | 追加到 AISBench perf 子进程命令后的字符串参数列表。 |
| `dataset` | 否 | 见下表 | AISBench Dataset reader、prompt 和评测展示配置。 |
| `model` | 否 | 见下表 | AISBench API Model 的流式、重试、并发和请求参数。 |

`aisbench.dataset`：

| 字段 | 默认值 | 作用与约束 |
|---|---|---|
| `abbr` | `null` | Dataset 展示名；为 null 时使用带时间戳的 `run_id`，也可配置非空字符串。 |
| `input_columns` | `["question", "max_out_len"]` | DatasetReader 输入列。为保证 prompt 与逐请求输出长度契约，当前必须保持该值。 |
| `output_column` | `"answer"` | 参考答案列；当前必须保持 `answer`。 |
| `prompt_template` | `"{question}"` | 正式发送模板；当前必须保持原值，否则实际 token 与 prepare 理论审计不一致。 |
| `pred_role` | `"BOT"` | AISBench 评测结果中的预测角色名，可配置为任意非空字符串。 |

`aisbench.model`：

| 字段 | 默认值 | 作用与约束 |
|---|---|---|
| `abbr` | `null` | Model 展示名；为 null 时使用 `<run_id>-vllm`，也可配置非空字符串。 |
| `stream` | `true` | 是否使用 SSE 流式请求。设为 false 后仍可测 Prefix Cache 命中率，但不能按 chunk 采集完整 TTFT/TPOT/ITL。 |
| `max_out_len` | `1` | AISBench Model 的兜底最大输出长度。正式请求优先使用 full.jsonl 中每行的 `max_tokens`。 |
| `retry` | `2` | API 失败重试次数，必须是非负整数。重试请求也可能进入服务端累计指标，应结合服务稳定性谨慎调整。 |
| `batch_size` | `1` | AISBench API Model 最大并发基值，必须是正整数；cold 模式同一 lane 仍由插件严格串行。 |
| `generation_kwargs` | `{"temperature":0,"ignore_eos":true}` | 合并到 vLLM 请求的生成参数对象，可增加 `top_p` 等当前服务支持的 JSON 参数。 |

整个 `aisbench` 段及其 `dataset`、`model` 子段都可省略，旧 Scenario 会补齐与当前行为一致的默认值。`config`、`work_dir` 必须是非空字符串，`extra_args` 必须是字符串列表。离线命令不消费这些在线参数，`run` 才会渲染配置并启动 AISBench。Python 类型、Manifest 工件路径、DP 路由和 Prefix Cache Inferencer 属于插件内部不变量，不允许在 Scenario 中替换。

## 12. 原示例最终表示的场景

- 生成 100 条正式请求；
- 输入长度固定为 1024 token；
- 每条最多输出 32 token；
- 创建 1 个 uniform 组；
- 目标全局命中率为 60%；
- 使用一个 16-token Block 作为全局唯一 seed；
- 每条请求至少保留 16 token 非共享区；
- 请求按组交错排列；
- 使用 warmup 模式；
- 单个 vLLM HTTP 入口内部有 2 个 DP rank（cold 路由 / warmup 计划使用）；
- 该组分别在 DP 0、DP 1 生成预热计划，共 2 条不进入正式统计的 warmup 请求；
- 理论/目标超过 1 pp、实际/理论超过 5 pp 时均只告警；
- `run` 使用示例 `prefix_cache_perf.py` 启动 AISBench，工作目录为 `./outputs/aisbench-prefix-cache-60`。

## 13. 建议检查顺序

```bash
ais-bench-prefix-cache inspect --scenario ./scenario.json
ais-bench-prefix-cache prepare --scenario ./scenario.json
ais-bench-prefix-cache validate --manifest <manifest路径>
ais-bench-prefix-cache run --scenario ./scenario.json
```

重点检查：

- `requested_target_hit_rate`：请求目标；
- `effective_target_hit_rate`：求解器选择的可达目标；
- `theoretical_hit_rate`：按最终顺序模拟的理论值；
- `reachable_min/max`：当前长度、Block、分组和路由下的范围；
- `target_reachable`：目标是否落在全局最大/最小可达区间；
- `groups`：请求分布与 canonical 前缀；
- `warmup.plan`：是否覆盖每个 Prefix Group × DP rank；
- `warnings`：目标偏差或目标不可达。

Manifest 还会记录输入/输出长度的 min/max/mean/P50/P90/P95/P99 与分桶计数、各组 reachable min/max、每条请求的确定性 `request_random_seed`，以及唯一差异块的碰撞检查状态。

## 14. CLI 行为和返回字段

### 14.1 `inspect`

`inspect` 会加载 tokenizer 和 GSM8K，在临时目录复用完整 prepare 流程计算可达范围，但不发送请求，也不保留 full/requests/analysis 三类正式数据产物；只在正式 `result/` 目录写入轻量 Manifest。

- 每次执行生成新时间戳；
- 日志写入 `output_dir_时间戳/log/<run_id_时间戳>.inspect.log`；
- 成功后写入 `output_dir_时间戳/result/<run_id_时间戳>.manifest.json`，`status="inspected"`；
- stdout JSON 包含 `log` 和 `manifest` 路径。

### 14.2 `prepare`

`prepare` 优先复用最近一次与当前 Scenario 匹配的 inspect Manifest 时间戳；没有匹配项时生成新时间戳。匹配的轻量 Manifest 会原位升级为正式 `status="prepared"` Manifest。生成 prompt 时进度条写入 stderr，每完成一条 prompt 增加 1；最后一行 stdout JSON 固定包含：

| 字段 | 含义 |
|---|---|
| `full` | full JSONL 路径。 |
| `requests` | 最小 requests JSONL 路径。 |
| `manifest` | Manifest JSON 路径。 |
| `analysis` | 理论分析 JSON 路径。 |
| `log` | prepare 日志路径；只有日志文件成功解析和创建时出现。 |

`--overwrite` 只允许覆盖当前时间戳目录内上述四个固定产物，不会删除整个输出目录。

### 14.3 `validate`

`validate` 不生成新数据，检查行数、字段集合、顺序对应关系及 full/requests SHA-256。stdout 固定返回：

| 字段 | 含义 |
|---|---|
| `ok` | 校验是否通过；成功时为 `true`。 |
| `rows` | 通过校验的正式请求行数。 |
| `run_id` | Manifest 中的运行 ID。 |

validate 日志写入 Manifest 对应时间戳目录的 `log/<run_id>.validate.log`，但当前返回 JSON 不包含 `log` 字段。

### 14.4 `run`

`run --scenario` 复用任务时间戳；若该目录还没有工件则自动 prepare。随后依次执行逐 DP probe、reset、可选的每组每 DP warmup、baseline、AISBench perf、after 和指标差分。`--config <path>` 可仅覆盖本次 AISBench 配置。stdout 返回更新后的完整 analysis，日志写入 `log/<run_id>.run.log`。

### 14.5 `analyze`

`analyze --manifest <path> --baseline <before.prom> --after <after.prom>` 不连接服务，只离线复算实际命中率并写回 analysis。stdout 返回完整 analysis，日志写入 `log/<run_id>.analyze.log`。

正常成功退出码为 `0`；Scenario、生成或产物校验错误返回 `2`。目标不可达和命中率偏差始终只是 warning，不改变成功退出码。

## 15. 请求产物字段

一次正式 `prepare` 后，`result/` 中有四个文件：

| 文件 | 格式 | 用途 |
|---|---|---|
| `<run_id>.requests.jsonl` | JSONL | 提供给 AISBench Dataset 的最小请求数据；一行对应一条正式请求。 |
| `<run_id>.full.jsonl` | JSONL | 与 requests 一一对应的完整审计数据，用于复现构造过程和理论命中率。 |
| `<run_id>.manifest.json` | JSON | 本次数据集的配置快照、语料/tokenizer 指纹、统计摘要和产物索引。 |
| `<run_id>.analysis.json` | JSON | prepare 阶段的理论结果；执行 `run` 或 `analyze` 后在原文件中追加实际指标和校验结果。 |

JSONL 文件没有包裹数组，必须逐行解析；JSON 文件是一个完整对象。Manifest 的 `artifacts` 只索引 full、requests、analysis，不索引 Manifest 自身。

### 15.1 `<run_id>.requests.jsonl`

每行固定以 `question`、`answer` 开头，并由 `output.output_key` 决定是否追加第三字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `question` | string | 最终完整 prompt。 |
| `answer` | string | AISBench 兼容占位值，当前固定为 `"none"`。 |
| `max_tokens` | integer | 可选；`output_key="max_tokens"` 时输出，值来自 `requests.output_length` 或组级覆盖。 |
| `output_tokens` | integer | 可选；`output_key="output_tokens"` 时输出，值与内部 `max_tokens` 相同。 |

默认 `output_key=null`，因此每行只有 `question`、`answer`。两个可选长度键不会同时出现。

### 15.2 `<run_id>.full.jsonl`

每行固定包含 26 个审计字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `request_id` | string | 稳定请求 ID，例如 `request-00000000`。 |
| `sequence_index` | integer | 最终发送顺序中的零基序号，必须连续。 |
| `group_id` | string | 所属 Prefix Group。 |
| `occurrence_index_within_group` | integer | 该请求在组内的出现序号。 |
| `dp_rank` | integer/null | cold 模式的目标 DP rank；warmup 正式请求为 `null`。 |
| `lane_sequence` | integer/null | cold `(group_id, dp_rank)` lane 内序号；warmup 为 `null`。 |
| `target_input_tokens` | integer | 长度配置要求的输入 token 数。 |
| `actual_input_tokens` | integer | prompt 经 tokenizer 重编码后的实际 token 数。 |
| `max_tokens` | integer | 最大输出 token 数。 |
| `shared_prefix_tokens` | integer | 求解器为该请求选择的公共前缀长度。 |
| `seed_tokens` | integer | 全局唯一 seed 的 token 数。 |
| `natural_suffix_tokens` | integer | seed 后 GSM8K 自然后缀的 token 数。 |
| `question` | string | 最终完整 prompt。 |
| `answer` | string | 当前固定为 `"none"`。 |
| `gsm_indices` | array[integer] | 本请求自然后缀使用的 GSM8K 零基行号。 |
| `gsm_hashes` | array[string] | 对应规范化 GSM8K question 的 SHA-256。 |
| `canonical_prefix_sha256` | string | 所属组 canonical 前缀指纹。 |
| `seed_sha256` | string | 本请求唯一 seed token 序列指纹。 |
| `request_random_seed` | integer | 实际参与该请求 seed 构造的确定性随机种子。 |
| `watermark_before` | integer | 请求到达前所在缓存 lane 的理论水位。 |
| `theoretical_hit_tokens` | integer | 本请求理论命中 token 数。 |
| `watermark_after` | integer | 请求完成后的理论水位。 |
| `theoretical_hit_rate` | number | `theoretical_hit_tokens / actual_input_tokens`。 |
| `divergence_block_sha256` | string | 差异块指纹，当前等于 `seed_sha256`。 |
| `divergence_unique` | boolean | 差异块是否通过全局唯一性检查。 |
| `collision_status` | string | 碰撞检查状态，成功产物为 `"pass"`。 |

## 16. Manifest 完整字段

Manifest 顶层字段：

| 字段 | 含义 |
|---|---|
| `schema_version` | Manifest 契约版本，当前为 `"1.0"`。 |
| `plugin_version` | 生成产物的插件版本。 |
| `status` | Manifest 生命周期状态；inspect-only 为 `"inspected"`，正式数据产物为 `"prepared"`。 |
| `run_id` | 已追加执行时间戳的运行 ID。 |
| `scenario_path` | 原 Scenario 绝对路径。 |
| `scenario_sha256` | 原 Scenario 文件 SHA-256。 |
| `effective_config` | 补齐默认值、解析路径并追加时间戳后的有效配置。 |
| `effective_config_sha256` | 有效配置的规范化 JSON 指纹。 |
| `corpus_sha256` | GSM8K 文件 SHA-256。 |
| `tokenizer` | tokenizer 身份和 Block 信息。 |
| `requests` | 请求数量、总 token 和长度分布。 |
| `prefix_cache` | 目标、可达范围、理论值和验证结论。 |
| `groups` | 各 Prefix Group 的 canonical、来源和理论统计。 |
| `dp` | DP 数量与 cold 路由策略。 |
| `warmup` | warmup 开关和预热计划。 |
| `divergence` | 全局唯一差异块审计。 |
| `artifacts` | 产物路径、大小、行数和哈希。 |

### 16.1 `tokenizer`

| 字段 | 含义 |
|---|---|
| `path`、`revision` | tokenizer 来源和固定版本。 |
| `class` | 实际加载的 tokenizer Python 类。 |
| `vocab_size` | tokenizer 词表大小。 |
| `special_token_ids` | 特殊 token ID 列表。 |
| `block_size` | Prefix Cache Block token 数。 |
| `fingerprint_sha256` | path/revision/class/vocab/special IDs 的规范化指纹。 |

### 16.2 `effective_config`

`effective_config` 是 Scenario 补齐默认值、解析相对路径、追加任务时间戳后的最终配置快照。其值代表“本次产物实际使用了什么”，不一定与用户原始 JSON 的书写形式完全相同。各子字段与本文第 2～13 节的同名配置含义一致：

| 路径 | 含义 |
|---|---|
| `schema_version` | 生效的 Scenario 契约版本。 |
| `run.run_id`、`run.output_dir` | 已追加 `_YYYYMMDD_HHMMSS` 的实际运行 ID 和输出目录。 |
| `run.random_seed`、`run.overwrite` | 实际使用的全局随机种子和覆盖配置。 |
| `tokenizer.path`、`revision`、`trust_remote_code`、`block_size` | tokenizer 加载配置和 Prefix Cache Block 大小。 |
| `corpus.path`、`field`、`selection` | 语料绝对路径、问题字段及样本选择方式；`selection` 按所选 mode 包含 `indices`、`question_sha256` 等字段。 |
| `requests.count`、`input_length`、`output_length` | 正式请求数及输入/输出长度生成配置；长度对象内部字段随 fixed、distribution 或 empirical 模式变化。 |
| `prefix_cache.mode` | cold 或 warmup。 |
| `prefix_cache.target_hit_rate`、`minimum_non_shared_length`、`seed_blocks` | 目标 token 命中率、最小非共享长度和 seed Block 数。 |
| `prefix_cache.groups.count`、`assignment`、`groups` | 组数、组分配策略和逐组覆盖；组覆盖只落盘实际存在的键。 |
| `prefix_cache.order.strategy` | 正式请求排序策略。 |
| `service.inference_url`、`metrics_url`、`reset_url`、`model` | 在线推理、指标、缓存重置地址和模型名。 |
| `service.dp_size`、`engine_label_map` | DP 数及可选的 metrics engine label 到 DP rank 映射。 |
| `service.timeout_seconds`、`poll_interval_seconds` | HTTP 超时和正式压测期间 KV 指标轮询间隔。 |
| `service.assume_empty_cache` | 无法 reset 时是否按显式配置继续并告警。 |
| `service.api_key_configured` | 是否配置过 API key；仅保存布尔值，不保存密钥明文。 |
| `aisbench.config`、`work_dir`、`extra_args` | AISBench 模板、工作目录和附加 CLI 参数。 |
| `aisbench.dataset`、`model` | Dataset reader/prompt/评测角色与 API Model 流式、重试、并发、生成参数；完整子字段见第 11 节。 |
| `validation.target_warning_pp`、`actual_warning_pp` | 理论目标偏差和理论/实际偏差的告警阈值，单位为百分点。 |
| `output.output_key` | requests.jsonl 是否输出 `max_tokens` 或 `output_tokens`。 |

### 16.3 `requests`

- `count`：正式请求数；
- `total_input_tokens`：所有正式请求实际输入 token 总和；
- `input_length_summary`、`output_length_summary`：输入/输出长度摘要。

每个长度摘要字段含义如下：

| 字段 | 含义 |
|---|---|
| `min`、`max` | 所有请求中的最小值和最大值。 |
| `mean` | 算术平均值。 |
| `p50`、`p90`、`p95`、`p99` | 排序后使用线性插值得到的 50/90/95/99 分位数，因此可以是小数。 |
| `bins` | 最多十个非空分桶，按长度从小到大排列。 |
| `bins[].min`、`bins[].max` | 该桶内实际出现的最小/最大观测值，不是预设桶边界。 |
| `bins[].count` | 落入该桶的请求数。所有桶的 count 之和等于 `requests.count`。 |

固定桶宽为 `max(1, ceil((全局最大值 - 全局最小值 + 1) / 10))`。空桶不会写入，所以 `bins` 相邻项的数值区间可能不连续。

### 16.4 `prefix_cache`

| 字段 | 含义 |
|---|---|
| `mode` | `cold` 或 `warmup`。 |
| `requested_target_hit_rate` | Scenario 请求的目标命中率。 |
| `effective_target_hit_rate` | 求解器选择的最近可达目标。 |
| `theoretical_hit_rate` | 按最终顺序模拟得到的理论值。 |
| `reachable_min`、`reachable_max` | 当前约束下全局理论可达范围。 |
| `target_reachable` | 请求目标是否位于可达范围内。 |
| `minimum_non_shared_length` | 每条请求预留的最小非共享 token 数。 |
| `adjusted` | 求解目标是否因约束被调整。 |
| `reason` | 调整原因；无需调整时可为 `null`。 |
| `validation_status` | `PASS` 或 `PASS_WITH_WARNING`。 |
| `target_signed_difference_pp` | `theoretical - requested` 的带符号百分点差。 |
| `target_absolute_difference_pp` | 上述差值的绝对值。 |

### 16.5 `groups.<group_id>`

- `canonical_prefix_sha256`、`canonical_prefix_tokens`：canonical 前缀指纹和总 token 数；
- `max_shared_prefix_tokens`：该组正式请求使用的最大公共前缀长度；
- `gsm_indices`、`gsm_question_sha256`：canonical 前缀语料来源；
- `reachable_min`、`reachable_max`：该组理论可达范围；
- `theoretical_hit_rate`：该组 token 加权理论命中率。

### 16.6 `dp`、`warmup`、`divergence`

- `dp.size`：DP 数；`cold_route_strategy`：cold 时为 `"group_round_robin"`，warmup 时为 `null`；
- `warmup.enabled`：是否启用；`warmup.plan`：预热项列表；
- 每个 warmup 项包含 `request_id`、`group_id`、`dp_rank`、`prompt`、`input_tokens`、`shared_prefix_tokens`、`max_tokens`、`included_in_formal_statistics`；最后一个字段固定为 `false`；
- `divergence.strategy`：当前为 `"globally_unique_seed_block"`；
- `unique_request_blocks`、`request_count`、`collision_status`：唯一 seed 数、请求数和碰撞检查结论。

### 16.7 `artifacts` 和密钥处理

- `artifacts.full`、`artifacts.requests`：`name`、`path`、`rows`、`bytes`、`sha256`；
- `artifacts.analysis`：`name`、`path`、`bytes`、`sha256_at_prepare`。

其中：

| 字段 | 含义 |
|---|---|
| `name` | 文件名，不含父目录。 |
| `path` | prepare 时解析出的绝对路径。 |
| `rows` | JSONL 行数；仅 full 和 requests 存在。 |
| `bytes` | prepare 完成时的文件字节数。 |
| `sha256` | full/requests 的内容摘要，`validate` 用它检测数据是否被篡改或截断。 |
| `sha256_at_prepare` | analysis 在 prepare 阶段的内容摘要。run/analyze 会合法重写 analysis，因此它不是运行完成后 analysis 的当前摘要，也不用于 full/requests 一致性校验。 |

Manifest 不保存 `service.api_key` 明文；它会被替换为 `effective_config.service.api_key_configured` 布尔值。

## 17. `analysis.json` 完整字段

| 字段 | 含义 |
|---|---|
| `schema_version` | 分析契约版本。 |
| `run_id` | 已追加时间戳的运行 ID。 |
| `status` | prepare 为 `"prepared"`，run 完成为 `"complete"`，analyze 复算为 `"analyzed"`。 |
| `requested_target_hit_rate` | Scenario 请求目标。 |
| `effective_target_hit_rate` | 最近可达目标。 |
| `theoretical_hit_rate` | 最终顺序理论值。 |
| `target_difference_pp` | 当前等于目标绝对偏差。 |
| `target_signed_difference_pp` | `theoretical - requested` 的带符号百分点差。 |
| `target_absolute_difference_pp` | 目标绝对偏差。 |
| `validation` | 展示状态和可达性。 |
| `theory` | 全局、分组和分 DP 理论 token 统计。 |
| `warnings` | 目标不可达或偏差告警列表。 |
| `runtime` | run/analyze 的 baseline/after 分 DP 累计指标；run 还记录阶段、探活、reset/warmup 与 AISBench 退出码。 |
| `actual` | 分 DP 与全局正式增量 queries、hits、hit rate。 |
| `theory_actual_difference_pp` | 实际与理论的绝对百分点差。 |
| `theory_actual_signed_difference_pp` | `actual - theoretical` 的带符号百分点差。 |
| `theory_actual_absolute_difference_pp` | 实际与理论的绝对百分点差。 |

### 17.1 `validation` 和 `warnings`

| 字段 | 含义 |
|---|---|
| `validation.status` | 汇总状态；没有任何 warning 时为 `PASS`，否则为 `PASS_WITH_WARNING`。 |
| `validation.target_reachable` | 用户目标是否在当前场景的理论可达范围内。 |
| `validation.warning_only` | 固定为 `true`，表示偏差只告警。 |
| `validation.affects_exit_code` | 固定为 `false`，表示告警不改变成功退出码。 |
| `validation.actual_status` | run/analyze 后新增；理论/实际偏差未超过 `actual_warning_pp` 时为 `PASS`，否则为 `PASS_WITH_WARNING`。 |
| `warnings` | 告警对象数组；无告警时为空数组。 |

`theory` 包含 `input_tokens`、`hit_tokens`、`groups`、`dp`；每个组或 DP 值包含 `input_tokens`、`hit_tokens`、`hit_rate`。warmup 正式请求没有固定 `dp_rank`，所以 `theory.dp` 可以为空对象。

`warnings` 可能包含：

- `TARGET_UNREACHABLE`：`code`、`requested_target_hit_rate`、`reachable_min`、`reachable_max`；
- `TARGET_DEVIATION`：`code`、`difference_pp`。

`run` 和 `analyze` 在实际/理论绝对差超过 `actual_warning_pp` 时添加 `ACTUAL_DEVIATION`；该告警始终不改变成功退出码。

reset 无法执行但 `assume_empty_cache=true` 时还可能添加 `ASSUME_EMPTY_CACHE`，其中 `message` 说明是未配置 reset URL，还是 reset 请求失败后继续。

### 17.2 `theory`

| 字段 | 含义 |
|---|---|
| `theory.input_tokens` | 全部正式请求的输入 token 总数。 |
| `theory.hit_tokens` | 按缓存水位模型模拟的理论命中 token 总数。 |
| `theory.groups.<group_id>.input_tokens` | 该 Prefix Group 的输入 token 总数。 |
| `theory.groups.<group_id>.hit_tokens` | 该组的理论命中 token 总数。 |
| `theory.groups.<group_id>.hit_rate` | 该组 `hit_tokens / input_tokens`，是 token 加权命中率。 |
| `theory.dp.<rank>.input_tokens`、`hit_tokens`、`hit_rate` | cold 模式下各 DP lane 的同类统计。warmup 正式请求不固定 rank，因此 `theory.dp` 可以为空对象。 |

### 17.3 `runtime`

`runtime` 在 prepare-only 文件中不存在。在线 `run` 会写入完整运行过程；离线 `analyze` 只写 `metrics_baseline` 和 `metrics_after`。

| 字段 | 含义 |
|---|---|
| `runtime.phases` | 在线阶段顺序；通常为 `precheck`、`reset`、可选 `warmup`、`baseline`、`formal`、`after`。 |
| `runtime.precheck.ok` | 每个 DP 都能完成探针请求并解析必需指标时为 `true`。 |
| `runtime.precheck.ranks` | 指标中成功识别出的 DP rank 列表。 |
| `runtime.precheck.metric_names` | 本次实际匹配到的 queries/hits/KV Prometheus 指标名。 |
| `runtime.warmup[]` | 仅 warmup 模式存在，记录逐组逐 DP 的预热发送结果。 |
| `runtime.warmup[].group_id`、`dp_rank` | 被预热的 Prefix Group 和 DP rank。 |
| `runtime.warmup[].success` | 该预热请求是否成功；当前只有成功项会完成落盘。 |
| `runtime.warmup[].elapsed_seconds` | 单条预热请求耗时（秒）。 |
| `runtime.aisbench_exit_code` | AISBench perf 子进程退出码；成功为 0，非 0 时 run 直接报错。 |

`metrics_baseline` 与 `metrics_after` 结构相同：

| 字段 | 含义 |
|---|---|
| `runtime.metrics_baseline` | reset/可选 warmup 后、正式 AISBench 压测前抓取的指标基线。它用于隔离正式阶段，不能简单理解为“所有字段必须为 0”。 |
| `runtime.metrics_after` | 正式 AISBench 结束后抓取的指标快照。 |
| `*.metric_names.queries` | 实际采用的 Prefix Cache query token counter 名。 |
| `*.metric_names.hits` | 实际采用的 Prefix Cache hit token counter 名。 |
| `*.metric_names.kv` | 实际采用的 KV Cache usage gauge 名。 |
| `*.by_dp.<rank>.queries` | 截止快照时该 DP 累计查询的 Prefix Cache token 数。 |
| `*.by_dp.<rank>.hits` | 截止快照时该 DP 累计命中的 Prefix Cache token 数。 |
| `*.by_dp.<rank>.kv_cache_usage` | 快照时刻该 DP 的 KV Cache 使用比例；范围通常为 0～1，1 表示 100%，不是累计 counter。 |
| `*.raw_prometheus` | 本次 `/metrics` 返回的完整原始文本，用于追溯指标标签、类型和未被插件消费的其他指标。 |

正式值对 counter 使用差分：`actual queries = after queries - baseline queries`，`actual hits = after hits - baseline hits`。这样 precheck、warmup 或服务已有累计 counter 不会进入正式命中率。KV 使用率是瞬时 gauge，不做 after-baseline 相减；`actual.by_dp.*.kv_cache_usage` 直接取 after 快照。

`runtime.kv_cache_polling` 仅在线 run 存在，表示 AISBench 正式运行期间的周期采样：

| 字段 | 含义 |
|---|---|
| `interval_seconds` | 配置的轮询间隔，即 `service.poll_interval_seconds`。 |
| `count` | 成功抓取并解析的轮询次数。抓取失败的轮次会跳过，因此不保证等于运行时长除以间隔。 |
| `samples[].elapsed_seconds` | 相对 AISBench 子进程启动时刻的采样时间（秒，保留三位小数）。 |
| `samples[].by_dp.<rank>` | 该时刻该 DP 的 KV Cache 使用比例；指标缺失时可以是 `null`。 |
| `summary.count` | 参与汇总的采样对象数，与外层 count 相同。 |
| `summary.by_dp.<rank>.sample_count` | 该 DP 的有效非 null 样本数。 |
| `summary.by_dp.<rank>.avg` | 该 DP 所有有效瞬时比例的算术平均。 |
| `summary.by_dp.<rank>.peak` | 该 DP 所有有效瞬时比例的最大值。 |
| `summary.global_avg` | 所有 DP、所有有效采样值合并后的算术平均。 |
| `summary.global_peak` | 所有 DP、所有有效采样值中的单点最大值；不是同一时刻各 DP 的求和。 |

设 DP `r` 的有效轮询值为 `u(r,1)...u(r,n)`，则 `avg(r)=sum(u)/n`，`peak(r)=max(u)`。全局 avg/peak 对所有 `(DP, 采样时刻)` 的有效值做相同聚合。

### 17.4 `actual`

| 字段 | 含义 |
|---|---|
| `actual.by_dp.<rank>.queries`、`hits` | after-baseline 得到的正式 query/hit token 数。 |
| `actual.by_dp.<rank>.hit_rate` | `hits / queries`；queries 为 0 时是 `null`。 |
| `actual.by_dp.<rank>.kv_cache_usage` | after 快照时该 DP 的瞬时 KV 使用比例。 |
| `actual.by_dp.<rank>.kv_cache_usage_avg`、`kv_cache_usage_peak` | 正式运行期间轮询得到的该 DP 均值和峰值；仅在线 run 有轮询样本时有意义。 |
| `actual.global_queries`、`global_hits` | 所有 DP 正式 counter 增量之和。 |
| `actual.global_hit_rate` | `global_hits / global_queries`，即最终实际 token 命中率。 |
| `actual.global_kv_cache_usage_avg`、`global_kv_cache_usage_peak` | 对全部 DP 轮询样本聚合的全局均值和单点峰值。 |

注意：`kv_cache_usage`、`kv_cache_usage_avg` 和 `kv_cache_usage_peak` 的原始值都是 0～1 比例。例如 `0.0083` 表示约 `0.83%`，展示成百分数时需要乘以 100。

### 17.5 本次示例结果如何解读

`gsm8k-token2048-prefix-cache-70_8_20260901_014447.analysis.json` 中：

- baseline 每个 DP 为 `queries=8`、`hits=0`、`kv_cache_usage=0.0`。8 个 query token 来自 baseline 之前的能力探针；reset 清空 Prefix Cache，但服务的 Prometheus counter 没有归零。该值会被差分扣除，不会污染正式统计。
- after 每个 DP 为 `queries=10248`。扣除 baseline 的 8 后，两个 DP 的正式 queries 都是 10240；正式 hits 分别为 7296 和 7040，所以全局命中率为 `(7296+7040)/(10240+10240)=0.7`。
- 正式阶段成功轮询 81 次。DP0 的 avg/peak 为 `0.0030832/0.0083247`，DP1 为 `0.0030688/0.0083203`；全局 avg 为 `0.0030760`，全局 peak 为 `0.0083247`，约等于平均 `0.3076%`、单点最高 `0.8325%`。
- after 时刻 `kv_cache_usage=0.0` 与运行期 avg/peak 非零并不矛盾：前者只描述正式压测结束后的单个时刻，后两者描述压测执行期间的 81 次采样。

## 18. `inspect` 摘要与轻量 Manifest 字段

inspect stdout JSON 字段：

| 字段 | 含义 |
|---|---|
| `run_id`、`mode` | 基础运行 ID 和缓存模式。 |
| `requested_target_hit_rate` | Scenario 请求目标。 |
| `effective_target_hit_rate` | 求解器选择的可达目标。 |
| `theoretical_hit_rate` | 临时构造数据的理论值。 |
| `reachable_min`、`reachable_max` | 全局可达范围。 |
| `target_reachable` | 请求目标是否可达。 |
| `group_reachability` | 每组的 `reachable_min`、`reachable_max`。 |
| `groups` | 每组正式请求数量。 |
| `input_tokens`、`output_tokens` | 长度摘要，并额外包含 `total`。 |
| `dp_route_counts` | cold 下各 DP rank 请求数；warmup 通常为空对象。 |
| `sends_requests` | 固定为 `false`，表示不访问推理服务。 |
| `log` | inspect 日志路径。 |
| `manifest` | 本次 inspect 轻量 Manifest 路径。 |

inspect 轻量 Manifest 顶层字段：

| 字段 | 含义 |
|---|---|
| `schema_version`、`plugin_version` | Manifest 契约版本和插件版本。 |
| `status` | 固定为 `"inspected"`。 |
| `run_id` | 已追加时间戳的运行 ID。 |
| `scenario_path`、`scenario_sha256` | 原 Scenario 路径与内容指纹。 |
| `effective_config` | 补齐默认值并追加时间戳后的有效配置；API key 只记录是否配置。 |
| `inspect.timestamp` | 可复用时间戳，格式 `YYYYMMDD_HHMMSS`。 |
| `inspect.base_run_id`、`inspect.base_output_dir` | 未追加时间戳的基础运行 ID 和输出目录。 |
| `inspect.sends_requests` | 固定为 `false`。 |
| `inspect.summary` | inspect stdout 摘要的落盘副本（写入时不包含随后追加的 `manifest` 路径）。 |

prepare/run 复用前会检查 Manifest 版本、状态、时间戳格式、时间戳化 run/output 与 Scenario SHA-256。prepare 只复用 `inspected`，run 可复用 `inspected` 或 `prepared`；直接组装 AISBench 配置只接受 `prepared`。
