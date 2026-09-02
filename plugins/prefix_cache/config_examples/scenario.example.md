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
| `prefix_cache` | 否 | 缓存模式、目标命中率、组和顺序。 |
| `service` | 否 | 校验契约保留的服务段；`dp_size` 用于 cold DP 路由。 |
| `validation` | 否 | 偏差告警阈值。 |
| `aisbench` | 否 | 后续 AISBench 在线流程的兼容保留段；当前离线分支不消费。 |

嵌套对象同样采用严格字段白名单：

- `corpus.selection`：`mode`、`values`、`indices`、`question_sha256`；
- `requests.input_length`：`mode`、`value`、`values`、`ranges`、`min`、`max`、`mean`、`std`、`path`；每个 range 项只允许 `min`、`max`、`count`；
- `requests.output_length`：`mode`、`value`、`min`、`max`、`mean`、`std`、`path`；
- `prefix_cache.groups`：`count`、`assignment`、`overrides`；assignment 内允许 `mode`、`exponent`、`weights`；
- `prefix_cache.order`（即 `order` 对象）：`strategy`；
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
| `run_id` | 否 | `"gsm8k-prefix-cache-60"` | 基础运行 ID。执行时追加时间戳，并作为四类产物的文件名前缀。prepare 可复用最近一次匹配 inspect 的时间戳。 |
| `random_seed` | 否 | `42` | 控制 GSM8K 随机选择、长度采样、组分配、顺序和唯一 seed。相同输入与配置应生成相同内容。 |
| `output_dir` | 否 | `"./outputs/gsm8k-prefix-cache-60"` | 基础产物目录。执行时在最后一级目录名后追加与 run ID 相同的时间戳；prepare 可复用 inspect 已创建的目录。 |
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

时间戳采用 `_YYYYMMDD_HHMMSS`。`inspect` 每次创建新时间戳，并在基础输出目录旁写入 `<output_dir>.inspect.json`；后续 `prepare` 在指针匹配且时间戳目录仍存在时复用该时间戳，否则创建新时间戳。指针只比较基础 `run_id` 和 `output_dir`，不比较 Scenario 哈希；修改其他参数后若要求使用新目录，应重新执行 `inspect`，或删除旧指针后再执行 `prepare`。

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

本分支只生成数据与预热计划，不实际执行预热请求。warmup 请求不进入正式请求数或理论分母。
省略时默认 `warmup`。

### 8.2 `target_hit_rate`

期望的全局 token 加权命中率，范围 `[0,1]`。它是求解器的主目标，不等于简单地把每条请求的固定百分比设成前缀。
省略时默认 `0.6`。

求解会考虑 Block 对齐、请求顺序、Prefix Group、水位和 cold DP 路由。目标不可精确达到时，采用最接近的可达值并记录 requested/effective/theoretical 及原因。

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

理论水位总是按最终发送顺序重新模拟。

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
| `inference_url` | 否 | `"http://127.0.0.1:8000/v1/completions"` | 后续在线流程使用的 vLLM Completions API 地址。当前离线生成只校验最终有效值为非空字符串。 |
| `metrics_url` | 否 | `"http://127.0.0.1:8000/metrics"` | Prometheus 地址，仅作非空校验字段。 |
| `reset_url` | 否 | `"http://127.0.0.1:8000/reset_prefix_cache"` | Prefix Cache reset 地址，仅作可选配置。 |
| `model` | 否 | `"model-name"` | 后续在线流程兼容字段。当前只做非空校验并保留在 effective config，不写入 `requests.jsonl` 或 warmup 请求。 |
| `dp_size` | 否 | `2` | 单入口内部 DP rank 数，必须为正整数。**离线用于 cold 模式的 DP 路由**与 warmup 预热计划。 |
| `assume_empty_cache` | 否 | `false` | 仅作可选配置，离线不消费。 |
| `engine_label_map` | 否 | `{}` | 仅作可选配置，离线不消费。 |
| `timeout_seconds` | 否 | `30` | 仅作可选配置，离线不消费。 |
| `api_key` | 否 | `""` | 仅作可选配置。Manifest 不保存明文，只记录是否配置；Scenario 文件本身仍需限制权限。 |

> 本分支不访问任何服务地址。`inference_url`、`metrics_url`、`model` 都有默认值，用户无需显式填写，但最终有效值必须为非空字符串；其余服务字段仅兼容保留。当前真正参与离线计算的是 `dp_size`。

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
| `actual_warning_pp` | `5.0` | 实际值与理论值相差超过多少百分点时记录 `ACTUAL_DEVIATION`。**在线流程**使用；本离线分支保留该字段但不消费。 |

单位是百分点（pp），不是相对百分比。例如 60% 与 58.5% 相差 1.5 pp。两种偏差始终只 warning，不改变原本成功的退出码。

分析产物同时记录带符号偏差、绝对偏差、目标是否在全局可达范围内，以及 `PASS`/`PASS_WITH_WARNING` 展示状态。该状态只用于展示，不控制退出码。

## 11. `aisbench`

```json
"aisbench": {
  "config": "./configs/prefix_cache.py",
  "work_dir": "./work_dirs/prefix_cache",
  "extra_args": ["--debug"]
}
```

| 字段 | 必填 | 默认值 | 当前用途 |
|---|---:|---|---|
| `config` | 否 | 无 | AISBench 在线配置兼容字段；离线 `inspect/prepare/validate` 不消费。 |
| `work_dir` | 否 | 无 | AISBench 在线工作目录兼容字段；离线流程不消费。 |
| `extra_args` | 否 | 无 | AISBench 在线附加参数兼容字段；离线流程不消费。 |

整个 `aisbench` 段省略时默认 `{}`。源码当前只限制该对象允许出现以上三个键，不校验三个值的类型，也不会据此启动 AISBench；配置这些字段不会改变本分支的离线产物。

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
- 理论/目标超过 1 pp 时只告警（`actual_warning_pp` 字段为在线流程保留）。

## 13. 建议检查顺序

```bash
ais-bench-prefix-cache inspect --scenario ./scenario.json
ais-bench-prefix-cache prepare --scenario ./scenario.json
ais-bench-prefix-cache validate --manifest <manifest路径>
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

`inspect` 会加载 tokenizer 和 GSM8K，在临时目录复用完整 prepare 流程计算可达范围，但不发送请求，也不在正式 `result/` 目录保留四类数据产物。

- 每次执行生成新时间戳；
- 日志写入 `output_dir_时间戳/log/<run_id_时间戳>.inspect.log`；
- 成功后写入 `<基础output_dir>.inspect.json`；
- stdout JSON 包含 `log` 路径。

### 14.2 `prepare`

`prepare` 优先复用最近一次有效 inspect 指针的时间戳；没有有效指针时生成新时间戳。生成 prompt 时进度条写入 stderr，每完成一条 prompt 增加 1；最后一行 stdout JSON 固定包含：

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

正常成功退出码为 `0`；Scenario、生成或产物校验错误返回 `2`。目标不可达和命中率偏差始终只是 warning，不改变成功退出码。

## 15. 请求产物字段

### 15.1 `<run_id>.requests.jsonl`

每行严格只包含以下三个字段，且落盘顺序固定：

| 字段 | 类型 | 含义 |
|---|---|---|
| `question` | string | 最终完整 prompt。 |
| `answer` | string | AISBench 兼容占位值，当前固定为 `"none"`。 |
| `max_tokens` | integer | 最大输出 token 数，来自 `requests.output_length` 或组级覆盖。 |

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

### 16.2 `requests`

- `count`：正式请求数；
- `total_input_tokens`：所有正式请求实际输入 token 总和；
- `input_length_summary`、`output_length_summary`：输入/输出长度摘要。

每个长度摘要包含 `min`、`max`、`mean`、`p50`、`p90`、`p95`、`p99`、`bins`。`bins` 最多十个非空桶；每项包含该桶内实际观测的 `min`、`max` 和 `count`。

### 16.3 `prefix_cache`

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

### 16.4 `groups.<group_id>`

- `canonical_prefix_sha256`、`canonical_prefix_tokens`：canonical 前缀指纹和总 token 数；
- `max_shared_prefix_tokens`：该组正式请求使用的最大公共前缀长度；
- `gsm_indices`、`gsm_question_sha256`：canonical 前缀语料来源；
- `reachable_min`、`reachable_max`：该组理论可达范围；
- `theoretical_hit_rate`：该组 token 加权理论命中率。

### 16.5 `dp`、`warmup`、`divergence`

- `dp.size`：DP 数；`cold_route_strategy`：cold 时为 `"group_round_robin"`，warmup 时为 `null`；
- `warmup.enabled`：是否启用；`warmup.plan`：预热项列表；
- 每个 warmup 项包含 `request_id`、`group_id`、`dp_rank`、`prompt`、`input_tokens`、`shared_prefix_tokens`、`max_tokens`、`included_in_formal_statistics`；最后一个字段固定为 `false`；
- `divergence.strategy`：当前为 `"globally_unique_seed_block"`；
- `unique_request_blocks`、`request_count`、`collision_status`：唯一 seed 数、请求数和碰撞检查结论。

### 16.6 `artifacts` 和密钥处理

- `artifacts.full`、`artifacts.requests`：`name`、`path`、`rows`、`bytes`、`sha256`；
- `artifacts.analysis`：`name`、`path`、`bytes`、`sha256_at_prepare`。

Manifest 不保存 `service.api_key` 明文；它会被替换为 `effective_config.service.api_key_configured` 布尔值。

## 17. `analysis.json` 完整字段

| 字段 | 含义 |
|---|---|
| `schema_version` | 分析契约版本。 |
| `run_id` | 已追加时间戳的运行 ID。 |
| `status` | 成功 prepare 时为 `"prepared"`。 |
| `requested_target_hit_rate` | Scenario 请求目标。 |
| `effective_target_hit_rate` | 最近可达目标。 |
| `theoretical_hit_rate` | 最终顺序理论值。 |
| `target_difference_pp` | 当前等于目标绝对偏差。 |
| `target_signed_difference_pp` | `theoretical - requested` 的带符号百分点差。 |
| `target_absolute_difference_pp` | 目标绝对偏差。 |
| `validation` | 展示状态和可达性。 |
| `theory` | 全局、分组和分 DP 理论 token 统计。 |
| `warnings` | 目标不可达或偏差告警列表。 |

`validation` 包含 `status`、`target_reachable`、`warning_only`、`affects_exit_code`。后两项固定为 `true`、`false`，表示告警不影响成功退出码。

`theory` 包含 `input_tokens`、`hit_tokens`、`groups`、`dp`；每个组或 DP 值包含 `input_tokens`、`hit_tokens`、`hit_rate`。warmup 正式请求没有固定 `dp_rank`，所以 `theory.dp` 可以为空对象。

`warnings` 可能包含：

- `TARGET_UNREACHABLE`：`code`、`requested_target_hit_rate`、`reachable_min`、`reachable_max`；
- `TARGET_DEVIATION`：`code`、`difference_pp`。

`actual_warning_pp` 供未来在线实际值分析使用，因此当前离线 analysis 不会生成 `ACTUAL_DEVIATION`。

## 18. `inspect` 摘要与复用指针字段

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

`<基础output_dir>.inspect.json` 字段：

| 字段 | 含义 |
|---|---|
| `schema_version` | 指针契约版本，当前为 `"1.0"`。 |
| `timestamp` | 可复用时间戳，格式 `YYYYMMDD_HHMMSS`。 |
| `run_id` | 基础运行 ID，不含时间戳。 |
| `output_dir` | 基础输出目录。 |
| `output_dir_with_timestamp` | inspect 日志所在的时间戳目录。 |

prepare 复用前会检查指针版本、基础 run/output、时间戳格式及时间戳目录是否存在；不会比较 Scenario SHA-256。
