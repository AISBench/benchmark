# Scenario 配置参数说明

本文逐字段解释 [scenario.example.json](scenario.example.json)，并补充示例中没有展开的可选参数和模式。

## 1. 基本规则

- 配置是严格 JSON，不能写注释、尾随逗号或未支持的字段；
- 相对路径以 Scenario JSON 所在目录为基准；
- 比例使用 `0.0～1.0`，例如 `0.6` 表示 60%；
- token 长度以 `tokenizer.path` 加载出的 tokenizer 编码结果为准；
- 示例中的 `{}` 和 `[]` 只是 JSON 对象与列表，不是参数。

## 2. 顶层字段

| 字段 | 必填 | 作用 |
|---|---:|---|
| `schema_version` | 是 | 配置契约版本，当前只能为字符串 `"1.0"`。 |
| `run` | 是 | 运行标识、随机种子和产物目录。 |
| `tokenizer` | 是 | token 计算和 Block 对齐。 |
| `corpus` | 是 | GSM8K 来源及样本选择方式。 |
| `requests` | 是 | 正式请求数量和输入/输出长度。 |
| `prefix_cache` | 是 | 缓存模式、目标命中率、组和顺序。 |
| `service` | 是 | 单个 vLLM HTTP 入口、指标、reset 和 DP。 |
| `validation` | 否 | 偏差告警阈值。 |
| `aisbench` | 否 | AISBench 配置、结果目录和附加参数。 |

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
| `run_id` | 是 | 无 | 稳定运行 ID，也是四类产物的文件名前缀。建议使用字母、数字、短横线和下划线。 |
| `random_seed` | 是 | 无 | 控制 GSM8K 随机选择、长度采样、组分配、顺序和唯一 seed。相同输入与配置应生成相同内容。 |
| `output_dir` | 是 | 无 | 四类产物目录；不存在时自动创建。 |
| `overwrite` | 否 | `false` | 一体化 `run` 发现 Scenario 已变化时，是否允许重建旧产物。`prepare --overwrite` 可临时覆盖。 |

示例会生成：

```text
gsm8k-prefix-cache-60.full.jsonl
gsm8k-prefix-cache-60.requests.jsonl
gsm8k-prefix-cache-60.manifest.json
gsm8k-prefix-cache-60.analysis.json
```

## 5. `tokenizer`

```json
"tokenizer": {
  "path": "/models/example",
  "block_size": 16,
  "trust_remote_code": false
}
```

| 字段 | 必填 | 默认值 | 作用 |
|---|---:|---|---|
| `path` | 是 | 无 | 传给 `AutoTokenizer.from_pretrained` 的本地目录或 Hugging Face 标识。必须与 vLLM 服务端 tokenizer 一致。 |
| `block_size` | 是 | 无 | Prefix Cache Block 的 token 数。公共前缀和 seed 按它对齐，必须与服务端实际值一致。 |
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
| `path` | 是 | 无 | GSM8K JSONL 路径，每个非空行必须是 JSON 对象。 |
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

## 7. `requests`

```json
"requests": {
  "count": 100,
  "input_length": {
    "mode": "range",
    "ranges": [{"min": 512, "max": 1024, "count": 100}]
  },
  "output_length": {"mode": "fixed", "value": 32}
}
```

### 7.1 `count`

正式请求总数，必须是正整数。warmup 请求不计入该数量，也不写入 requests JSONL。

### 7.2 `input_length`

定义每条正式请求的目标输入 token 总数：

```text
公共前缀 + 全局唯一 seed + GSM8K 自然后缀
```

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

#### CSV 指定

```json
"input_length": {"mode": "csv", "path": "./input_lengths.csv"}
```

CSV 行数必须等于 `requests.count`，并包含以下任一正整数列：

- `input_prompt_tokens`；
- `content_tokens`；
- `input_tokens`。

### 7.3 `output_length`

该值写入 requests JSONL 的 `max_tokens`，运行时映射为 AISBench `max_out_len`。

#### 固定值

```json
"output_length": {"mode": "fixed", "value": 32}
```

#### 均匀分布

```json
"output_length": {"mode": "uniform", "min": 16, "max": 64}
```

在包含上下界的整数区间均匀采样。

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
  "groups": {
    "count": 4,
    "assignment": {"mode": "zipf", "exponent": 1.0}
  },
  "order": {"strategy": "interleave"}
}
```

### 8.1 `mode`

- `cold`：reset 后直接执行正式请求，按 `(Prefix Group, DP rank)` 从零维护缓存水位；
- `warmup`：reset 后先逐 `Prefix Group × DP rank` 预热，再采集正式 baseline。

warmup 请求不进入正式请求数、性能统计、理论分母或正式指标增量。

### 8.2 `target_hit_rate`

期望的全局 token 加权命中率，范围 `[0,1]`。它是求解器的主目标，不等于简单地把每条请求的固定百分比设成前缀。

求解会考虑 Block 对齐、请求顺序、Prefix Group、水位和 cold DP 路由。目标不可精确达到时，采用最接近的可达值并记录 requested/effective/theoretical 及原因。

### 8.3 `seed_blocks`

唯一 seed 的 Block 数，默认 `1`，必须为正整数：

```text
seed token 数 = seed_blocks × tokenizer.block_size
```

seed 位于公共前缀和自然后缀之间，所有正式请求全局唯一，防止请求在公共前缀之后继续意外共享。输入长度必须能容纳 seed。

### 8.4 `groups.count`

Prefix Group 数量。插件生成 `group-0`、`group-1` 等 ID。每个组独立生成 canonical 前缀、维护水位、统计理论命中率，并在 warmup 时逐 DP 预热。

### 8.5 `groups.assignment`

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

### 8.6 `groups.overrides`

可按组覆盖全局设置：

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

### 8.7 `order.strategy`

- `sequential`：保持目标分配阶段顺序；
- `within_group_shuffle`：各组内部打乱，再按组输出；
- `interleave`：各组轮转交错，默认，适合多租户流量；
- `global_shuffle`：全局确定性打乱。

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
| `inference_url` | 是 | 无 | vLLM Completions API 完整地址，用于 probe、warmup 和正式请求。 |
| `metrics_url` | 是 | 无 | Prometheus 地址，必须提供 Prefix Cache query/hit 指标。 |
| `reset_url` | 否 | `null` | Prefix Cache reset 地址。cold 和 warmup 默认都先 reset。 |
| `model` | 是 | 无 | 写入请求体的模型名称，必须被服务端接受。 |
| `dp_size` | 否 | `1` | 单入口内部 DP rank 数，必须与实际指标一致。 |
| `assume_empty_cache` | 否 | `false` | reset 缺失或失败时是否继续；启用后会告警，用户负责保证缓存为空。 |
| `engine_label_map` | 否 | `{}` | Prometheus `engine` 标签到 DP rank 的显式映射。 |
| `timeout_seconds` | 否 | `30` | probe、warmup、reset 和指标请求的单次 HTTP 超时秒数。 |
| `api_key` | 否 | `""` | 可选 Bearer Token。运行时写入 Header，Manifest 不保存明文；Scenario 文件本身仍需限制权限。 |

默认从 `engine` 标签末尾数字推导 rank，例如 `engine_0 → 0`。非数字标签需要配置：

```json
"engine_label_map": {
  "worker-east": 0,
  "worker-west": 1
}
```

多 DP 必须完整识别 `0～dp_size-1`，缺少、重复或越界都会失败。

支持的优先指标：

```text
vllm:prefix_cache_queries
vllm:prefix_cache_hits
vllm:kv_cache_usage_perc
```

同时兼容 `vllm:gpu_prefix_cache_queries[_total]`、`vllm:gpu_prefix_cache_hits[_total]` 和 `vllm:gpu_cache_usage_perc`。

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
| `actual_warning_pp` | `5.0` | 实际值与理论值相差超过多少百分点时记录 `ACTUAL_DEVIATION`。 |

单位是百分点（pp），不是相对百分比。例如 60% 与 58.5% 相差 1.5 pp。两种偏差始终只 warning，不改变原本成功的退出码。

## 11. `aisbench`

```json
"aisbench": {
  "config": "./prefix_cache_perf.py",
  "work_dir": "./outputs/aisbench",
  "extra_args": []
}
```

| 字段 | 必填 | 默认值 | 作用 |
|---|---:|---|---|
| `config` | `run` 时必需 | 无 | AISBench Python 配置；相对 Scenario 目录解析，可被 `run --config` 临时覆盖。 |
| `work_dir` | 否 | 示例 Python 配置的默认值 | 通过环境变量传给示例配置，决定 AISBench 自身日志和性能结果目录。 |
| `extra_args` | 否 | `[]` | 原样追加到 AISBench CLI；参数名和值必须各占一个字符串元素。 |

示例：

```json
"extra_args": ["--debug", "--max-num-workers", "1"]
```

插件固定使用 `--mode perf`，不要在 `extra_args` 中再传另一个 `--mode`。

## 12. 原示例最终表示的场景

- 生成 100 条正式请求；
- 输入长度在 512～1024 token 闭区间采样；
- 每条最多输出 32 token；
- 创建 4 个组，按指数 1.0 的 Zipf 热度分配；
- 目标全局命中率为 60%；
- 使用一个 16-token Block 作为全局唯一 seed；
- 请求按组交错排列；
- 使用 warmup 模式；
- 单个 vLLM HTTP 入口内部有 2 个 DP rank；
- 每组分别在 DP 0、DP 1 预热，共 8 条不进入正式统计的 warmup 请求；
- 理论/目标超过 1 pp、理论/实际超过 5 pp 时只告警。

## 13. 建议检查顺序

```powershell
ais-bench-prefix-cache inspect --scenario .\scenario.json
ais-bench-prefix-cache prepare --scenario .\scenario.json
ais-bench-prefix-cache validate --manifest <manifest路径>
ais-bench-prefix-cache run --scenario .\scenario.json
```

重点检查：

- `requested_target_hit_rate`：请求目标；
- `effective_target_hit_rate`：求解器选择的可达目标；
- `theoretical_hit_rate`：按最终顺序模拟的理论值；
- `reachable_min/max`：当前长度、Block、分组和路由下的范围；
- `groups`：请求分布与 canonical 前缀；
- `warmup.plan`：是否覆盖每个 Prefix Group × DP rank；
- `warnings`：目标偏差、空缓存假定或实际偏差。
