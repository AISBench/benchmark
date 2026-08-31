# AISBench Prefix Cache 插件 — 模块与函数级说明

> 本文档对 `plugins/prefix_cache/ais_bench_prefix_cache/` 下每个新增模块逐一说明，并细化到**函数/方法级别**。建议与 `ARCHITECTURE.md`（数据流视角）对照阅读。
>
> 代码总量约 2100 行（不含测试），按职责分为：异常、配置契约、产物 IO、数据生成、编排 pipeline、指标解析、运行时、CLI、AISBench 集成（dataset/model/inferencer/config）、打包。

---

## 目录

1. [`errors.py`](#1-errorspy--异常体系)
2. [`scenario.py`](#2-scenariopy--配置契约与校验)
3. [`artifacts.py`](#3-artifactspy--产物读写与校验)
4. [`generation.py`](#4-generationpy--数据生成与命中率求解)
5. [`pipeline.py`](#5-pipelinepy--prepareinspect-编排)
6. [`metrics.py`](#6-metricspy--prometheus-指标解析)
7. [`runtime.py`](#7-runtimepy--运行时编排)
8. [`cli.py`](#8-clipy--命令行入口)
9. [`config.py`](#9-configpy--aisbench-配置拼接)
10. [`datasets/prefix_cache_dataset.py`](#10-datasetsprefix_cache_datasetpy--数据集插件)
11. [`models/vllm_prefix_cache_api.py`](#11-modelsvllm_prefix_cache_apipy--模型插件)
12. [`openicl/icl_inferencer/prefix_cache_gen_inferencer.py`](#12-inferencer--推理器插件)
13. [`setup.py`](#13-setuppy--打包与入口注册)
14. [`config_examples/prefix_cache_perf.py`](#14-示例-aisbench-配置)

---

## 1. `errors.py` — 异常体系

所有插件异常都继承自基类 `PrefixCacheError`，CLI 层捕获它并返回退出码 2。分层设计让上层能精确捕获并区分错误来源。

| 类 | 继承 | 语义 | 典型抛出位置 |
|---|---|---|---|
| `PrefixCacheError` | `Exception` | 面向用户的插件基类 | — |
| `ScenarioValidationError` | `PrefixCacheError` | 配置或源数据非法 | `scenario.py`、`generation.py`（语料/CSV 校验） |
| `ArtifactValidationError` | `PrefixCacheError` | 产物不完整或不一致 | `artifacts.py`、`generation.py` |
| `PromptRoundTripError` | `ArtifactValidationError` | prompt 经 decode/re-encode 后 token 布局改变 | `generation.build_prompt` |
| `RuntimeCapabilityError` | `PrefixCacheError` | 服务能力不满足场景（缺指标/缺 rank/无法 reset 等） | `runtime.py`、`metrics.py` |

> 注意 `PromptRoundTripError` 是 `ArtifactValidationError` 的子类，因此 pipeline 中针对它的重试逻辑能单独捕获，同时它仍会被当作产物错误兜底。

---

## 2. `scenario.py` — 配置契约与校验

职责：把 `scenario.json` 变成强类型 `Scenario`，任何未知字段、非法值、路径问题都在这里被拒绝。

### 模块级常量

| 常量 | 作用 |
|---|---|
| `_ALLOWED` | 逐级字段白名单（顶层 `""`、`run`、`tokenizer`、`corpus`、`corpus.selection`、`requests`、`requests.input_length`、`requests.output_length`、`output`、`prefix_cache`、`prefix_cache.groups`、`prefix_cache.groups.assignment`、`prefix_cache.order`、`service`、`validation`、`aisbench`）。`output` 只允许 `output_key`；其他未在白名单内的字段一律报 `unknown field`。 |
| `_MODES` | 各节合法 mode 集合：`input`（fixed/explicit/range/truncated_normal/csv）、`output`（fixed/uniform/truncated_normal/csv）、`selection`（random/indices/question_sha256/mixed）、`assignment`（uniform/zipf/weights）、`order`（sequential/within_group_shuffle/interleave/global_shuffle/input_len_asc）、`cache`（cold/warmup）。 |

### 函数

| 函数 | 签名 | 职责 |
|---|---|---|
| `_require_dict` | `(value, path) -> dict` | 非 dict 时抛错；用于保证嵌套节是对象。 |
| `_strict_keys` | `(value, path) -> None` | 递归白名单校验；发现未知字段抛 `ScenarioValidationError`（标注完整字段路径，如 `prefix_cache.groups.foo`）。 |
| `_positive` | `(value, path) -> int` | 校验正整数（拒绝 bool、非 int、<1）。 |
| `_mode` | `(section, allowed, path) -> str` | 校验 `mode` 字段属于合法集合。 |
| `_validate_input_config` | `(config, path, base, expected_count) -> None` | 校验输入长度：`fixed`（value 正）、`explicit`（values 非空且长度==expected_count）、`range`（ranges 合法、count 求和==expected_count、max>=min）、`truncated_normal`（min/max/std 约束）、`csv`（path 解析为绝对路径）。同时拒绝多余字段。 |
| `_validate_output_config` | `(config, path, base) -> None` | 校验输出长度：`fixed`/`uniform`/`truncated_normal`/`csv`，约束同上（无 expected_count）。 |
| `_minimum_input_tokens` | `(config, path) -> int` | 计算该输入长度配置的**最小可能 token 数**（csv 模式会实际读 CSV 求最小值）。用于校验能否容纳非共享区。 |
| `_resolve_path` | `(base, value) -> str` | 相对路径转绝对（相对 Scenario 所在目录）。 |
| `_validate` | `(raw, source) -> dict` | **主校验入口**：先 `_strict_keys`，再校验必填字段、`schema_version=="1.0"`、run_id/random_seed、注入 `validation`/`aisbench` 默认节、解析路径、注入 tokenizer/selection/order/service 默认值，校验 `seed_blocks`、`minimum_non_shared_length >= seed 长度`、最小输入长度 >= 非共享区、组 overrides（ID 合法、字段白名单、覆盖后最小值仍容纳非共享区）、cold 多 DP 需 inference_url 等。返回深拷贝后的 `data`。 |
| `load_scenario` | `(path) -> Scenario` | 读取 JSON（utf-8）→ `_require_dict` → `_validate` → 构造 `Scenario`。读写失败或 JSON 非法抛 `ScenarioValidationError`。 |

### `Scenario`（frozen dataclass）

| 成员 | 说明 |
|---|---|
| `source_path` / `data` | 源文件路径 / 校验后配置 dict |
| `run_id` / `random_seed` / `output_dir` / `block_size` / `cache_mode` / `dp_size` | 常用字段的只读属性 |
| `section(name)` | 取配置子节 |
| `to_effective_dict()` | 深拷贝 `data`（供写入 Manifest 的有效配置） |

---

## 3. `artifacts.py` — 产物读写与校验

职责：四类产物的原子化读写、SHA256 计算、完整性校验。

### `ArtifactPaths`（frozen dataclass）

字段 `full / requests / manifest / analysis` 四个 `Path`。

### 函数

| 函数 | 签名 | 职责 |
|---|---|---|
| `sha256_file` | `(path) -> str` | 流式（1MB 块）计算文件 SHA256。 |
| `_atomic_text` | `(path, text, overwrite) -> None` | 原子写：先建父目录；若文件已存在且 `overwrite=False` 抛 `ArtifactValidationError`（拒绝覆盖）；写临时文件 `.name.tmp-pid` 后 `os.replace` 原子替换；失败清理临时文件。 |
| `write_json` | `(path, value, overwrite) -> None` | 写 JSON（`ensure_ascii=False, indent=2, sort_keys=True`），走 `_atomic_text`。 |
| `write_jsonl` | `(path, rows, overwrite) -> int` | 先物化行并保持插入顺序；requests.jsonl 固定先写 `question,answer`，随后是 `output.output_key` 指定的可选字段。逐行 `json.dumps` 后原子写，返回行数。 |
| `read_jsonl` | `(path) -> list[dict]` | 逐行 `json.loads`，跳过空行；IO/JSON 错误抛 `ArtifactValidationError`。 |
| `artifact_paths` | `(output_dir, run_id) -> ArtifactPaths` | 在 `output_dir/result/` 下由 `run_id` 生成四个文件名。 |
| `validate_artifacts` | `(manifest_path) -> dict` | 读取 Manifest，校验：full/requests 行数一致且等于 `manifest["requests"]["count"]`；`sequence_index` 连续；requests 每行严格匹配 `output.output_key` 的两/三字段契约；可选 `output_tokens` 映射到 full 的 `max_tokens`；full/requests 的 SHA256 与 Manifest 一致。旧 Manifest 继续按固定 `max_tokens` 校验。返回 `{ok, rows, run_id}`。 |

---

## 4. `generation.py` — 数据生成与命中率求解

职责：核心算法层。语料加载/选择、长度生成、分组、排序、DP 路由、前缀求解、唯一 seed、prompt 构造、理论命中率模拟。所有函数均确定性（相同输入相同输出）。

### 类型与协议

| 定义 | 说明 |
|---|---|
| `TokenizerLike` (Protocol) | 最小 tokenizer 协议：`encode(text, add_special_tokens=False)`、`decode(ids, skip_special_tokens=False)`。 |
| `GSMRecord` (frozen) | `line_index / question / question_sha256`。 |
| `CanonicalPrefix` (frozen) | `group_id / text / token_ids / sha256 / gsm_indices / gsm_hashes`。 |
| `RequestPlan` (frozen) | 单条请求完整计划（见 ARCHITECTURE 第 2 节）；含 `to_dict()`。 |
| `TheorySummary` (frozen) | `rows / total_input_tokens / total_hit_tokens / global_hit_rate / group_stats / dp_stats`。 |
| `SolveResult` (frozen) | 求解结果：`shared_prefix_tokens / requested_hit_tokens / effective_hit_tokens / effective_hit_rate / min_reachable_rate / max_reachable_rate / target_reachable / group_reachability / adjusted / reason`。 |

### 4.1 语料与选择

| 函数 | 签名 | 职责 |
|---|---|---|
| `normalize_question` | `(value) -> str` | 折叠连续空白为一个空格（去首尾）。 |
| `load_gsm8k` | `(path, field="question") -> list[GSMRecord]` | 读 utf-8-sig JSONL，逐行解析取 `field`，规范化，算 `question_sha256`；空行/非法行/空字段抛 `ScenarioValidationError`；空语料报错。 |
| `select_gsm8k` | `(records, config, count, seed) -> list[GSMRecord]` | 按 mode 选择：`random`（确定性打乱，超量循环）、`indices`（零基行号，超量循环）、`question_sha256`（哈希唯一匹配）、`mixed`（先 indices 再 hash）；不足时按合并顺序循环复用。 |

### 4.2 长度生成

| 函数 | 签名 | 职责 |
|---|---|---|
| `_csv_values` | `(path, aliases) -> list[int]` | 读 CSV（utf-8-sig），从别名列表中定位列，返回正整数列表；缺列/非法值报错。 |
| `build_input_lengths` | `(config, count, seed) -> list[int]` | 按 mode 生成 `count` 条输入长度：fixed/explicit/csv（行数==count）/range（每区间 randint）/truncated_normal。 |
| `_truncated_normal_values` | `(config, count, seed) -> list[int]` | 截断正态采样：`mean` 默认中点、`std` 默认 `max(1,(high-low)/4)`；`min==max` 直接固定；拒绝采样超上限抛错。 |
| `build_output_lengths` | `(config, count, seed) -> list[int]` | fixed/csv（`output_tokens` 列）/uniform/truncated_normal。 |

### 4.3 分组与排序

| 函数 | 签名 | 职责 |
|---|---|---|
| `assign_groups` | `(count, config, seed) -> list[str]` | 按 uniform/zipf/weights 分配请求到 `group-{i}`；用最大余数法把配额精确落到整数；zipf 再确定性打乱组序列。 |
| `order_indices` | `(group_ids, strategy, seed, input_lengths=None) -> list[int]` | 返回重排后的下标：sequential 保持原序、global_shuffle 全局打乱、within_group_shuffle 组内打乱再按组输出、interleave 各组轮转交错、input_len_asc 组内按长度升序再轮转（需要 input_lengths）。 |

### 4.4 DP 路由与理论模拟

| 函数 | 签名 | 职责 |
|---|---|---|
| `assign_cold_routes` | `(group_ids, dp_size, explicit=None) -> (ranks, lane_sequences)` | cold 路由：组内按出现次序对 `dp_size` 取模轮转（显式传入时校验合法性）；`lane_sequence` 是每个 `(group, rank)` lane 内的递增序号。 |
| `simulate_theory` | `(plans, mode, warmup_watermarks=None) -> TheorySummary` | 按最终顺序模拟缓存水位：warmup 用 `group_id` 为水位键（初值=各组最大前缀），cold 用 `(group_id, dp_rank)` 为键（从 0 开始）。对每条 `hit=min(prefix, before)`、`after=max(before, prefix)`，产出带 `watermark_before/hit/after` 的 rows 及全局/分组/分 DP 统计。 |
| `_plans_for_prefixes` | `(input_lengths, output_lengths, group_ids, ranks, lane_sequences, prefixes) -> list[RequestPlan]` | 内部辅助：按给定前缀序列快速构造 RequestPlan 列表，供求解器评分。 |
| `_balanced_warmup_prefixes` | `(input_lengths, caps, desired_hit_units, block_size) -> list[int]` | warmup 轨迹构造：按累计输入比例跟踪最终有效目标，并用剩余容量下界保证最终 Block 总量精确，避免前置填满、尾部归零。 |
| `_target_convergent_cold_prefixes` | `(input_lengths, group_ids, ranks, caps, desired_hit_units, block_size, beam_width=256) -> list[int]` | cold 轨迹构造：beam 状态保存各 `(group,rank)` 水位和累计 hit；按最大/总超调、累计率回落、目标距离的字典序选择精确总量解。 |

### 4.5 前缀求解器

| 函数 | 签名 | 职责 |
|---|---|---|
| `solve_prefix_lengths` | `(input_lengths, output_lengths, group_ids, ranks, lane_sequences, block_size, minimum_non_shared_tokens, mode, target_hit_rate) -> SolveResult` | **核心求解器**。① 构造 block 对齐候选；② 计算全局/组级可达区间；③ 把最终目标钳制到最近可达 Block 总量；④ warmup 均衡跟踪目标，cold 做 lane 水位感知轨迹搜索；⑤ 搜索受限时回退 `lane_hit = Σprefix − max(prefix)` 的 exact anchor 构造；⑥ 用同一 simulator 强校验最终 hit token，并返回 reachable/adjusted/reason。 |

### 4.6 唯一 seed 与边界安全 token

| 函数 | 签名 | 职责 |
|---|---|---|
| `_safe_token_text` | `(tokenizer, token_id, special) -> str \| None` | 判断单个 token 是否“边界安全”：非特殊 token、可 decode、单独 re-encode 等于自身、前/后拼接 "X" 后仍能还原该 token（避免 BPE 边界合并）。 |
| `find_boundary_safe_token_ids` | `(tokenizer, minimum) -> list[int]` | 遍历词表找边界安全 token，优先空格前缀 token（BPE 中不会与前文合并），返回至少 `minimum` 个。 |
| `_seed_round_trips` | `(tokenizer, seed) -> bool` | seed 序列 decode 后 re-encode 是否还原。 |
| `build_unique_seed` | `(tokenizer, safe_ids, request_id, seed_length, random_seed, exclude=None) -> tuple[int,...]` | 用 `sha256(random_seed:request_id:nonce)` 从 `safe_ids` 确定性选 `seed_length` 个 token，保证不重复、可往返；`nonce` 最多 4096 次。 |
| `build_unique_seed_tokens` | `(safe_ids, request_ids, seed_length, random_seed, tokenizer=None) -> dict[str, tuple]` | 批量生成多个唯一 seed（用于 warmup plan）。 |

### 4.7 前缀与 prompt 构造

| 函数 | 签名 | 职责 |
|---|---|---|
| `_repeat_tokens` | `(records, tokenizer, target) -> (tokens, indices, hashes)` | 循环拼接语料问题直到达到 `target` token，返回截断后的 token 及来源索引/哈希。 |
| `build_canonical_prefixes` | `(tokenizer, group_sources, max_lengths, block_size) -> dict[str, CanonicalPrefix]` | 为每组构造 canonical 前缀：轮换组内语料直到首 block 不与已用组碰撞；全碰撞时加确定性组标记兜底；最后做 decode/re-encode 往返校验。 |
| `build_prompt` | `(tokenizer, canonical, shared_prefix_tokens, seed, suffix_records, target_tokens) -> (text, tokens, indices, hashes)` | 拼接 `前缀[:shared] + seed + 自然后缀`，decode 后 re-encode 校验，不一致抛 `PromptRoundTripError`。 |

---

## 5. `pipeline.py` — prepare/inspect 编排

职责：把 `generation.py` 的算法按顺序编排，产出四类产物（`prepare_scenario`）或只读汇总（`inspect_scenario`）。

### 内部函数

| 函数 | 签名 | 职责 |
|---|---|---|
| `_tokenizer_loader` | `(scenario) -> tokenizer` | 从 `transformers.AutoTokenizer.from_pretrained` 加载（revision/trust_remote_code 透传）；`transformers` 未装抛 `ArtifactValidationError`。 |
| `_sha256_json` | `(value) -> str` | 规范化 JSON（sort_keys、紧凑分隔）后 SHA256，用于有效配置/tokenizer 指纹。 |
| `_request_random_seed` | `(global_seed, request_id) -> int` | 由 `sha256(global_seed:request_id)` 前 8 字节派生的每请求确定性种子。 |
| `_percentile` | `(sorted_values, percentile) -> float` | 线性插值分位数。 |
| `_length_summary` | `(values) -> dict` | 长度摘要：min/max/mean/p50/p90/p95/p99 + 最多 10 个分桶。 |
| `_tokenizer_manifest` | `(tokenizer, effective, block_size) -> dict` | tokenizer 指纹（path/revision/class/vocab_size/special_ids）的 SHA256 与 block_size。 |
| `_build_prompt_with_seed_retry` | `(tokenizer, canonical, prefix_len, seeds, request_id, rotated_pool, target_tokens, safe_ids, seed_length, random_seed)` | 调 `build_prompt`，遇 `PromptRoundTripError` 换 seed 重试（最多 64 次，换用 `random_seed + attempt*10007+1`）。 |

### 主函数

| 函数 | 签名 | 职责 |
|---|---|---|
| `prepare_scenario` | `(path, overwrite=None, tokenizer_loader=None, progress=None, execution_timestamp=None) -> ArtifactPaths` | **prepare 主流程**（详见 ARCHITECTURE 第 3 节）。可追加指定/新执行时间戳，并按每条 prompt 回调进度；随后依次完成生成、求解、落盘、warmup 计划、analysis/manifest 与 `validate_artifacts` 自检。Manifest 中 `service.api_key` 明文被替换为 `api_key_configured` 布尔。 |
| `inspect_scenario` | `(path, tokenizer_loader=None) -> dict` | 只读检查：把 run_id/output_dir 改写进临时目录后调用 `prepare_scenario`，汇总 requested/effective/theoretical、可达范围、组分布、输入/输出长度摘要、DP 路由计数。不访问 vLLM、不留正式产物、不发请求。 |

> `prepare_scenario` 中各类种子派生顺序见 ARCHITECTURE 第 9 节，是复现性的关键。

---

## 6. `metrics.py` — Prometheus 指标解析

职责：解析 `/metrics` 文本，识别各 DP rank 的 query/hit/kv 指标，计算 token 加权实际命中率。

### 类型

| 定义 | 字段 |
|---|---|
| `RankMetrics` (frozen) | `queries / hits / kv_cache_usage` |
| `MetricSnapshot` (frozen) | `by_rank: {rank→RankMetrics} / metric_names: {逻辑名→实际指标名} / raw_text` |
| `ActualMetrics` (frozen) | `by_rank / global_queries / global_hits / global_hit_rate` |

### 模块级常量

- `_SAMPLE`：Prometheus 样本正则（名称 + 可选 `{labels}` + 数值 + 可选时间戳）。
- `_LABEL`：`key="value"` 标签解析（支持转义）。
- `_ALIASES`：逻辑名 → 候选指标名，兼容新旧别名：
  - queries：`vllm:prefix_cache_queries[_total]`、`vllm:gpu_prefix_cache_queries[_total]`
  - hits：`vllm:prefix_cache_hits[_total]`、`vllm:gpu_prefix_cache_hits[_total]`
  - kv：`vllm:kv_cache_usage_perc`、`vllm:gpu_cache_usage_perc`

### 函数

| 函数 | 签名 | 职责 |
|---|---|---|
| `_rank` | `(labels, dp_size, mapping) -> int` | 从 `engine` 标签解析 DP rank：显式 map 优先，否则取末尾数字；无标签且 dp_size==1 返回 0，否则报错。 |
| `parse_metrics` | `(text, dp_size, engine_label_map=None) -> MetricSnapshot` | 逐行解析样本，按逻辑名选中指标，逐 rank 校验（缺 queries/hits 报错、rank 越界/重复报错、hits>queries 报错、缺 rank 报错），产出 `MetricSnapshot`。 |
| `diff_metrics` | `(before, after) -> ActualMetrics` | 要求前后 rank 集合一致；每 rank 做 after−before 增量，回归（负增量）或 hits>queries 报错；全局命中率 = `Σhits/Σqueries`（token 加权，非简单平均）。 |
| `metrics_to_dict` | `(actual) -> dict` | `ActualMetrics` → JSON 友好 dict（by_dp 含每 rank hit_rate）。 |
| `summarize_kv_usage` | `(samples) -> dict` | 聚合跑分期间轮询的 KV 用量样本（`{rank: 占比}` 列表，None 表示缺失）：返回每 rank `peak`/`avg`/`sample_count` 与 `global_peak`/`global_avg`。 |
| `snapshot_to_dict` | `(snapshot, include_raw=True) -> dict` | `MetricSnapshot` → dict，可选附 `raw_prometheus` 原文。 |

---

## 7. `runtime.py` — 运行时编排

职责：封装 vLLM HTTP 客户端、AISBench 配置渲染、`run` 主流程、离线 `analyze`。

### `VLLMClient` 类

| 方法 | 职责 |
|---|---|
| `__init__(scenario)` | 读取 `service` 配置，设超时；有 `api_key` 则加 `Authorization: Bearer`。 |
| `_request(url, method="GET", body=None, dp_rank=None)` | 底层 urllib 请求；`dp_rank` 非空时加 `X-data-parallel-rank` 头；网络错误抛 `RuntimeCapabilityError`。 |
| `send_completion(prompt, max_tokens, dp_rank=None)` | POST `/v1/completions`（body 含 model/prompt/max_tokens/temperature=0/stream=False），返回 JSON；解析失败抛 `RuntimeCapabilityError`。 |
| `snapshot()` | GET `/metrics` → `parse_metrics` → `MetricSnapshot`。 |
| `precheck()` | 对每个 DP rank 发一条 probe completion（多 DP 加定向头），再 snapshot；返回 `{ok, ranks, metric_names}`，用于验证推理/指标/全部 DP 可用。 |
| `reset()` | POST `reset_url`；未配置且 `assume_empty_cache` 返回 `[ASSUME_EMPTY_CACHE]` 告警；失败且允许假定则继续，否则抛 `RuntimeCapabilityError`。 |
| `warm_every_group_rank(plan)` | 校验 plan 覆盖每个 `(group, rank)`；按 `(group, rank)` 排序逐条定向 `send_completion`，返回每条的耗时记录。 |

### 配置渲染

| 函数 | 签名 | 职责 |
|---|---|---|
| `_ConfigTypeRef` | 类 | 类引用标记，`__repr__` 渲染为 import 别名表达式。 |
| `_render_config_value` | `(value, imports, refs)` | 递归把 `type` 渲染为 import 别名、dict/list 递归处理、其他值原样返回。 |
| `render_aisbench_config` | `(config_path, scenario) -> Path` | 设置 `AISBENCH_PREFIX_CACHE_SCENARIO`、精确 `_MANIFEST` 与 `_WORK_DIR` 环境变量，`exec` 用户 AISBench 配置，校验 `datasets/models/infer` 三个必需键，把配置渲染为静态 `config.py`（写到临时目录）返回其路径。exec 失败或缺键抛 `PrefixCacheError`。 |

### 主流程

| 函数 | 签名 | 职责 |
|---|---|---|
| `run_scenario` | `(scenario_path, aisbench_config=None, *, execution_timestamp=None, progress=None) -> dict` | **run 主流程**（见 ARCHITECTURE 第 4 节）：追加/复用时间戳 → `result/` 产物生成/校验 → scenario_sha256 校验 → precheck → reset → warmup → baseline → 渲染配置 → subprocess 跑 AISBench → after → `diff_metrics` → 追加 ACTUAL_DEVIATION warning → 写回 analysis.json（`status: complete`）。AISBench 非零退出抛 `PrefixCacheError`。 |
| `_read_json` | `(path) -> dict` | 读 JSON（模块内辅助）。 |
| `analyze_snapshots` | `(manifest_path, baseline_path, after_path) -> dict` | 离线重算：`validate_artifacts` → 读 manifest 得 dp_size/engine_label_map → `parse_metrics` 两份文件 → `diff_metrics` → 写回 analysis.json（`status: analyzed`）。不连接 vLLM。 |

---

## 8. `cli.py` — 命令行入口

| 函数 | 签名 | 职责 |
|---|---|---|
| `build_parser` | `() -> argparse.ArgumentParser` | 定义 5 个子命令：`prepare/--scenario/--overwrite`、`inspect/--scenario`、`validate/--manifest`、`run/--scenario/--config`、`analyze/--manifest/--baseline/--after`。 |
| `_reusable_execution_timestamp` | `(scenario, *, inspected_only) -> str | None` | 从匹配的时间戳 Manifest 发现可复用时间戳；prepare 只接受 `inspected`，run 接受 `inspected/prepared`。 |
| `_persist_inspect_manifest` | `(scenario_path, result, log_file, timestamp) -> Path` | 把 inspect 摘要、脱敏有效配置和 Scenario SHA 写入 `status="inspected"` 的轻量 Manifest。 |
| `main` | `(argv=None) -> int` | 解析并分发五个命令，维护 Manifest 驱动的时间戳复用、分层日志和 prepare/run 进度；捕获 `PrefixCacheError` 打印 `ERROR` 并返回 2，否则输出 JSON 并返回 0。 |
| `console_main` | `() -> None` | `raise SystemExit(main())`，作为 `console_scripts` 入口。 |

---

## 9. `config.py` — AISBench 配置拼接

职责：把 Scenario 与产物路径拼成 AISBench 能识别的 dataset/model 配置字典，供示例 `prefix_cache_perf.py` 调用。

| 函数 | 签名 | 职责 |
|---|---|---|
| `_manifest` | `(scenario) -> (Path, dict)` | 优先读取 `AISBENCH_PREFIX_CACHE_MANIFEST` 指向的本次时间戳 Manifest；直接调用时先尝试 `output_dir/result/<run_id>.manifest.json`，再扫描并校验最近的 `status="prepared"` 时间戳 Manifest。 |
| `build_dataset_config` | `(scenario_path) -> dict` | 返回 dataset 配置：`type=PrefixCacheDataset`、`requests_path/full_path/manifest_path`、`reader_cfg`（input `question/max_out_len`，output `answer`）、`infer_cfg`（`PromptTemplate "{question}"`、`ZeroRetriever`、`PrefixCacheGenInferencer`）、`eval_cfg`（`AccEvaluator`、`pred_role=BOT`）。 |
| `build_model_config` | `(scenario_path) -> dict` | 返回 model 配置：`type=VLLMPrefixCacheAPI`、`path=tokenizer.path`、`model/inference_url/api_key`、`stream=True`、`max_out_len=1`、`retry=2`、`generation_kwargs(temperature=0, ignore_eos=True)`、`batch_size=1`。显式流式模式保证 AISBench 能按 chunk 时间点计算 TTFT、TPOT 和 ITL。 |

---

## 10. `datasets/prefix_cache_dataset.py` — 数据集插件

职责：通过 `@LOAD_DATASET.register_module()` 注册到 AISBench，把产物转成 `datasets.Dataset`。

| 成员 | 职责 |
|---|---|
| `PrefixCacheDataset(BaseDataset)` | AISBench 数据集插件类。 |
| `PrefixCacheDataset.load(requests_path, full_path, manifest_path, **kwargs) -> Dataset` | 静态方法：`validate_artifacts` → 读 manifest 得 `prefix_cache.mode` → 逐行合并 requests/full（校验 `sequence_index` 顺序）→ 每行输出 `question/answer/max_out_len` + 元数据 `dp_rank/group_id/lane_sequence/cache_mode` → `Dataset.from_list`。 |

> 关键点：AISBench 只需要 `requests.jsonl` 的最小字段，但插件把 `full.jsonl` 的路由元数据按 `sequence_index` 合并注入，供 Inferencer 做 cold lane 排序。

---

## 11. `models/vllm_prefix_cache_api.py` — 模型插件

职责：通过 `@MODELS.register_module()` 注册，继承 `VLLMCustomAPI`，实现并发安全的每请求 DP 路由。

| 成员 | 职责 |
|---|---|
| `_DP_KEY = "_aisbench_prefix_cache_dp_rank"` | 请求体内部传递 DP rank 的私有键。 |
| `VLLMPrefixCacheAPI(VLLMCustomAPI)` | vLLM Completions 模型插件。 |
| `__init__(inference_url, *args, **kwargs)` | 从 `inference_url` 拆出 base URL 给父类（若路径以 `/v1/completions` 结尾则去掉该后缀），并把完整 URL 存到 `self.url`。 |
| `get_request_body(input_data, max_out_len, output, dp_rank=None, **args)` | 调父类构造 body 后，把 `dp_rank` 写入 `body[_DP_KEY]`。 |
| `_payload_and_headers(request_body)` | 从 body 剥离 `_DP_KEY` 生成 payload，若有 rank 则写 `X-data-parallel-rank` 头。 |
| `text_infer(request_body, output)` | 非流式推理：`_payload_and_headers` → POST → 状态码校验 → `parse_text_response`；JSON 非法抛 `AISBenchValueError(PARSE_TEXT_RSP_INVALID_FORMAT)`。 |
| `stream_infer(request_body, output)` | 正式 perf 使用的流式推理：POST 后逐行解析 SSE（`data:`/`[DONE]`），为每个有效 chunk 记录时间点并调 `parse_stream_response`，用于 TTFT、TPOT 和 ITL 计算。 |

---

## 12. Inferencer — 推理器插件

文件 `openicl/icl_inferencer/prefix_cache_gen_inferencer.py`，通过 `@ICL_INFERENCERS.register_module()` 注册。

### `LaneSequencer` 类

用 `asyncio.Condition` 实现的逐 lane 顺序屏障：

| 方法 | 职责 |
|---|---|
| `__init__` | `_conditions: defaultdict[(group,rank)→Condition]`、`_next: defaultdict[(group,rank)→int]`。 |
| `wait_turn(lane, sequence)` | 等待直到 `_next[lane] == sequence`（保证 lane 内按序执行）。 |
| `complete(lane)` | 递增 `_next[lane]` 并 `notify_all` 放行下一条。 |

### `PrefixCacheGenInferencer(GenInferencer)` 类

| 方法 | 职责 |
|---|---|
| `__init__` | 调父类，初始化 `_lane_sequencer`。 |
| `get_data_list(retriever)` | 调父类后，校验 `data_list` 长度等于 dataset 源长度（防止顺序被改），把 `dp_rank/group_id/lane_sequence/cache_mode` 注入每条 data。 |
| `do_request(data, token_bucket, session)` | cold 模式下按 `(group_id, dp_rank)` + `lane_sequence` 经 `LaneSequencer.wait_turn` 排队，执行父类 `do_request` 后 `complete`；非 cold 直接走父类。 |

---

## 13. `setup.py` — 打包与入口注册

| 项 | 说明 |
|---|---|
| 包名 | `ais-bench-prefix-cache`（版本 0.1.0） |
| `python_requires` | `>=3.10` |
| `install_requires` | `ais-bench-benchmark`、`aiohttp`、`datasets>=2.12.0,<=3.6.0`、`transformers` |
| `entry_points` | ① `ais_bench.benchmark_plugins` 组：`prefix_cache = ais_bench_prefix_cache`（让 AISBench 发现插件）；② `console_scripts`：`ais-bench-prefix-cache = ais_bench_prefix_cache.cli:console_main`（CLI）。 |

---

## 14. 示例 AISBench 配置

文件 `config_examples/prefix_cache_perf.py`：一个可被 `render_aisbench_config` 执行并静态化的 AISBench 配置模板。

| 项 | 说明 |
|---|---|
| `scenario` | 从环境变量 `AISBENCH_PREFIX_CACHE_SCENARIO` 读取（由 `render_aisbench_config`/`run_scenario` 注入）。 |
| `datasets` / `models` | 调用 `config.build_dataset_config` / `build_model_config`。 |
| `infer` | `NaivePartitioner` + `LocalRunner(max_num_workers=1, task=OpenICLApiInferTask)`。 |
| `summarizer` | `attr="accuracy", summary_groups=[]`。 |
| `work_dir` | 环境变量 `AISBENCH_PREFIX_CACHE_WORK_DIR`，缺省 `outputs/prefix_cache`。 |

> 该文件本身不包含类导入，因此能被 `exec` 后由 `_render_config_value` 把所有 `type` 引用渲染成静态 import，满足 AISBench（mmengine）惰性加载的约束。

---

## 附：模块依赖关系速查

```
cli.py ──▶ pipeline.py ──▶ generation.py ──▶ (算法纯函数)
       │        │
       │        └─▶ scenario.py ─▶ errors.py
       │        └─▶ artifacts.py
       │
       └─▶ runtime.py ──▶ metrics.py
                │              └─▶ errors.py
                ├─▶ pipeline.py（prepare_scenario 复用）
                └─▶ scenario.py

config.py ──▶ datasets/prefix_cache_dataset.py ──▶ artifacts.py
         │
         ├─▶ models/vllm_prefix_cache_api.py
         └─▶ openicl/.../prefix_cache_gen_inferencer.py
```

- `generation.py` 是纯算法层（除读取 CSV/语料文件），不依赖 runtime/metrics。
- `pipeline.py` 依赖 `generation` + `scenario` + `artifacts`，是离线生成的核心编排。
- `runtime.py` 依赖 `pipeline`（复用 prepare）、`metrics`（采集/差值）、`scenario`。
- AISBench 集成层（`config/datasets/models/openicl`）通过 entry point 独立加载，仅在 `run` 的子进程内被 AISBench 使用，与离线层通过四类产物解耦。
