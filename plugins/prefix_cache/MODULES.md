# AISBench Prefix Cache 插件 — 模块与函数级说明

> 本文档对 `plugins/prefix_cache/ais_bench_prefix_cache/` 下每个新增模块逐一说明，并细化到**函数/方法级别**。建议与 `ARCHITECTURE.md`（数据流视角）对照阅读。
>
> 本分支只保留离线数据生成与校验能力，代码总量约 1500 行（不含测试），按职责分为：异常、配置契约、产物 IO、数据生成、编排 pipeline、CLI、打包。

---

## 目录

1. [`errors.py`](#1-errorspy--异常体系)
2. [`scenario.py`](#2-scenariopy--配置契约与校验)
3. [`artifacts.py`](#3-artifactspy--产物读写与校验)
4. [`generation.py`](#4-generationpy--数据生成与命中率求解)
5. [`pipeline.py`](#5-pipelinepy--prepareinspect-编排)
6. [`cli.py`](#6-clipy--命令行入口)
7. [`setup.py`](#7-setuppy--打包与入口注册)

---

## 1. `errors.py` — 异常体系

所有插件异常都继承自基类 `PrefixCacheError`，CLI 层捕获它并返回退出码 2。分层设计让上层能精确捕获并区分错误来源。

| 类 | 继承 | 语义 | 典型抛出位置 |
|---|---|---|---|
| `PrefixCacheError` | `Exception` | 面向用户的插件基类 | — |
| `ScenarioValidationError` | `PrefixCacheError` | 配置或源数据非法 | `scenario.py`、`generation.py`（语料/CSV 校验） |
| `ArtifactValidationError` | `PrefixCacheError` | 产物不完整或不一致 | `artifacts.py`、`generation.py` |
| `PromptRoundTripError` | `ArtifactValidationError` | prompt 经 decode/re-encode 后 token 布局改变 | `generation.build_prompt` |

> 注意 `PromptRoundTripError` 是 `ArtifactValidationError` 的子类，因此 pipeline 中针对它的重试逻辑能单独捕获，同时它仍会被当作产物错误兜底。

---

## 2. `scenario.py` — 配置契约与校验

职责：把 `scenario.json` 变成强类型 `Scenario`，任何未知字段、非法值、路径问题都在这里被拒绝。

### 模块级常量

| 常量 | 作用 |
|---|---|
| `_ALLOWED` | 逐级字段白名单（顶层 `""`、`run`、`tokenizer`、`corpus`、`corpus.selection`、`requests`、`requests.input_length`、`requests.output_length`、`prefix_cache`、`prefix_cache.groups`、`prefix_cache.groups.assignment`、`prefix_cache.order`、`service`、`validation`）。未在白名单内的字段一律报 `unknown field`。 |
| `_MODES` | 各节合法 mode 集合：`input`（fixed/explicit/range/truncated_normal/csv）、`output`（fixed/uniform/truncated_normal/csv）、`selection`（random/indices/question_sha256/mixed）、`assignment`（uniform/zipf/weights）、`order`（sequential/within_group_shuffle/interleave/global_shuffle/input_len_asc）、`cache`（cold/warmup）。 |

> `service` 段整体保留并可完全省略：URL、model、DP 等默认值来自当前 Scenario 示例；用户显式提供时仍执行非空与类型校验。`dp_size` 用于 cold 模式的 DP 路由，离线生成阶段不实际访问任何地址。

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
| `_validate` | `(raw, source) -> dict` | **主校验入口**：先 `_strict_keys`，再按当前 `scenario.example.json` 补全省略值，校验 `schema_version=="1.0"`、run_id/random_seed、解析路径，校验 `seed_blocks`、`minimum_non_shared_length >= seed 长度`、最小输入长度 >= 非共享区、组 overrides、cold 多 DP 等。多态长度配置只给 fixed 模式注入 fixed 默认，避免污染 range/csv 等模式。 |
| `new_execution_timestamp` | `() -> str` | 生成本地秒级、文件名安全的 `YYYYMMDD_HHMMSS` 时间戳。 |
| `with_execution_timestamp` | `(scenario, timestamp) -> Scenario` | 同时把一个时间戳追加到 `run_id` 和 `output_dir` 最后一层目录名，返回新 Scenario。 |
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
| `write_jsonl` | `(path, rows, overwrite) -> int` | 先物化行，**保持插入顺序**（requests.jsonl 有公开字段顺序约定 `question,answer,max_tokens`），逐行 `json.dumps` 后原子写，返回行数。 |
| `read_jsonl` | `(path) -> list[dict]` | 逐行 `json.loads`，跳过空行；IO/JSON 错误抛 `ArtifactValidationError`。 |
| `artifact_paths` | `(output_dir, run_id) -> ArtifactPaths` | 在 `output_dir/result/` 下由 `run_id` 生成四个文件名。 |
| `validate_artifacts` | `(manifest_path) -> dict` | 读取 Manifest，校验：full/requests 行数一致且等于 `manifest["requests"]["count"]`；`sequence_index` 连续；requests 每行**严格只含** `question/answer/max_tokens`；requests 与 full 逐行对应；full/requests 的 SHA256 与 Manifest 一致。返回 `{ok, rows, run_id}`。 |

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

### 4.5 前缀求解器

| 函数 | 签名 | 职责 |
|---|---|---|
| `solve_prefix_lengths` | `(input_lengths, output_lengths, group_ids, ranks, lane_sequences, block_size, minimum_non_shared_tokens, mode, target_hit_rate) -> SolveResult` | **核心求解器**。① 对每条请求构造 block 对齐候选 `[0, block, 2b, …, floor((len−min_non_shared)/b)*b]`；② 先用全 0 与全最大前缀计算全局/组级 `min/max reachable rate`；③ 将目标命中 token 钳制到可达区间内最近的 Block 整数倍；④ warmup 按请求容量直接分配目标 Block；⑤ cold 按 `(Prefix Group, DP rank)` lane 使用 `lane_hit = Σprefix − max(prefix)`，以最大容量请求为 anchor，在线性时间内构造精确目标，不再使用可能陷入局部最优的爬山搜索；⑥ 返回 `target_reachable`、`adjusted` 和明确的越界/对齐原因。 |

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
| `prepare_scenario` | `(path, overwrite=None, tokenizer_loader=None, progress=None, execution_timestamp=None) -> ArtifactPaths` | **prepare 主流程**。加载后追加统一执行时间戳；每生成一条 prompt 调用 `progress(completed,total)`；四类产物写入 `output_dir_时间戳/result/`，再生成 analysis/manifest 并自检。Manifest 中 `service.api_key` 明文被替换为 `api_key_configured` 布尔。 |
| `inspect_scenario` | `(path, tokenizer_loader=None) -> dict` | 只读检查：把 run_id/output_dir 改写进临时目录后调用 `prepare_scenario`，汇总 requested/effective/theoretical、可达范围、组分布、输入/输出长度摘要、DP 路由计数。不访问 vLLM、不留正式产物、不发请求。 |

> `prepare_scenario` 中各类种子派生顺序见 ARCHITECTURE 第 8 节，是复现性的关键。

---

## 6. `cli.py` — 命令行入口

`PromptProgress` 使用纯标准库在 stderr 显示 `Generate prompts` 进度，不影响 stdout 最终 JSON。prepare 日志写入 `output_dir_时间戳/log/`。

| 函数 | 签名 | 职责 |
|---|---|---|
| `build_parser` | `() -> argparse.ArgumentParser` | 定义 3 个子命令：`prepare/--scenario/--overwrite`、`inspect/--scenario`、`validate/--manifest`。 |
| `main` | `(argv=None) -> int` | 解析并分发：prepare→`prepare_scenario`（打印产物路径）、validate→`validate_artifacts`、inspect→`inspect_scenario`。捕获 `PrefixCacheError` 打印 `ERROR` 并返回 2，否则 0。 |
| `console_main` | `() -> None` | `raise SystemExit(main())`，作为 `console_scripts` 入口。 |

> `main` 中通过自带的 `_install_logger` 安装 `ais_bench_prefix_cache` logger（不依赖 `ais_bench` 的 `AISLogger`）：解析到 `.log` 文件时日志只写入文件、不在终端打印；场景无法加载时回退为仅控制台输出。

---

## 7. `setup.py` — 打包与入口注册

| 项 | 说明 |
|---|---|
| 包名 | `ais-bench-prefix-cache`（版本 0.1.2） |
| `python_requires` | `>=3.10` |
| `install_requires` | `ais-bench-benchmark`（运行 AISBench 压测所需）、`transformers`（加载 tokenizer） |
| `entry_points` | `console_scripts`：`ais-bench-prefix-cache = ais_bench_prefix_cache.cli:console_main`（CLI）。不注册 `ais_bench.benchmark_plugins`，本分支不提供 AISBench 插件集成。 |

---

## 附：模块依赖关系速查

```
cli.py ──▶ pipeline.py ──▶ generation.py ──▶ (算法纯函数)
       │        │
       │        └─▶ scenario.py ─▶ errors.py
       │        └─▶ artifacts.py
```

- `generation.py` 是纯算法层（除读取 CSV/语料文件），不依赖任何运行时。
- `pipeline.py` 依赖 `generation` + `scenario` + `artifacts`，是离线生成的核心编排。
- 本分支不包含 `runtime.py`、`metrics.py`、`config.py`、`datasets/`、`models/`、`openicl/` 等在线/AISBench 集成模块，四类产物即为最终交付物。
