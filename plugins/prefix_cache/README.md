# AISBench Prefix Cache 数据生成与压测插件

这是一个独立的 AISBench 插件，用于构造具有可控公共前缀的数据集，并比较 vLLM Prefix Cache 的理论与实际命中率。它同时提供离线的 `inspect` / `prepare` / `validate`，在线的 `run`，以及使用已保存 Prometheus 快照复算的 `analyze`。

插件只增加 `plugins/prefix_cache` 下的新代码，不修改 AISBench 核心逻辑。

Scenario 示例见 [config_examples/scenario.example.json](config_examples/scenario.example.json)，完整字段说明见 [config_examples/scenario.example.md](config_examples/scenario.example.md)。
实现架构和模块职责见 [ARCHITECTURE.md](ARCHITECTURE.md)、[MODULES.md](MODULES.md) 与 [架构可视化](../../prefix_cache_architecture.html)。

## 1. 安装前准备

运行环境需要满足：

- Python 3.10 或更高版本；
- 当前 AISBench 仓库及其依赖可以正常导入；
- `transformers`：加载与 vLLM 模型一致的 tokenizer；
- `datasets` 和 `aiohttp`：AISBench Dataset 与 API 压测依赖；
- 一份 GSM8K JSONL 文件，每行至少包含 `question` 字段；
- 在线 `run` 还需要一个启用 Prefix Cache 的 vLLM Completions 服务。

数据生成时使用的 tokenizer 必须与 vLLM 服务端模型一致。否则本地计算的 token 长度、Block 边界和理论命中率会与服务端实际行为不一致。

## 2. 安装步骤及每条命令的作用

以下命令假设当前目录是 AISBench 仓库根目录，即包含 `setup.py`、`ais_bench/` 和 `plugins/` 的目录。

### 2.1 创建隔离环境（推荐）

```bash
python -m venv .venv
```

作用：在当前仓库创建独立 Python 环境，避免插件依赖与系统 Python 中的其他包互相影响。

```bash
source .venv/bin/activate
```

作用：在 Linux Bash 中启用该虚拟环境。后续 `python` 和 `pip` 都会使用 `.venv` 中的解释器和依赖。如果不激活，也可以直接使用 `.venv/bin/python` 执行后续命令。

### 2.2 安装 AISBench

```bash
python -m pip install -e .
```

作用：以 editable 模式安装当前 AISBench 仓库及其依赖。`-e` 表示直接引用工作区源码，后续修改源码后通常不需要重新安装。如果环境中已经安装了与当前源码匹配的 `ais-bench-benchmark`，可以跳过此步。

### 2.3 安装 Prefix Cache 插件

```bash
python -m pip install -e ./plugins/prefix_cache
```

这条命令会：

1. 安装 `ais_bench_prefix_cache` Python 包；
2. 注册 Prefix Cache Dataset、Inferencer 和 vLLM API Model 的 AISBench 插件入口；
3. 安装 `ais-bench-prefix-cache` 命令行入口。

### 2.4 验证安装

```bash
ais-bench-prefix-cache --help
```

作用：验证命令行入口是否安装成功，并列出 `inspect`、`prepare`、`validate`、`run` 和 `analyze` 五个子命令。

如果系统找不到该命令，可以使用等价形式：

```bash
python -m ais_bench_prefix_cache.cli --help
```

## 3. 首次使用

### 3.1 复制并修改配置

```bash
cp ./plugins/prefix_cache/config_examples/scenario.example.json ./scenario.json
```

作用：复制一份可编辑的 Scenario。执行前至少核对：

- `tokenizer.path`：与 vLLM 一致的 tokenizer；
- `corpus.path`：本地 GSM8K JSONL；
- `tokenizer.block_size`：与服务端 Prefix Cache Block 大小一致；
- `service.dp_size`：需要模拟 cold 多 DP 路由或生成 warmup 计划时，应与目标服务的 DP 数量一致。
- `service.inference_url`、`metrics_url`、`reset_url` 和 `model`：在线 `run` 使用；
- `aisbench.config`：正式压测使用的 AISBench Python 配置。

`inspect`、`prepare` 和 `validate` 不访问服务；`run` 会使用 `service` 段完成探活、reset、逐 DP warmup、正式压测和指标采集。当前只支持一个 HTTP 入口后面的单 DP 或多 DP，不支持多个彼此独立的 vLLM 实例。

各参数的默认值、约束和模式见 [Scenario 参数说明](config_examples/scenario.example.md)。下面给出完整字段索引和最关键的数据构造参数，README 本身可作为快速使用手册。

Scenario 中省略的字段会使用 `scenario.example.json` 的当前值作为默认值，包括默认 run、tokenizer、GSM8K 路径、100 条固定 1024-token 请求、`output.output_key=null`（requests 不带输出长度字段）、warmup 60% 目标、单一 uniform Prefix Group 和 DP 2。`minimum_non_shared_length` 是安全例外：它按 `seed_blocks × block_size` 动态推导；使用示例默认值时仍为 16。

### 3.2 Prefix Cache 数据构造参数

Scenario 采用严格白名单，完整配置层级如下；未列出的字段会被拒绝：

- `schema_version`；
- `run`：`run_id`、`random_seed`、`output_dir`、`overwrite`；
- `tokenizer`：`path`、`block_size`、`revision`、`trust_remote_code`；
- `corpus`：`path`、`field`、`selection`；`selection` 支持 `mode`、`values`、`indices`、`question_sha256`；
- `requests`：`count`、`input_length`、`output_length`；
  - `input_length` 支持 `mode`、`value`、`values`、`ranges`、`min`、`max`、`mean`、`std`、`path`，其中 `ranges` 项只允许 `min`、`max`、`count`；
  - `output_length` 支持 `mode`、`value`、`min`、`max`、`mean`、`std`、`path`；
- `output`：`output_key`，允许 `null`、`"max_tokens"`、`"output_tokens"`；
- `prefix_cache`：`mode`、`target_hit_rate`、`seed_blocks`、`minimum_non_shared_length`、`groups`、`order`；
  - `groups` 支持 `count`、`assignment`、`overrides`；
  - `assignment` 支持 `mode`、`exponent`、`weights`；
  - 每个 `overrides.group-N` 支持 `input_length`、`output_length`、`corpus_selection`；
  - `order` 支持 `strategy`；
- `service`：`inference_url`、`metrics_url`、`reset_url`、`model`、`dp_size`、`assume_empty_cache`、`engine_label_map`、`timeout_seconds`、`api_key`、`poll_interval_seconds`；
- `validation`：`target_warning_pp`、`actual_warning_pp`；
- `aisbench`：`config`、`work_dir`、`extra_args`；离线命令不消费，`run` 用于渲染配置并启动 AISBench perf。

各字段逐项含义见 [Scenario 完整字段说明](config_examples/scenario.example.md)。

#### requests.jsonl 输出字段

```json
"output": {"output_key": null}
```

该配置参考 `extract_qa.py --output-key`：

- `null`：默认，只输出 `question`、`answer`；
- `"max_tokens"`：追加 `max_tokens`；
- `"output_tokens"`：追加 `output_tokens`，值仍来自本请求生成的 `max_tokens`。

无论公开的 requests 文件是否带第三字段，完整审计文件 `full.jsonl` 都保留 `max_tokens`，AISBench Dataset 也从 full 文件读取 `max_out_len`，因此默认省略不会改变实际生成长度。

#### 输入长度模式

`requests.input_length` 控制每条正式请求的总输入 token 数。总长度由公共前缀、全局唯一 seed 和 GSM8K 自然后缀共同组成。

固定长度：

```json
"input_length": {"mode": "fixed", "value": 1024}
```

显式长度列表：

```json
"input_length": {
  "mode": "explicit",
  "values": [512, 768, 1024, 2048]
}
```

`values` 必须全部为正整数。作为全局配置时，元素数量必须等于 `requests.count`；作为 Prefix Group 覆盖配置时，元素数量必须等于该组实际请求数。

多个闭区间采样：

```json
"input_length": {
  "mode": "range",
  "ranges": [
    {"min": 512, "max": 1024, "count": 80},
    {"min": 2048, "max": 4096, "count": 20}
  ]
}
```

每个区间包含 `min` 和 `max`，所有 `count` 之和必须等于对应的请求数量。采样由 `run.random_seed` 确定，相同配置可重复生成相同长度序列。

截断正态分布：

```json
"input_length": {
  "mode": "truncated_normal",
  "min": 512,
  "max": 2048,
  "mean": 1024,
  "std": 256
}
```

只接受 `[min,max]` 内的整数采样。`mean` 默认取区间中点；`std` 默认根据区间宽度推导，显式设置时必须大于 0；`min=max` 等价于固定长度。

CSV 长度文件：

```json
"input_length": {"mode": "csv", "path": "./input_lengths.csv"}
```

CSV 行数必须等于对应请求数，并包含 `input_prompt_tokens`、`content_tokens` 或 `input_tokens` 中的一列。

#### 最小非共享长度与唯一 seed

```json
"prefix_cache": {
  "seed_blocks": 1,
  "minimum_non_shared_length": 16
}
```

唯一 seed 长度按下式计算：

```text
seed_tokens = seed_blocks × tokenizer.block_size
```

`minimum_non_shared_length` 默认等于 `seed_tokens`，不能小于 seed 长度。公共前缀的最大长度为：

```text
floor((input_length - minimum_non_shared_length) / block_size) × block_size
```

当最小非共享长度大于 seed 长度时，剩余空间由 GSM8K 自然后缀填充。每条正式请求都会使用实际参与构造的确定性 `request_random_seed` 生成差异 seed，seed token 序列在整个数据集中保持唯一，避免公共前缀结束后继续误共享。

Scenario 加载阶段会检查每种输入长度模式的最小值是否能容纳非共享区，不满足时在生成数据前直接报错。

#### 请求顺序策略

```json
"order": {"strategy": "input_len_asc"}
```

支持以下策略：

- `sequential`：保持数据生成阶段的稳定顺序；
- `within_group_shuffle`：每个 Prefix Group 内确定性打乱，再按组输出；
- `interleave`：不同 Prefix Group 按轮次交错；
- `global_shuffle`：所有请求全局确定性打乱；
- `input_len_asc`：每个 Prefix Group 内按输入长度从短到长排序，再按组轮转交错；相同长度保持原始顺序。

理论命中率始终按照最终发送顺序重新模拟。使用 `input_len_asc` + `cold` 时，短到长顺序会贯穿四个阶段：prepare 先重排长度和 Group，`requests.jsonl` / `full.jsonl` 按该顺序落盘，Dataset 校验 `sequence_index` 与行号一致，Inferencer 最后用 `LaneSequencer` 按 `(Prefix Group, DP rank)` 的 `lane_sequence` 串行放行。只有前一条请求完成后才发送同一 lane 的下一条，因此即使 AISBench 并发创建任务，也能模拟“首次请求无缓存，后续短请求到长请求逐步建立 Cache”。不同 Group 或 DP 的缓存彼此独立，仍可并发，不要求全局串行。

#### 可达性与长度统计

求解器会同时计算：

- 全局 `reachable_min` 和 `reachable_max`；
- 每个 Prefix Group 的 `reachable_min` 和 `reachable_max`；
- `target_reachable`，表示目标命中率是否处于全局可达区间；
- requested、effective 和 theoretical hit rate；
- 理论值减目标值的带符号偏差和绝对偏差。

目标高于 `reachable_max` 或低于 `reachable_min` 时，求解器直接选择对应边界解；目标位于区间内时，按 Block 单位选择最近可达命中量。最终 hit token 精度是硬约束，在此基础上再优化请求过程：warmup 按累计输入比例均衡分配前缀；cold 按 `(Prefix Group, DP rank)` 独立水位搜索，依次最小化累计命中率的最大超调、总超调和回落，再尽快贴近目标。因此不会再把大部分前缀集中到前几条、最后用零/短前缀回调。

严格单调并非所有顺序都可实现，例如后置 Prefix Group/DP lane 的首次 cold 请求必然 miss，或某条请求没有足够前缀容量。此时求解器仍优先选择超调和回落最小的精确总量解；极端搜索空间下会回退到确定性的 exact lane construction，确保最终 effective/theoretical 命中率不被轨迹优化破坏。

Manifest 的输入和输出长度摘要包含 `min`、`max`、`mean`、`p50`、`p90`、`p95`、`p99` 以及最多十个长度分桶。`inspect` 也会展示这些摘要和组级可达范围。

#### 验证状态与退出码

`analysis.json` 使用 `PASS` 或 `PASS_WITH_WARNING` 展示验证状态：

- 目标超出可达区间时记录 `TARGET_UNREACHABLE`；
- 理论值与目标相差超过 `target_warning_pp` 时记录 `TARGET_DEVIATION`。

这些状态和差异只用于展示，`warning_only=true` 且 `affects_exit_code=false`。在线实际值与理论值超过 `actual_warning_pp` 时也只记录 `ACTUAL_DEVIATION`；配置、产物、服务能力或 AISBench 执行错误才会失败。

### 3.3 `inspect`：检查配置和理论范围

```bash
ais-bench-prefix-cache inspect --scenario ./scenario.json
```

作用：

- 加载 tokenizer 和 GSM8K；
- 在临时目录构造数据并计算目标可达范围；
- 展示 requested/effective/theoretical hit rate；
- 展示组分布、输入/输出长度和 cold DP 路由摘要；
- 不访问 vLLM、不发送请求，也不在 Scenario 的 `output_dir` 留下四类正式数据产物；
- 与 `prepare` 一样生成 `_YYYYMMDD_HHMMSS` 时间戳，并把详细日志缓存到 `output_dir_时间戳/log/<run_id_时间戳>.inspect.log`；
- 成功后在 `output_dir_时间戳/result/` 写入同名轻量 `manifest.json`，其 `status="inspected"`，并把 inspect 摘要保存在 `inspect.summary`；不会再生成 `<output_dir>.inspect.json`；
- 输出的 JSON 摘要包含 `log` 和 `manifest` 路径，可直接定位日志与检查结果。

### 3.4 `prepare`：生成正式数据产物

```bash
ais-bench-prefix-cache prepare --scenario ./scenario.json
```

作用：根据 Scenario 确定性生成并校验四个文件：

- `result/<run_id_时间戳>.full.jsonl`；
- `result/<run_id_时间戳>.requests.jsonl`；
- `result/<run_id_时间戳>.manifest.json`；
- `result/<run_id_时间戳>.analysis.json`。

执行时会先显示 prompt 生成进度，且每成功生成一条 prompt 增加 1：

```text
Generate prompts [###############---------------] 50/100  50%
Generate prompts [##############################] 100/100 100%
{"full":"...","requests":"...","manifest":"...","analysis":"...","log":"..."}
```

进度写入 stderr，最后一行结果 JSON 写入 stdout，方便脚本继续解析。

时间戳采用 `_YYYYMMDD_HHMMSS`。单独执行 `prepare` 且没有可复用 inspect Manifest 时会生成新时间戳；`prepare` 会发现最近一次与当前 Scenario SHA-256 匹配的 `status="inspected"` Manifest，并在同一路径把它安全升级为正式 `status="prepared"` Manifest。`run` 可复用匹配的 inspected/prepared Manifest，使预览、生成、压测和校验位于同一个时间戳目录。

例如配置为：

```text
run_id:    gsm8k-prefix-cache-60
output_dir: ./outputs/gsm8k-prefix-cache-60
```

本次实际目录可能为：

```text
./outputs/gsm8k-prefix-cache-60_20260825_123456/
├── log/
│   └── gsm8k-prefix-cache-60_20260825_123456.prepare.log
└── result/
    ├── gsm8k-prefix-cache-60_20260825_123456.full.jsonl
    ├── gsm8k-prefix-cache-60_20260825_123456.requests.jsonl
    ├── gsm8k-prefix-cache-60_20260825_123456.manifest.json
    └── gsm8k-prefix-cache-60_20260825_123456.analysis.json
```

因此正常工作流不需要手动修改 `run_id` 或 `output_dir`，基础输出目录旁也不会出现额外的 `.inspect.json` 文件。插件直接扫描 `<output_dir>_时间戳/result/<run_id_时间戳>.manifest.json`，并校验 Manifest 版本、状态、时间戳化 run/output 以及 `scenario_sha256`；修改 Scenario 后旧 Manifest 会自动失配并创建新时间戳。

默认不覆盖同名文件。确定需要重建时使用：

```bash
ais-bench-prefix-cache prepare --scenario ./scenario.json --overwrite
```

`--overwrite` 只覆盖本次时间戳目录内该 run 对应的四个确定文件，不会清理整个输出目录。匹配的 inspect-only Manifest 可由 prepare 自动升级，不需要 `--overwrite`；正式 prepared Manifest 不会被后续 prepare 当作 inspect 占位复用。

### 3.5 `validate`：校验已有产物

```bash
ais-bench-prefix-cache validate --manifest ./outputs/gsm8k-prefix-cache-60_<时间戳>/result/gsm8k-prefix-cache-60_<时间戳>.manifest.json
```

作用：不生成数据、不访问 vLLM，只检查：

- Manifest、full 和 requests 行数是否一致；
- `sequence_index` 是否连续；
- requests 是否严格只含 `question`、`answer` 以及 `output.output_key` 指定的可选第三字段；
- requests 与 full 是否逐行对应；
- full 和 requests 的 SHA-256 是否匹配 Manifest。

它用于发现文件被手工编辑、截断、换序或使用了错误版本。

与 `inspect`/`prepare` 一样，validate 的详细日志写入 Manifest 所在时间戳输出目录的 `log/<run_id_时间戳>.validate.log`，终端只打印校验结果 JSON。

### 3.6 `run`：执行 vLLM Prefix Cache 压测

```bash
ais-bench-prefix-cache run --scenario ./scenario.json
```

完整流程为：校验或自动生成本时间戳的产物；逐 DP 探活；reset Prefix Cache（或按配置记录 `ASSUME_EMPTY_CACHE`）；warmup 模式按每个 `Prefix Group × DP rank` 定向预热；预热完成后采集正式 baseline；运行 AISBench `perf`；采集 after 并计算每 DP、全局实际命中率；最后把 `runtime`、`actual`、理论/实际差值和告警写回 `result/<run_id>.analysis.json`。warmup 在 baseline 之前完成，因此不进入正式吞吐、时延或命中率统计。

临时覆盖 Scenario 中的 AISBench 配置：

```bash
ais-bench-prefix-cache run --scenario ./scenario.json --config ./my_prefix_cache_perf.py
```

多 DP 使用同一个 HTTP 入口，并要求服务支持 `X-data-parallel-rank` 定向路由及带 `engine` 标签的分 DP Prometheus 指标。每个 DP 都会独立预热；不支持多实例服务编排。

### 3.7 `analyze`：用保存的指标快照离线复算

```bash
ais-bench-prefix-cache analyze \
  --manifest ./outputs/gsm8k-prefix-cache-60_<时间戳>/result/gsm8k-prefix-cache-60_<时间戳>.manifest.json \
  --baseline ./baseline.prom \
  --after ./after.prom
```

该命令不连接 vLLM、不运行 AISBench，只解析两份 Prometheus 文本，重新计算正式阶段计数器增量并写回 Manifest 对应的 analysis。详细日志位于同一时间戳目录的 `log/<run_id>.analyze.log`。

## 4. 推荐工作流

```bash
ais-bench-prefix-cache inspect --scenario ./scenario.json
ais-bench-prefix-cache prepare --scenario ./scenario.json
ais-bench-prefix-cache validate --manifest <manifest路径>
ais-bench-prefix-cache run --scenario ./scenario.json
```

这样可以在实际压测前人工审计数据。`prepare` 和 `run` 会通过匹配 Manifest 复用时间戳；`run` 发现同一时间戳下已有且与 Scenario 匹配的正式产物时直接复用。

## 5. cold 与 warmup

### cold

- 每个 `(group_id, dp_rank)` 从零水位开始；
- 同一组的请求按组内 round-robin 定向 DP；
- 插件保证同一 lane 内请求顺序；
- 可输出严格的分 DP 理论命中率。

### warmup

- 对每个 Prefix Group、每个 DP rank 分别预热；
- warmup 不进入 requests JSONL、理论分母或正式指标增量；
- 全局理论命中率有效，分 DP 主要展示实际指标。

预热计划落在 `result/<run_id_时间戳>.manifest.json` 的 `warmup.plan` 字段。`prepare` 只生成计划；`run` 才会执行预热请求，并在全部预热完成后采集 baseline。

## 6. 分层产物

所有正式数据产物位于实际时间戳输出目录的 `result/` 下，详细日志位于同级 `log/` 下。

### `<run_id>.full.jsonl`

完整审计数据，每行固定包含以下字段：

| 字段 | 含义 |
|---|---|
| `request_id` | 稳定请求 ID。 |
| `sequence_index` | 最终发送顺序中的零基序号。 |
| `group_id` | 所属 Prefix Group。 |
| `occurrence_index_within_group` | 该请求在组内第几次出现。 |
| `dp_rank` | cold 模式的目标 DP rank；warmup 正式请求为 `null`。 |
| `lane_sequence` | cold `(group_id, dp_rank)` lane 内序号；warmup 为 `null`。 |
| `target_input_tokens` | 配置要求的输入长度。 |
| `actual_input_tokens` | tokenizer 重编码后的实际输入长度。 |
| `max_tokens` | 最大输出 token 数。 |
| `shared_prefix_tokens` | 本请求使用的公共前缀 token 数。 |
| `seed_tokens` | 全局唯一 seed 的 token 数。 |
| `natural_suffix_tokens` | seed 后 GSM8K 自然后缀的 token 数。 |
| `question` | 最终完整 prompt。 |
| `answer` | AISBench 兼容占位值，当前固定为 `"none"`。 |
| `gsm_indices` | 本请求自然后缀使用的 GSM8K 零基行号。 |
| `gsm_hashes` | 对应规范化 question 的 SHA-256。 |
| `canonical_prefix_sha256` | 所属组 canonical 前缀指纹。 |
| `seed_sha256` | 本请求唯一 seed token 序列指纹。 |
| `request_random_seed` | 实际参与本请求 seed 构造的确定性随机种子。 |
| `watermark_before` | 请求到达前所在缓存 lane 的理论水位。 |
| `theoretical_hit_tokens` | 本请求理论命中 token 数。 |
| `watermark_after` | 请求完成后的理论水位。 |
| `theoretical_hit_rate` | `theoretical_hit_tokens / actual_input_tokens`。 |
| `divergence_block_sha256` | 差异块指纹，当前等于 `seed_sha256`。 |
| `divergence_unique` | 差异块是否通过全局唯一性检查。 |
| `collision_status` | 碰撞检查状态，成功产物为 `"pass"`。 |

### `<run_id>.requests.jsonl`

最小输入，每行固定先写 `question`、`answer`，再按 `output.output_key` 决定是否追加第三字段：

- `question`：最终完整 prompt；
- `answer`：当前固定为 `"none"`；
- `max_tokens` 或 `output_tokens`：可选；值为该请求最大输出 token 数。默认 `output_key=null`，两者都不写。

`full.jsonl` 始终保留 `max_tokens`。DP 路由等字段也只存在于 full 文件，不污染通用请求格式。

### `<run_id>.manifest.json`

复现和校验入口。顶层字段如下：

| 字段 | 含义 |
|---|---|
| `schema_version`、`plugin_version` | Manifest 契约版本和插件版本。 |
| `run_id` | 已追加执行时间戳的运行 ID。 |
| `scenario_path`、`scenario_sha256` | Scenario 绝对路径及原文件 SHA-256。 |
| `effective_config`、`effective_config_sha256` | 补齐默认值、解析路径后的有效配置及其指纹。 |
| `corpus_sha256` | GSM8K 文件 SHA-256。 |
| `tokenizer` | tokenizer 来源、类、词表、特殊 token、Block 和指纹。 |
| `requests` | 请求数、总输入 token、输入/输出长度摘要。 |
| `prefix_cache` | 模式、目标、理论值、可达区间、调整原因和验证状态。 |
| `groups` | 各组 canonical 来源、最大前缀、可达区间和理论命中率。 |
| `dp` | DP 数量及 cold 路由策略。 |
| `warmup` | 是否启用及逐组逐 DP 的预热计划。 |
| `divergence` | 唯一差异块策略、数量和碰撞状态。 |
| `artifacts` | full、requests、analysis 的名称、路径、行数、大小和哈希。 |

重要嵌套字段：

- `tokenizer`：`path`、`revision`、`class`、`vocab_size`、`special_token_ids`、`block_size`、`fingerprint_sha256`；
- `requests`：`count`、`total_input_tokens`、`input_length_summary`、`output_length_summary`；每个 summary 包含 `min`、`max`、`mean`、`p50`、`p90`、`p95`、`p99`、`bins`，每个 bin 包含 `min`、`max`、`count`；
- `prefix_cache`：`mode`、`requested_target_hit_rate`、`effective_target_hit_rate`、`theoretical_hit_rate`、`reachable_min`、`reachable_max`、`target_reachable`、`minimum_non_shared_length`、`adjusted`、`reason`、`validation_status`、`target_signed_difference_pp`、`target_absolute_difference_pp`；
- `groups.<group_id>`：`canonical_prefix_sha256`、`canonical_prefix_tokens`、`max_shared_prefix_tokens`、`gsm_indices`、`gsm_question_sha256`、`reachable_min`、`reachable_max`、`theoretical_hit_rate`；
- `dp`：`size`、`cold_route_strategy`；warmup 模式的路由策略为 `null`；
- `warmup`：`enabled`、`plan`；plan 每项包含 `request_id`、`group_id`、`dp_rank`、`prompt`、`input_tokens`、`shared_prefix_tokens`、`max_tokens`、`included_in_formal_statistics`；
- `divergence`：`strategy`、`unique_request_blocks`、`request_count`、`collision_status`；
- `artifacts.full/requests`：`name`、`path`、`rows`、`bytes`、`sha256`；`artifacts.analysis` 包含 `name`、`path`、`bytes`、`sha256_at_prepare`。

`api_key` 明文不会写入 Manifest；`effective_config.service` 中改为布尔字段 `api_key_configured`。

### `<run_id>.analysis.json`

固定字段为：

- `schema_version`、`run_id`、`status`；
- `requested_target_hit_rate`、`effective_target_hit_rate`、`theoretical_hit_rate`；
- `target_difference_pp`、`target_signed_difference_pp`、`target_absolute_difference_pp`，其中 `target_difference_pp` 当前等于绝对偏差；
- `validation`：`status`、`target_reachable`、`warning_only`、`affects_exit_code`；
- `theory`：`input_tokens`、`hit_tokens`、`groups`、`dp`；每个组或 DP 统计包含 `input_tokens`、`hit_tokens`、`hit_rate`；
- `warnings`：零个或多个告警。`TARGET_UNREACHABLE` 包含 requested target 和可达上下界，`TARGET_DEVIATION` 包含 `difference_pp`。

成功生成时 `status="prepared"`；`run` 完成后为 `"complete"`，`analyze` 复算后为 `"analyzed"`。在线阶段新增 `runtime.metrics_baseline/metrics_after`（含原始 Prometheus 文本和分 DP 累计值）、`actual`、`theory_actual_*_difference_pp` 及 `validation.actual_status`。偏差告警只改变展示值，不改变成功退出码。

`run` 在正式跑分期间按 `service.poll_interval_seconds` 周期轮询 `metrics_url` 采样 KV 用量（kv 是瞬时 gauge，跑分结束后会归零，必须在跑分期间采样），结果写入 `runtime.kv_cache_polling`：`interval_seconds`、`count`、`summary`（每 DP 的 `peak`/`avg`/`sample_count` 及 `global_peak`/`global_avg`）与逐样本明细 `samples`（`elapsed_seconds` + 每 DP 用量）。同时把峰值/均值合并进 `actual`：`actual.by_dp.*.kv_cache_usage_peak`、`kv_cache_usage_avg`、`actual.global_kv_cache_usage_peak`、`global_kv_cache_usage_avg`；`actual.by_dp.*.kv_cache_usage` 仍为 after 快照的瞬时值。`poll_interval_seconds` 设为 0 可关闭轮询；单次抓取失败只跳过该样本，不中断跑分。

### `inspect` Manifest 和 CLI 返回字段

`inspect` 终端 JSON 包含：`run_id`、`mode`、`requested_target_hit_rate`、`effective_target_hit_rate`、`theoretical_hit_rate`、`reachable_min`、`reachable_max`、`target_reachable`、`group_reachability`、`groups`、`input_tokens`、`output_tokens`、`dp_route_counts`、`sends_requests`、`log`、`manifest`。其中 `sends_requests` 固定为 `false`。

inspect 写出的轻量 Manifest 顶层包含 `schema_version`、`plugin_version`、`status`、`run_id`、`scenario_path`、`scenario_sha256`、`effective_config` 和 `inspect`。`status` 固定为 `"inspected"`；`inspect` 包含 `timestamp`、`base_run_id`、`base_output_dir`、`sends_requests` 与 `summary`。prepare 原位升级后，该文件变为完整正式 Manifest，`status="prepared"`，并包含请求、组、DP、warmup 和 artifacts 等正式字段。

CLI 最后一段 JSON 的固定字段为：

- `prepare`：`full`、`requests`、`manifest`、`analysis`、`log`；
- `inspect`：上述 inspect 摘要、`log` 和 `manifest`；
- `validate`：`ok`、`rows`、`run_id`；
- `run`：更新后的完整 analysis JSON；
- `analyze`：离线复算后的完整 analysis JSON。validate/run/analyze 均写日志，返回 JSON 当前不单独添加 `log` 字段。

## 7. 退出码

- 理论与目标差异超过 `target_warning_pp`：`TARGET_DEVIATION`；
- 目标超出可达区间：`TARGET_UNREACHABLE`；
- 实际与理论差异超过 `actual_warning_pp`：`ACTUAL_DEVIATION`；
- 三者始终只告警，不改变原本成功的退出码；
- 配置错误、产物损坏、服务能力不足或 AISBench 执行失败会返回非零退出码。

## 8. 常见问题

### 目标命中率为什么不完全相等？

公共前缀按 `block_size` 对齐，cold 还受顺序、组、DP 路由和缓存水位约束。插件选择最接近的可达结果，并记录 requested、effective、theoretical 和偏差原因。

如果多个 Prefix Group 选择了相同的首个 GSM8K 样本，插件会先尝试轮换组内样本；所有轮换仍碰撞时才使用确定性的组标记兜底，避免小语料或重复 indices 让整个 prepare 直接失败。

### warmup 为什么不进入正式统计？

warmup 只负责建立缓存。如果计入请求数、吞吐、时延或命中率，正式结果会混入准备阶段成本。

### 修改 Scenario 后为什么通常不再需要手动改 run_id？

单独执行 `prepare` 时会使用新的秒级时间戳；执行推荐的 `inspect → prepare → run` 工作流时，后续命令通过匹配的 Manifest 复用同一时间戳。时间戳同时追加到 `run_id` 和 `output_dir`，因此不需要手动改名。`--overwrite` 仅用于明确重建同一时间戳目录。
