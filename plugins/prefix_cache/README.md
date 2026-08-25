# AISBench Prefix Cache 数据集生成插件

这是一个独立的 AISBench 插件，用于**离线构造**具有可控公共前缀的 Prefix Cache 数据集，并校验产物的完整性。本分支只保留数据生成与校验能力（`inspect` / `prepare` / `validate` 三个命令），不包含任何在线压测（连接 vLLM、跑 AISBench、采集指标）相关功能。

插件只增加 `plugins/prefix_cache` 下的新代码，不修改 AISBench 核心逻辑。

Scenario 示例见 [config_examples/scenario.example.json](config_examples/scenario.example.json)，完整字段说明见 [config_examples/scenario.example.md](config_examples/scenario.example.md)。

## 1. 安装前准备

运行环境需要满足：

- Python 3.10 或更高版本；
- 当前 AISBench 仓库及其依赖可以正常导入；
- `transformers`：加载与 vLLM 模型一致的 tokenizer；
- 一份 GSM8K JSONL 文件，每行至少包含 `question` 字段。

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
2. 安装 `ais-bench-prefix-cache` 命令行入口。

### 2.4 验证安装

```bash
ais-bench-prefix-cache --help
```

作用：验证命令行入口是否安装成功，并列出 `inspect`、`prepare` 和 `validate` 三个子命令。

如果系统找不到该命令，可以使用等价形式：

```bash
python -m ais_bench_prefix_cache.cli --help
```

## 3. 首次使用

### 3.1 复制并修改配置

```bash
cp ./plugins/prefix_cache/config_examples/scenario.example.json ./scenario.json
```

作用：复制一份可编辑的 Scenario。至少修改：

- `tokenizer.path`：与 vLLM 一致的 tokenizer；
- `corpus.path`：本地 GSM8K JSONL；
- `service.model` 和三个服务 URL；
- `service.dp_size`：真实 DP 数量。

各参数含义见 [Scenario 参数说明](config_examples/scenario.example.md)。下面同时列出本插件最关键的数据构造参数，README 本身即可作为快速使用手册。

Scenario 中省略的字段会使用 `scenario.example.json` 的当前值作为默认值，包括默认 run、tokenizer、GSM8K 路径、100 条固定 1024-token 请求、warmup 60% 目标、单一 uniform Prefix Group 和 DP 2。`minimum_non_shared_length` 是安全例外：它按 `seed_blocks × block_size` 动态推导；使用示例默认值时仍为 16。

### 3.2 Prefix Cache 数据构造参数

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

理论命中率始终按照最终发送顺序重新模拟。cold 模式下，同一个 `(Prefix Group, DP rank)` lane 的正式请求会保持该顺序，即使 AISBench 并发发送也不会破坏理论缓存水位。

#### 可达性与长度统计

求解器会同时计算：

- 全局 `reachable_min` 和 `reachable_max`；
- 每个 Prefix Group 的 `reachable_min` 和 `reachable_max`；
- `target_reachable`，表示目标命中率是否处于全局可达区间；
- requested、effective 和 theoretical hit rate；
- 理论值减目标值的带符号偏差和绝对偏差。

目标高于 `reachable_max` 或低于 `reachable_min` 时，求解器直接选择对应边界解；目标位于区间内时，按 Block 单位选择最近可达命中量。cold 模式按 `(Prefix Group, DP rank)` lane 精确构造，不使用可能停在局部最优的爬山搜索。

Manifest 的输入和输出长度摘要包含 `min`、`max`、`mean`、`p50`、`p90`、`p95`、`p99` 以及最多十个长度分桶。`inspect` 也会展示这些摘要和组级可达范围。

#### 验证状态与退出码

`analysis.json` 使用 `PASS` 或 `PASS_WITH_WARNING` 展示验证状态：

- 目标超出可达区间时记录 `TARGET_UNREACHABLE`；
- 理论值与目标相差超过 `target_warning_pp` 时记录 `TARGET_DEVIATION`。

这些状态和差异只用于展示，`warning_only=true` 且 `affects_exit_code=false`。偏差不会改变原本成功的退出码；只有配置、产物生成或校验错误才会失败。

### 3.3 `inspect`：检查配置和理论范围

```bash
ais-bench-prefix-cache inspect --scenario ./scenario.json
```

作用：

- 加载 tokenizer 和 GSM8K；
- 在临时目录构造数据并计算目标可达范围；
- 展示 requested/effective/theoretical hit rate；
- 展示组分布、输入/输出长度和 cold DP 路由摘要；
- 不访问 vLLM、不发送请求，也不在 Scenario 的 `output_dir` 留下产物。

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

每次 `prepare` 会生成一次精确到秒的本地时间戳，并同时追加到 `run_id` 和 `output_dir` 最后一层目录。例如配置为：

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

因此连续执行时不需要手动修改 `run_id` 或 `output_dir`。

默认不覆盖同名文件。确定需要重建时使用：

```bash
ais-bench-prefix-cache prepare --scenario ./scenario.json --overwrite
```

`--overwrite` 只覆盖本次时间戳目录内该 run 对应的四个确定文件，不会清理整个输出目录。正常 CLI 执行每次都会使用新时间戳，因此通常不会遇到重名。

### 3.5 `validate`：校验已有产物

```bash
ais-bench-prefix-cache validate --manifest ./outputs/gsm8k-prefix-cache-60_<时间戳>/result/gsm8k-prefix-cache-60_<时间戳>.manifest.json
```

作用：不生成数据、不访问 vLLM，只检查：

- Manifest、full 和 requests 行数是否一致；
- `sequence_index` 是否连续；
- requests 是否严格只含 `question`、`answer`、`max_tokens`；
- requests 与 full 是否逐行对应；
- full 和 requests 的 SHA-256 是否匹配 Manifest。

它用于发现文件被手工编辑、截断、换序或使用了错误版本。

## 4. 推荐工作流

```bash
ais-bench-prefix-cache inspect --scenario ./scenario.json
ais-bench-prefix-cache prepare --scenario ./scenario.json
ais-bench-prefix-cache validate --manifest <manifest路径>
```

这样可以在任何实际压测前人工审计数据。`prepare` 遇到同名产物时默认拒绝覆盖。

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

> 本分支只负责生成 cold / warmup 模式下的数据与预热计划，不实际执行预热请求。预热计划落在 `result/<run_id_时间戳>.manifest.json` 的 `warmup.plan` 字段。

## 6. 分层产物

所有正式数据产物位于实际时间戳输出目录的 `result/` 下，详细日志位于同级 `log/` 下。

### `<run_id>.full.jsonl`

完整审计数据：最终 prompt、输入/输出 token 数、组、公共前缀、唯一 seed、每请求确定性随机种子、GSM8K 来源、DP 路由、理论命中 token、水位变化和差异块碰撞状态。

### `<run_id>.requests.jsonl`

最小输入，每行只包含 `question`、`answer`、`max_tokens`。DP 路由等字段由插件根据 full 文件合并，不污染通用请求格式。

### `<run_id>.manifest.json`

复现和校验入口：有效配置、输入哈希、tokenizer 指纹、输入/输出长度分位数和分桶、全局及组级可达范围、差异块唯一性、组、DP、warmup 计划以及产物路径、大小和哈希。`api_key` 明文不会写入 Manifest，只记录是否配置。

### `<run_id>.analysis.json`

保存 requested/effective/theoretical hit rate、目标可达性、带符号/绝对偏差、只展示不控制退出码的验证状态、分组和分 DP 理论结果及 warnings。

## 7. 退出码

- 理论与目标差异超过 `target_warning_pp`：`TARGET_DEVIATION`；
- 目标超出可达区间：`TARGET_UNREACHABLE`；
- 两者始终只告警，不改变原本成功的退出码；
- 配置错误、产物损坏会返回非零退出码。

## 8. 常见问题

### 目标命中率为什么不完全相等？

公共前缀按 `block_size` 对齐，cold 还受顺序、组、DP 路由和缓存水位约束。插件选择最接近的可达结果，并记录 requested、effective、theoretical 和偏差原因。

如果多个 Prefix Group 选择了相同的首个 GSM8K 样本，插件会先尝试轮换组内样本；所有轮换仍碰撞时才使用确定性的组标记兜底，避免小语料或重复 indices 让整个 prepare 直接失败。

### warmup 为什么不进入正式统计？

warmup 只负责建立缓存。如果计入请求数、吞吐、时延或命中率，正式结果会混入准备阶段成本。

### 修改 Scenario 后为什么通常不再需要手动改 run_id？

每次 `prepare` 都会把同一个秒级时间戳追加到 `run_id` 和 `output_dir`，正常连续任务不需要再手动改名。`--overwrite` 仍保留给显式复用同一执行时间戳的程序调用或特殊恢复流程。
