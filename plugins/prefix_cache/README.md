# AISBench Prefix Cache Plugin

这是一个独立的 AISBench 插件，用于构造具有可控公共前缀的数据集，并对 vLLM Prefix Cache 的理论命中率和实际命中率进行压测分析。

插件只增加 `plugins/prefix_cache` 下的新代码，不修改 AISBench 核心逻辑。当前支持一个 vLLM HTTP 入口及其内部多个 DP rank，不支持多个彼此独立的 vLLM 实例。

Scenario 示例见 [config_examples/scenario.example.json](config_examples/scenario.example.json)，完整字段说明见 [config_examples/scenario.example.md](config_examples/scenario.example.md)。

## 1. 安装前准备

运行环境需要满足：

- Python 3.10 或更高版本；
- 当前 AISBench 仓库及其依赖可以正常导入；
- `transformers`：加载与 vLLM 模型一致的 tokenizer；
- `datasets` 和 `aiohttp`：AISBench 数据集与 API 压测依赖；
- 一份 GSM8K JSONL 文件，每行至少包含 `question` 字段；
- 一个已经启用 Prefix Cache 的 vLLM 服务。

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
2. 注册 AISBench 插件入口，使其发现自定义 Dataset、Inferencer 和 vLLM API Model；
3. 安装 `ais-bench-prefix-cache` 命令行入口。

同样使用 editable 模式，修改插件代码后不必重复安装。

### 2.4 验证安装

```bash
ais-bench-prefix-cache --help
```

作用：验证命令行入口是否安装成功，并列出 `inspect`、`prepare`、`validate`、`run` 和 `analyze` 子命令。

如果系统找不到该命令，可以使用等价形式：

```bash
python -m ais_bench_prefix_cache.cli --help
```

## 3. vLLM 服务要求

插件依赖三个服务能力：

- `/v1/completions`：发送 probe、warmup 和正式请求；
- `/metrics`：读取 Prefix Cache query/hit token 计数及 KV Cache 使用率；
- `/reset_prefix_cache`：在正式测试前清空缓存。

新版 vLLM 使用 reset 接口时可能要求设置 `VLLM_SERVER_DEV_MODE=1`。如果确实无法 reset，可以在 Scenario 中显式设置 `assume_empty_cache=true`；插件会继续并记录 `ASSUME_EMPTY_CACHE` 告警，但用户必须自行保证测试前缓存为空。

多 DP 还必须满足：

- 所有 DP rank 共用 Scenario 中的一个 HTTP 入口；
- 服务支持通过 `X-data-parallel-rank` Header 定向请求；
- `/metrics` 通过 `engine` 标签暴露全部 DP rank；
- 每个 DP rank 拥有独立 KV Cache。

任一 DP 无法定向或指标缺失时，预检查直接失败，不会降级为概率性预热或单 DP 统计。

## 4. 首次使用

### 4.1 复制并修改配置

```bash
cp ./plugins/prefix_cache/config_examples/scenario.example.json ./scenario.json
```

作用：复制一份可编辑的 Scenario。至少修改：

- `tokenizer.path`：与 vLLM 一致的 tokenizer；
- `corpus.path`：本地 GSM8K JSONL；
- `service.model` 和三个服务 URL；
- `service.dp_size`：真实 DP 数量；
- `aisbench.config`：AISBench Python 配置。

各参数含义见 [Scenario 参数说明](config_examples/scenario.example.md)。

### 4.2 `inspect`：检查配置和理论范围

```bash
ais-bench-prefix-cache inspect --scenario ./scenario.json
```

作用：

- 加载 tokenizer 和 GSM8K；
- 在临时目录构造数据并计算目标可达范围；
- 展示 requested/effective/theoretical hit rate；
- 展示组分布、输入/输出长度和 cold DP 路由摘要；
- 不访问 vLLM、不发送请求，也不在 Scenario 的 `output_dir` 留下产物。

### 4.3 `prepare`：生成正式数据产物

```bash
ais-bench-prefix-cache prepare --scenario ./scenario.json
```

作用：根据 Scenario 确定性生成并校验四个文件：

- `<run_id>.full.jsonl`；
- `<run_id>.requests.jsonl`；
- `<run_id>.manifest.json`；
- `<run_id>.analysis.json`。

默认不覆盖同名文件。确定需要重建时使用：

```bash
ais-bench-prefix-cache prepare --scenario ./scenario.json --overwrite
```

`--overwrite` 只覆盖该 run 对应的四个确定文件，不会清理整个输出目录。

### 4.4 `validate`：校验已有产物

```bash
ais-bench-prefix-cache validate --manifest ./outputs/gsm8k-prefix-cache-60/gsm8k-prefix-cache-60.manifest.json
```

作用：不生成数据、不访问 vLLM，只检查：

- Manifest、full 和 requests 行数是否一致；
- `sequence_index` 是否连续；
- requests 是否严格只含 `question`、`answer`、`max_tokens`；
- requests 与 full 是否逐行对应；
- full 和 requests 的 SHA-256 是否匹配 Manifest。

它用于发现文件被手工编辑、截断、换序或使用了错误版本。

### 4.5 `run`：执行完整压测

```bash
ais-bench-prefix-cache run --scenario ./scenario.json
```

完整流程为：

1. 产物不存在时自动执行 prepare；
2. 校验产物和 Scenario 哈希；
3. 探测推理接口、指标和全部 DP；
4. reset Prefix Cache，或记录显式的空缓存假定；
5. warmup 模式下逐 `Prefix Group × DP rank` 定向预热；
6. 预热后采集正式 baseline，使 warmup 不进入正式统计；
7. 调用 AISBench 执行正式性能压测；
8. 采集 after，计算分 DP 和全局实际命中率；
9. 将原始指标快照、差值和 warnings 写入 analysis。

临时覆盖 Scenario 中 AISBench 配置的方法：

```bash
ais-bench-prefix-cache run --scenario ./scenario.json --config ./my_prefix_cache_perf.py
```

`--config` 只影响本次执行，不修改 Scenario 文件。

### 4.6 `analyze`：使用离线指标重新分析

```bash
ais-bench-prefix-cache analyze \
  --manifest ./outputs/gsm8k-prefix-cache-60/gsm8k-prefix-cache-60.manifest.json \
  --baseline ./baseline.prom \
  --after ./after.prom
```

作用：读取保存好的两份 Prometheus 文本，重新计算正式阶段 query/hit token 增量、分 DP 命中率、全局 token 加权命中率和理论/实际差值。该命令不连接 vLLM，也不重新执行 AISBench。

## 5. 推荐工作流

正式测试建议分阶段执行：

```bash
ais-bench-prefix-cache inspect --scenario ./scenario.json
ais-bench-prefix-cache prepare --scenario ./scenario.json
ais-bench-prefix-cache validate --manifest <manifest路径>
ais-bench-prefix-cache run --scenario ./scenario.json
```

这样可以在发送任何请求前人工审计数据。`run` 发现已有且匹配当前 Scenario 的产物时会直接复用。

自动化环境可以直接执行：

```bash
ais-bench-prefix-cache run --scenario ./scenario.json
```

## 6. cold 与 warmup

### cold

- reset 后立即执行正式请求；
- 每个 `(group_id, dp_rank)` 从零水位开始；
- 同一组的请求按组内 round-robin 定向 DP；
- 插件保证同一 lane 内请求顺序；
- 可输出严格的分 DP 理论命中率。

### warmup

- reset 后对每个 Prefix Group、每个 DP rank 分别预热；
- 全部预热成功后才采集正式 baseline；
- warmup 不进入 requests JSONL、AISBench 性能数据、理论分母或正式指标增量；
- 正式请求交给 vLLM 内部负载均衡；
- 全局理论命中率有效，分 DP 主要展示实际指标。

## 7. 四类产物

### `<run_id>.full.jsonl`

完整审计数据：最终 prompt、输入/输出 token 数、组、公共前缀、唯一 seed、GSM8K 来源、DP 路由、理论命中 token 和水位变化。

### `<run_id>.requests.jsonl`

AISBench 最小输入，每行只包含 `question`、`answer`、`max_tokens`。DP 路由等字段由插件根据 full 文件合并，不污染通用请求格式。

### `<run_id>.manifest.json`

复现和校验入口：有效配置、输入哈希、tokenizer 指纹、组、DP、warmup 计划以及产物路径、大小和哈希。`api_key` 明文不会写入 Manifest，只记录是否配置。

### `<run_id>.analysis.json`

保存 requested/effective/theoretical hit rate、分组和分 DP 理论结果、baseline/after 原始指标、实际命中率、偏差及 warnings。

## 8. 命中率和退出码

全局实际命中率按 token 汇总：

```text
sum(所有 DP 的 hit token 增量) / sum(所有 DP 的 query token 增量)
```

不会对各 DP 百分比做简单平均。

- 理论与目标差异超过 `target_warning_pp`：`TARGET_DEVIATION`；
- 实际与理论差异超过 `actual_warning_pp`：`ACTUAL_DEVIATION`；
- 两者始终只告警，不改变原本成功的退出码；
- 配置错误、产物损坏、服务能力不足或 AISBench 失败会返回非零退出码。

## 9. 常见问题

### 目标命中率为什么不完全相等？

公共前缀按 `block_size` 对齐，cold 还受顺序、组、DP 路由和缓存水位约束。插件选择最接近的可达结果，并记录 requested、effective、theoretical 和偏差原因。

如果多个 Prefix Group 选择了相同的首个 GSM8K 样本，插件会先尝试轮换组内样本；所有轮换仍碰撞时才使用确定性的组标记兜底，避免小语料或重复 indices 让整个 prepare 直接失败。

### warmup 为什么不进入正式统计？

warmup 只负责建立缓存。如果计入请求数、吞吐、时延或命中率，正式结果会混入准备阶段成本。因此全部预热完成后会重新采集 baseline。

### 多 DP 为什么必须逐 DP warmup？

每个 DP rank 有独立 KV Cache。仅向入口重复发送无法保证每个 DP 都收到相同 Prefix Group，所以插件通过 Header 明确定向每个 rank。

### 修改 Scenario 后为什么 run 拒绝旧产物？

Manifest 保存 Scenario 哈希。配置与旧产物不一致时复用会破坏可复现性。请执行 `prepare --overwrite`，或设置 `run.overwrite=true` 让 run 自动重建。
