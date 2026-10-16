# AISBench Prefix Cache 变长与指定命中率数据生成/压测和多模态数据生成

简体中文 | [English](README_en.md)

这是一个独立的 AISBench 插件，可生成两类数据集并比较 vLLM Prefix Cache 的理论与实际命中率：纯文本数据集支持可控变长的输入/输出长度、公共前缀比例、Prefix Group 分组和请求顺序，并可指定全局理论 Prefix Cache 命中率；多模态数据集基于 MMMU 图片生成固定图片场景。它同时提供离线的 `inspect` / `prepare` / `validate`，在线的 `run`，以及使用已保存 Prometheus 快照复算的 `analyze`。

固定图片的多模态理解数据构造与压测（1319 条单图 1080P、1319 条五图 720P、30-token GSM8K 文本、256-token 输出，以及 TTFT/TPOT/ITL 验收）见 [MULTIMODAL.md](MULTIMODAL.md)。

插件只增加 `plugins/prefix_cache` 下的新代码，不修改 AISBench 核心逻辑。

Scenario 示例见 [config_examples/scenario.example.json](config_examples/scenario.example.json)，完整字段说明见 [config_examples/scenario.example.md](config_examples/scenario.example.md)。

## 1. 安装前准备

运行环境需要满足：

- Python 3.10 或更高版本；
- **JSONL 语料**。每个非空行必须是 JSON 对象，并包含 `corpus.field` 指定的文本字段，默认是 GSM8K 数据集中的 `question`；若想用类似 GSM8K 的语料，需参考 GSM8K 格式。
- 在线 `run` 还需要一个启用 Prefix Cache 的 vLLM Completions 服务。

依赖安装分两种场景（见第 2 节）：

- 仅离线生成数据集（`inspect` / `prepare` / `validate` / `analyze`）：安装插件本身即可，`transformers`、`datasets`、`aiohttp` 等依赖随插件自动安装；
- 在线压测（`run`）：需要完整安装 AISBench 及服务化压测依赖，参考 [AISBench 工具安装文档](../../docs/source_zh_cn/get_started/install.md)。

数据生成时使用的 tokenizer 必须与 vLLM 服务端模型一致。否则本地计算的 token 长度、Block 边界和理论命中率会与服务端实际行为不一致。

## 2. 安装步骤及每条命令的作用

以下命令假设当前目录是 AISBench 仓库根目录，即包含 `setup.py`、`ais_bench/` 和 `plugins/` 的目录。

### 2.1 仅安装插件（离线数据生成）

```bash
conda create --name ais_bench python=3.10 -y
conda activate ais_bench
pip3 install -e ./plugins/prefix_cache
```

只使用 `inspect`、`prepare`、`validate`、`analyze` 等离线命令时，安装插件本身即可。这条命令会：

1. 安装 `ais_bench_prefix_cache` Python 包及其依赖（含 `transformers`、`datasets`、`aiohttp`）；
2. 注册 Prefix Cache Dataset、Inferencer 和 vLLM API Model 的 AISBench 插件入口；
3. 安装 `ais-bench-prefix-cache` 命令行入口。

### 2.2 完整安装（在线压测 `run`）

`run` 需要以 `perf` 模式启动 AISBench 正式压测，请先按 [AISBench 工具安装文档](../../docs/source_zh_cn/get_started/install.md) 安装 AISBench 及服务化压测依赖（推荐使用 Conda 管理环境）：

```bash
pip3 install -e ./ --use-pep517
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
ais-bench-prefix-cache --help
```

各命令的作用：

- `pip3 install -e ./ --use-pep517`：以 editable 模式安装当前 AISBench 仓库及其依赖；`-e` 表示直接引用工作区源码，后续修改源码后通常不需要重新安装；
- `pip3 install -r requirements/api.txt`、`pip3 install -r requirements/extra.txt`：服务化压测（vLLM 等 API 服务）所需的 AISBench 依赖。

### 2.3 验证安装

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
- `aisbench.dataset`、`aisbench.model`：AISBench Dataset/Model 的全部用户可见配置；包括 abbr、reader 契约、prompt、pred_role、attr、stream、retry、batch_size 和 generation_kwargs。用户无需编辑插件 `config.py`。

`inspect`、`prepare` 和 `validate` 不访问服务；`run` 会使用 `service` 段完成探活、reset、逐 DP warmup、正式压测和指标采集。当前只支持一个 HTTP 入口后面的单 DP 或多 DP，不支持多个彼此独立的 vLLM 实例。

各参数的默认值、约束和模式见 [Scenario 参数说明](config_examples/scenario.example.md)。下面给出完整字段索引和最关键的数据构造参数，README 本身可作为快速使用手册。

Scenario 中省略的字段会使用 `scenario.example.json` 的当前值作为默认值，包括默认 run、tokenizer、GSM8K 路径、100 条固定 1024-token 请求、`output.output_key=null`（requests 不带输出长度字段）、warmup 60% 目标、单一 uniform Prefix Group 和 DP 2。`minimum_non_shared_length` 是安全例外：它按 `seed_blocks × block_size` 动态推导；使用示例默认值时仍为 16。

### 3.2 Prefix Cache 数据构造参数

Scenario 采用严格白名单，未列出的字段会被拒绝；省略的字段使用示例默认值。各字段逐项含义见 [Scenario 完整字段说明](config_examples/scenario.example.md)。

| 配置路径 | 简短说明 |
|---|---|
| `schema_version` | Scenario 配置格式版本。 |
| `run` | 任务标识、随机性和输出控制。 |
| `run.run_id` | 任务名称；执行时会追加时间戳。 |
| `run.random_seed` | 数据生成和随机选择的全局种子。 |
| `run.output_dir` | Prefix Cache 产物基础目录。 |
| `run.overwrite` | 是否允许覆盖同名正式产物。 |
| `tokenizer` | Tokenizer 加载与 Block 配置。 |
| `tokenizer.path` | Tokenizer 模型或目录路径。 |
| `tokenizer.block_size` | Prefix Cache 的 token Block 大小。 |
| `tokenizer.revision` | Tokenizer 仓库版本；`null` 表示默认版本。 |
| `tokenizer.trust_remote_code` | 是否信任 Tokenizer 自定义代码。 |
| `corpus` | 自然后缀语料配置。 |
| `corpus.path` | JSONL 语料文件路径；语料格式需与 GSM8K 一致。 |
| `corpus.field` | JSONL 中的问题文本字段名，默认 `question`。 |
| `corpus.selection` | 语料样本选择规则。 |
| `corpus.selection.mode` | 选择模式：`random`（按 `run.random_seed` 确定性打乱）、`indices`（按零基行号选择）、`question_sha256`（按规范化 question 的 SHA-256 选择）、`mixed`（先加入 `indices`，再加入 `question_sha256`）。 |
| `corpus.selection.values` | 非 mixed 模式的行号或哈希值列表；指定样本不足时按已选顺序循环复用。 |
| `corpus.selection.indices` | mixed 模式的零基行号列表。 |
| `corpus.selection.question_sha256` | mixed 模式的问题 SHA-256 列表；与 `indices` 不能同时为空。 |
| `requests` | 正式请求数量和长度分布。 |
| `requests.count` | 正式请求总数。 |
| `requests.input_length` | 输入 token 长度生成规则；所有长度必须为正整数，数量必须等于对应请求数。 |
| `requests.input_length.mode` | 输入长度模式：`fixed`（固定长度）、`explicit`（显式长度列表）、`range`（一个或多个闭区间采样）、`truncated_normal`（截断正态分布，只接受 `[min,max]` 内的整数采样）、`csv`（从 CSV 文件读取）。 |
| `requests.input_length.value` | `fixed` 模式的固定长度。 |
| `requests.input_length.values` | `explicit` 模式的长度列表；元素数量必须等于对应请求数。 |
| `requests.input_length.ranges` | `range` 模式的区间列表；所有区间 `count` 之和必须等于对应请求数。 |
| `requests.input_length.ranges[].min` | 单个采样区间下界（含）。 |
| `requests.input_length.ranges[].max` | 单个采样区间上界（含）。 |
| `requests.input_length.ranges[].count` | 单个区间生成的请求数。 |
| `requests.input_length.min` | `truncated_normal` 分布下界；`min=max` 等价于固定长度。 |
| `requests.input_length.max` | `truncated_normal` 分布上界。 |
| `requests.input_length.mean` | `truncated_normal` 均值；默认取区间中点。 |
| `requests.input_length.std` | `truncated_normal` 标准差；默认按区间宽度推导，显式设置时必须大于 0。 |
| `requests.input_length.path` | `csv` 模式的长度文件路径；需包含 `input_prompt_tokens`、`content_tokens` 或 `input_tokens` 中的一列，行数必须等于对应请求数。 |
| `requests.output_length` | 最大输出 token 长度生成规则；所有长度必须为正整数，数量必须等于对应请求数。 |
| `requests.output_length.mode` | 输出长度模式：`fixed`（固定长度）、`uniform`（区间均匀采样）、`truncated_normal`（截断正态分布）、`csv`（从 CSV 文件读取）。 |
| `requests.output_length.value` | `fixed` 模式的固定长度。 |
| `requests.output_length.min` | `uniform` / `truncated_normal` 模式下界。 |
| `requests.output_length.max` | `uniform` / `truncated_normal` 模式上界。 |
| `requests.output_length.mean` | `truncated_normal` 均值；默认取区间中点。 |
| `requests.output_length.std` | `truncated_normal` 标准差。 |
| `requests.output_length.path` | `csv` 模式的长度文件路径；列名必须为 `output_tokens`，行数必须等于对应请求数。 |
| `output` | 精简请求文件的字段控制。 |
| `output.output_key` | 可选输出长度字段名：`null`（默认，每行只有 `question`、`answer`）、`"max_tokens"` 或 `"output_tokens"`（追加第三字段，值来自该请求的最大输出 token 数）；`full.jsonl` 始终保留 `max_tokens`，省略不影响运行。 |
| `prefix_cache` | Prefix Cache 数据和运行策略。 |
| `prefix_cache.mode` | 缓存模式：`cold`（每个 `(group_id, dp_rank)` lane 从零缓存水位开始，同一组的请求按组内出现顺序 round-robin 路由到各 DP rank）或 `warmup`（为每个 `Prefix Group × DP rank` 生成预热计划写入 Manifest 的 `warmup.plan`，`run` 在正式 baseline 之前定向预热；warmup 请求不写入 `requests.jsonl`，也不进入正式统计）。 |
| `prefix_cache.target_hit_rate` | 期望的全局 Prefix Cache 命中率。 |
| `prefix_cache.seed_blocks` | 每条请求唯一 seed 占用的 Block 数；seed 长度为 `seed_blocks × block_size`，seed token 序列全局唯一，防止公共前缀结束后继续误共享。 |
| `prefix_cache.minimum_non_shared_length` | 每条请求至少保留的非共享 token 数；默认等于 seed 长度，不能小于 seed 长度；公共前缀最大长度为 `floor((input_length - minimum_non_shared_length) / block_size) × block_size`。 |
| `prefix_cache.groups` | Prefix Group 数量、分配和覆盖配置；每个 Group 独立生成 canonical 前缀并统计理论命中率。 |
| `prefix_cache.groups.count` | Prefix Group 总数。 |
| `prefix_cache.groups.assignment` | 请求到 Group 的分配规则。 |
| `prefix_cache.groups.assignment.mode` | Group 分配模式：`uniform`（尽量均匀分配）、`zipf`（用 `exponent` 控制热点集中程度）、`weights`（用 `weights` 提供各组相对权重）。 |
| `prefix_cache.groups.assignment.exponent` | `zipf` 分布热点指数。 |
| `prefix_cache.groups.assignment.weights` | `weights` 模式的各组相对权重。 |
| `prefix_cache.groups.overrides` | 各 Group 的可选独立覆盖。 |
| `prefix_cache.groups.overrides.group-N.input_length` | 指定 Group 的输入长度规则；数量必须等于该组实际请求数。 |
| `prefix_cache.groups.overrides.group-N.output_length` | 指定 Group 的输出长度规则；数量必须等于该组实际请求数。 |
| `prefix_cache.groups.overrides.group-N.corpus_selection` | 指定 Group 的语料选择规则。 |
| `prefix_cache.order` | 正式请求排列规则；理论命中率始终按最终发送顺序重新模拟。 |
| `prefix_cache.order.strategy` | 排列策略：`sequential`（保持生成顺序）、`within_group_shuffle`（每个 Group 内确定性打乱）、`interleave`（不同 Group 按轮次交错）、`global_shuffle`（全局确定性打乱）、`input_len_asc`（每个 Group 内按输入长度从短到长排序，再按组轮转交错）。要模拟“无预热、短请求到长请求逐步建立 Cache”，使用 `cold` + `input_len_asc`。 |
| `service` | 在线推理服务与指标采集配置。 |
| `service.inference_url` | vLLM 推理接口地址。 |
| `service.metrics_url` | Prometheus 指标接口地址。 |
| `service.reset_url` | Prefix Cache 重置接口地址。 |
| `service.model` | 请求体中的服务模型名。 |
| `service.dp_size` | 单实例内部 DP rank 数量。 |
| `service.assume_empty_cache` | 无重置接口时是否假定缓存为空。 |
| `service.engine_label_map` | Prometheus engine 标签到 DP rank 的映射。 |
| `service.timeout_seconds` | 探活、预热和指标请求超时秒数。 |
| `service.api_key` | 推理服务鉴权密钥；不会明文落盘。 |
| `service.poll_interval_seconds` | 正式压测期间 KV 指标采样间隔；设为 `0` 可关闭。 |
| `validation` | 结果偏差告警阈值。 |
| `validation.target_warning_pp` | 理论值偏离目标的告警百分点。 |
| `validation.actual_warning_pp` | 实际值偏离理论值的告警百分点。 |
| `aisbench` | AISBench 正式压测启动配置。 |
| `aisbench.config` | AISBench Python 配置模板路径。 |
| `aisbench.work_dir` | AISBench 结果基础目录。 |
| `aisbench.extra_args` | 附加 CLI 参数键值对象；参数名映射到单值、多值列表或布尔开关；离线命令不消费，`run` 会展开参数并启动 AISBench perf。 |
| `aisbench.dataset` | Dataset reader 和 Prompt 映射配置；为保证理论 token 与实际 prompt 一致，`input_columns`、`output_column`、`prompt_template` 三个数据契约字段必须保持示例值，`abbr` / `pred_role` 可改。 |
| `aisbench.dataset.abbr` | Dataset 展示简称；`null` 时自动生成。 |
| `aisbench.dataset.input_columns` | Dataset reader 的输入列。 |
| `aisbench.dataset.output_column` | Dataset reader 的参考答案列。 |
| `aisbench.dataset.prompt_template` | 正式请求的 Prompt 模板。 |
| `aisbench.dataset.pred_role` | 预测结果的角色名称。 |
| `aisbench.model` | AISBench API Model 配置；旧 Scenario 省略时自动使用示例默认值。 |
| `aisbench.model.abbr` | Model 展示简称；`null` 时自动生成。 |
| `aisbench.model.attr` | 模型属性；当前只能为 `"service"`，用于启用 AISBench service 性能链路和 TTFT 等指标汇总。 |
| `aisbench.model.stream` | 是否使用 SSE 流式响应。 |
| `aisbench.model.max_out_len` | Model 层的兜底最大输出长度。 |
| `aisbench.model.retry` | API 请求失败重试次数。 |
| `aisbench.model.batch_size` | AISBench API 最大并发基值。 |
| `aisbench.model.generation_kwargs` | 透传给 vLLM 的生成参数。 |

#### 可达性与长度统计

求解器会同时计算：

- 全局 `reachable_min` 和 `reachable_max`；
- 每个 Prefix Group 的 `reachable_min` 和 `reachable_max`；
- `target_reachable`，表示目标命中率是否处于全局可达区间；
- requested、effective 和 theoretical hit rate；
- 理论值减目标值的带符号偏差和绝对偏差。

目标高于 `reachable_max` 或低于 `reachable_min` 时，求解器直接选择对应边界解；目标位于区间内时，按 Block 单位选择最近可达命中量。最终 hit token 精度是硬约束，在此基础上再优化请求过程：warmup 按累计输入比例均衡分配前缀；cold 的小型、不规则场景使用受限 beam search，大型且 lane 内长度单调的场景按 `(Prefix Group, DP rank)` 分解，先按容量和输入 token 分配目标，再构造逐步增长的 lane 水位。后者不再枚举每条请求的全部前缀候选，每条独立 lane 的累计命中率平缓收敛到其目标，不会先明显超过目标再用短前缀回调。

设 `N` 为请求数、`cap_i` 为第 `i` 条请求可用的 Prefix Block 数、`T` 为生成 prompt 的输入 token 总数。内部规划路径如下：

| 模式/条件 | 规划路径 | 规划复杂度与行为 |
| --- | --- | --- |
| warmup | 累计输入比例分配 | `O(N)`；精确命中 Block 对齐后的目标。 |
| cold 且 `256 × Σ(cap_i + 1) <= 2,000,000` | 受限 beam search | 最多保留 256 个状态；状态转移总量受阈值限制。 |
| cold、大于上述阈值且 lane 内长度/容量非递减 | scalable lane 构造 | 以线性扫描和余数排序为主，`O(N log N)`、`O(N)` 空间。 |
| cold、大于阈值但 lane 不满足 scalable 前提 | exact lane fallback | 保证最终 hit token 精确，并在 `trajectory.issues` 中记录具体回退原因。 |

`inspect` 只执行规划，不处理全部 prompt token；`prepare` 还必须完成 tokenizer round-trip 和 JSONL 写入，所以端到端复杂度约为 `O(T + N log N)`，实际墙钟时间取决于 tokenizer、CPU、磁盘和 prompt 总长度，不作为固定性能承诺。1818 请求、总输入 156,377,088 token、`block_size=128` 的大规格验证场景估算为 312,524,800 次 beam 状态转移，因此会直接选择 scalable 路径，而不会执行完整 beam 枚举。

严格单调并非所有顺序都可实现，例如某条请求没有足够前缀容量。cold 模式按真实缓存共享边界分别检查每个 `(Prefix Group, DP rank)` lane；不同 lane 的首次请求必然 miss，不再把这种合法的全局累计回落误报为轨迹异常。某一 lane 本身无法平滑或求解器进入精确回退时，`trajectory.status` 记为 `BEST_EFFORT`，`issues` 说明原因，同时追加只告警的 `TRAJECTORY_BEST_EFFORT`；满足约束时状态为 `PASS`。

Manifest 的输入和输出长度摘要包含 `min`、`max`、`mean`、`p50`、`p90`、`p95`、`p99` 以及最多十个长度分桶。`inspect` 也会展示这些摘要和组级可达范围。

#### 验证状态与退出码

`analysis.json` 使用 `PASS` 或 `PASS_WITH_WARNING` 展示验证状态：

- 目标超出可达区间时记录 `TARGET_UNREACHABLE`；
- 理论值与目标相差超过 `target_warning_pp` 时记录 `TARGET_DEVIATION`。
- 请求顺序或容量无法满足平滑轨迹时记录 `TRAJECTORY_BEST_EFFORT`。

这些状态和差异只用于展示，`warning_only=true` 且 `affects_exit_code=false`。在线实际值与理论值超过 `actual_warning_pp` 时也只记录 `ACTUAL_DEVIATION`；配置、产物、服务能力或 AISBench 执行错误才会失败。

### 3.3 `inspect`：检查配置和理论范围

```bash
ais-bench-prefix-cache inspect --scenario ./scenario.json
```

作用：

- 只解析并校验 Scenario，生成确定性的长度、分组、顺序和 cold DP 路由计划，不加载 tokenizer 或 GSM8K；
- 直接计算目标可达范围和共享前缀理论计划，不构造 prompt、不写临时 full/requests/analysis；
- 展示 requested/effective/theoretical hit rate；
- 展示组分布、输入/输出长度和 cold DP 路由摘要；
- 不访问 vLLM、不发送请求，也不在 Scenario 的 `output_dir` 留下四类正式数据产物；
- 与 `prepare` 一样生成 `_YYYYMMDD_HHMMSS` 时间戳，并把详细日志缓存到 `output_dir_时间戳/log/<run_id_时间戳>.inspect.log`；
- 成功后在 `output_dir_时间戳/result/` 写入同名轻量 `manifest.json`，其 `status="inspected"`，并把 inspect 摘要保存在 `inspect.summary`；不会再生成 `<output_dir>.inspect.json`；
- 输出的 JSON 摘要包含 `log` 和 `manifest` 路径，可直接定位日志与检查结果。

`planning_only=true` 表示这是配置级理论规划；`content_validation_deferred_to_prepare=true` 表示语料选择、canonical 前缀构造以及 tokenizer round-trip 校验留到 `prepare` 执行。

### 3.4 `prepare`：生成正式数据产物

```bash
ais-bench-prefix-cache prepare --mode text --scenario ./scenario.json
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
ais-bench-prefix-cache prepare --mode text --scenario ./scenario.json --overwrite
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

完整流程为：读取显式 `prepare` 生成的 Manifest 并自动识别 text/mm 模式；校验本时间戳的产物；逐 DP 探活；reset Prefix Cache（或按配置记录 `ASSUME_EMPTY_CACHE`）；warmup 模式按每个 `Prefix Group × DP rank` 定向预热；预热完成后采集正式 baseline；运行 AISBench `perf`；采集 after 并计算每 DP、全局实际命中率；最后把 `runtime`、`actual`、理论/实际差值和告警写回 `result/<run_id>.analysis.json`。warmup 在 baseline 之前完成，因此不进入正式吞吐、时延或命中率统计。

`run` 的 Prefix Cache 插件流程日志只写入 `output_dir_时间戳/log/<run_id_时间戳>.run.log`，不会作为插件日志回显到 CLI 终端。text 模式结束时，stdout 先打印 AISBench 风格的 `[时间] [ais_bench_prefix_cache] [INFO] Prefix Cache Results of task [run_id]:` 标题，再以 `Prefix Cache Metric | Value` 两列表格展示总体目标命中率、总体理论命中率、总体实际命中率、理论与实际命中率的绝对偏差、理论与目标命中率的绝对偏差；表格后输出 `[INFO] Detailed analysis is available at: <path>`，完整 analysis 仍落盘在 `result/<run_id_时间戳>.analysis.json`。mm 模式 stdout 输出 AISBench 多模态任务结果 JSON，性能指标可通过 `report` 查看。AISBench 子进程继续继承 stdout/stderr，其运行过程会实时展示在 CLI 中。插件日志覆盖执行上下文、产物复用、precheck、reset、每个 Group × DP warmup、baseline/after 指标、AISBench 静态配置与启动命令、KV 周期采样、每 DP query/hit 差值、全局命中率及告警。日志只记录 Prompt 长度和 SHA-256，不打印 Prompt 正文、API key、Authorization Header 或原始请求体。

Dataset、Model 和 Inferencer 运行在 AISBench 正式任务子进程中，统一使用 AISBench 的 `AISLogger` 和 `ais_bench` handle，详细过程使用 debug 级别。默认 `aisbench.work_dir="./outputs/default"` 时，AISBench 会把该任务 stdout/stderr 写入 `./outputs/default/<AISBench时间戳>/logs/infer/*.out`；若显式修改 `aisbench.work_dir`，日志会随之移动。当前 AISBench 全局日志级别默认为 INFO，只有将其设为 DEBUG 时这些详细日志才会实际写入 `.out`。

正式 AISBench 请求默认使用 vLLM SSE 流式响应，因为 `aisbench.model.stream` 默认为 `true`。流式模式会在请求开始、首个响应 chunk 以及后续 chunk 到达时记录时间点，供 `DefaultPerfSummarizer` 计算 TTFT、TPOT、ITL、E2EL 和吞吐量；这些性能汇总文件位于 `aisbench.work_dir/performances/<model-abbr>/`。设为 `false` 后仍可统计 Prefix Cache 命中率，但无法按 chunk 生成完整 TTFT、TPOT 和 ITL。逐 DP 探活和插件 warmup 使用独立的非流式请求，它们发生在正式 baseline 之前，不会混入上述性能指标。

插件的 Group × DP warmup 与 AISBench 自带的 `--num-warmups` 是两套独立机制。前者由 `prefix_cache.mode="warmup"` 控制并发生在 baseline 之前；后者属于 AISBench perf 子进程。如果需要让正式 baseline 之后只包含正式请求，应配置 `"extra_args": {"--num-warmups": 0}`。`extra_args` 使用“CLI 参数名 → 参数值”的对象结构，`run` 会将其展开为 `--num-warmups 0` 后传给 AISBench。

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

该命令不连接 vLLM、不运行 AISBench，只解析两份 Prometheus 文本，重新计算正式阶段计数器增量并写回 Manifest 对应的 analysis。当前 CLI 将 `analyze` 和 `validate` 的插件日志写入同一时间戳目录的 `log/<run_id>.validate.log`；连续执行时后一次命令会重新创建该日志文件。

`baseline.prom` 和 `after.prom` 不是 `prepare` 生成的数据集文件，而是同一个 vLLM 服务 `/metrics` 端点在正式统计窗口前后的完整 Prometheus 文本快照。手工采集时，应先完成 reset；warmup 场景还要先完成插件预热，然后在第一条正式请求前保存 baseline，并在最后一条正式请求完成后立即保存 after：

```bash
METRICS_URL="http://127.0.0.1:8000/metrics"
curl -fsS "$METRICS_URL" -o baseline.prom
# 在这里执行正式压测请求，期间不要重启 vLLM 或重置指标
curl -fsS "$METRICS_URL" -o after.prom
```

两份快照必须来自同一服务进程并覆盖全部 DP rank；多 DP 指标需保留 `engine` 标签，且 after 中的累计 queries/hits 不得小于 baseline。正常执行 `run` 时，插件已经自动采集这两个时点，并把原始文本写入 `<run_id>.analysis.json` 的 `runtime.metrics_baseline.raw_prometheus` 和 `runtime.metrics_after.raw_prometheus`，但不会额外生成 `.prom` 文件。如需离线复算，可从已有 analysis 导出：

```bash
ANALYSIS="./outputs/<run_id_时间戳>/result/<run_id_时间戳>.analysis.json"
jq -r '.runtime.metrics_baseline.raw_prometheus' "$ANALYSIS" > baseline.prom
jq -r '.runtime.metrics_after.raw_prometheus' "$ANALYSIS" > after.prom
```

`analyze` 会校验 Manifest 引用的数据产物，解析两份快照，按 DP 计算 `after - baseline` 的 queries/hits 增量，再汇总实际命中率并与理论值比较。它适用于历史结果复核、解析逻辑升级后的重算、外部压测流量分析和 CI 校验；成功的 `run` 本身已完成同样的在线差分分析，通常无需再次执行。

## 4. 推荐工作流

```bash
ais-bench-prefix-cache inspect --scenario ./scenario.json
ais-bench-prefix-cache prepare --mode text --scenario ./scenario.json
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
- `prefix_cache`：`mode`、`requested_target_hit_rate`、`effective_target_hit_rate`、`theoretical_hit_rate`、`reachable_min`、`reachable_max`、`target_reachable`、`minimum_non_shared_length`、`adjusted`、`reason`、`validation_status`、`target_signed_difference_pp`、`target_absolute_difference_pp`、`trajectory`（求解策略、轨迹状态和问题列表）；
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
- `trajectory`：`strategy`、`status`、`issues`；
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
- `run`：两列表格仅展示 5 个总体指标，并在下一行打印完整 `analysis.json` 路径；
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
