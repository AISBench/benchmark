# Handoff — 生成 Prefix Cache 数据集分支 0825_generate_prefix_cache

## 1. 任务目标与背景

从 `0818_prefix_cache` 分支（HEAD = `fe5a0f9`「修改前缀方法」）中，提取与以下两个命令相关的代码，合成新分支 `0825_generate_prefix_cache`：

- `ais-bench-prefix-cache inspect --scenario ./scenario.json`
- `ais-bench-prefix-cache prepare --scenario ./scenario.json`

参考实现位于 `plugins/prefix_cache`，配套文档为 `README.md`、`MODULES.md`、`ARCHITECTURE.md`、`config_examples/scenario.example.md`。

原插件是完整的 prefix-cache 压测插件，包含 5 个子命令：`inspect` / `prepare` / `validate` / `run` / `analyze`。其中：

- `inspect` / `prepare` / `validate` 是**离线数据生成与校验**：只加载 tokenizer + GSM8K 语料，构造带公共前缀的数据集，不发任何网络请求。
- `run` / `analyze` 是**在线压测**：连接 vLLM、跑 AISBench 子进程、采集 Prometheus 指标。

新分支只保留离线三命令，得到一个专注于「生成 prefix-cache 数据集」的精简分支。

## 2. 关键决策

| 决策点 | 结论 |
|---|---|
| 代码范围 | 保留 `inspect + prepare + validate` 三个离线子命令；删除 `run`/`analyze` 及所有 AISBench 在线集成层 |
| 工作区处理 | 基于 HEAD 干净状态重建：丢弃未提交的 `scenario.example.json` 调试改动；不纳入 `outputs/`、`.superpowers/`、`scenario.example - 副本.json` 等未跟踪文件 |
| 依赖保留 | `ais-bench-benchmark`（`cli.py` 的 `AISLogger` 依赖）+ `transformers`（加载 tokenizer）；移除 `aiohttp`、`datasets` |
| `service` 段 | 保留（`scenario.py` 必填校验段；`dp_size` 用于 cold DP 路由）；删除仅在线使用的 `aisbench` 段 |
| 测试 | 保留 7 个离线单元测试；删除 3 个在线相关测试 |

## 3. 删除的文件

```
plugins/prefix_cache/ais_bench_prefix_cache/runtime.py
plugins/prefix_cache/ais_bench_prefix_cache/metrics.py
plugins/prefix_cache/ais_bench_prefix_cache/config.py
plugins/prefix_cache/ais_bench_prefix_cache/datasets/        (整个目录)
plugins/prefix_cache/ais_bench_prefix_cache/models/          (整个目录)
plugins/prefix_cache/ais_bench_prefix_cache/openicl/         (整个目录)
plugins/prefix_cache/config_examples/prefix_cache_perf.py
plugins/prefix_cache/tests/test_metrics.py
```

> 注意：`git rm` 后目录仍会残留未跟踪的 `__pycache__`，需再用 `rm -rf` 清理。

## 4. 修改的文件

| 文件 | 改动 |
|---|---|
| `ais_bench_prefix_cache/cli.py` | 删除 `runtime` 导入与 `run`/`analyze` 子命令及分发分支；`build_parser()` 只保留 `prepare`（`--scenario/--overwrite`）、`inspect`（`--scenario`）、`validate`（`--manifest`） |
| `ais_bench_prefix_cache/errors.py` | 删除 `RuntimeCapabilityError`；保留 `PrefixCacheError`、`ScenarioValidationError`、`ArtifactValidationError`、`PromptRoundTripError` |
| `setup.py` | description 改为「dataset generation and offline validation」；`install_requires = ["ais-bench-benchmark", "transformers"]`；entry_points 仅保留 `console_scripts`，删除 `ais_bench.benchmark_plugins` |
| `tests/test_pipeline.py` | 删除 `runtime` 导入与 3 个在线测试（`test_analysis_deviation_is_warning_with_zero_exit`、`test_render_aisbench_config_is_static`、`test_run_rejects_stale_prepared_scenario_before_network`）；保留 7 个离线测试 |
| `config_examples/scenario.example.json` | 删除 `aisbench` 段；恢复为 HEAD 版本 |

## 5. 保留的离线功能与依赖链

离线功能依赖链：

```
cli.py → pipeline.py (prepare_scenario / inspect_scenario)
       → scenario.py → errors.py
       → artifacts.py (write/read/validate) → errors.py
       → generation.py → errors.py
```

- `generation.py` 是纯算法层（除读 CSV/语料文件），确定性生成：语料选择、长度采样、分组、排序、DP 路由、前缀求解、唯一 seed、prompt 构造、理论命中率模拟。
- 种子派生顺序（复现性关键）：`seed`/`seed+1` 长度、`seed+2` 分组、`seed+300+group_index` 组内语料池、`seed+4` 顺序、`seed+5` 每请求 seed、`seed+6` warmup seed。
- Manifest 记录 `scenario_sha256`、`effective_config_sha256`、`corpus_sha256`、tokenizer 指纹与产物 SHA256。

## 6. 验证方式与测试结果

### 单元测试（离线部分应全部通过）

```bash
python -m pytest plugins/prefix_cache/tests/test_core.py plugins/prefix_cache/tests/test_pipeline.py -q
```

- 测试用 `FakeTokenizer`，不真实加载 transformers；
- 需在已 `pip install -e .` 的环境运行（`cli.py` 依赖 `ais_bench.benchmark`）。

### CLI 冒烟

```bash
python -m ais_bench_prefix_cache.cli --help   # 应只显示 prepare/inspect/validate
```

### 端到端（需真实 tokenizer + GSM8K 语料，可选）

```bash
python -m ais_bench_prefix_cache.cli inspect --scenario <scenario.json>
python -m ais_bench_prefix_cache.cli prepare --scenario <scenario.json>
python -m ais_bench_prefix_cache.cli validate --manifest <manifest.json>
```

### 无残留引用检查

```bash
grep -r "runtime\|metrics\|config\|datasets\|models\|openicl" plugins/prefix_cache/ais_bench_prefix_cache/
```

应无 `import` 残留。

## 7. 分支信息

- 源分支：`0818_prefix_cache`（HEAD `fe5a0f9`「修改前缀方法」）
- 新分支：`0825_generate_prefix_cache`
- 提交描述建议：中文，如「生成前缀缓存数据集：仅保留 inspect/prepare/validate 离线功能」

## 8. 注意事项

- `outputs/`、`.superpowers/`、`scenario.example - 副本.json` 等未跟踪文件**不纳入提交**，保持工作区原样。
- `service.api_key` 明文不会写入 Manifest（替换为 `api_key_configured` 布尔），但 Scenario 文件本身仍需限制权限。
- warmup 模式的预热计划（`manifest.json` 的 `warmup.plan`）由本分支生成，但实际预热请求的执行属于在线流程。
