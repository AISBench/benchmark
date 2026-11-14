# `dataset-onboard`：接入新数据集 / 模型后端

本 action 属于同一 Skill，不另建目录；共享仓库画像、门禁与账本。

## 清单（缺一不可）

| 步骤 | 产物 | 检查 |
|---|---|---|
| 1 数据集加载 | `ais_bench/benchmark/datasets/<name>.py` 中的 `Dataset.load` | 返回 `datasets.Dataset`；多模态按 `<AIS_TEXT_START>…` 标记拼接 |
| 2 导出 | `ais_bench/benchmark/datasets/__init__.py` | 类可被导入 |
| 3 （可选）评估器 | `Evaluator.score` | 返回 `{metric: score}` |
| 4 （可选）后处理 | `<name>_postprocess` | 输入输出均为字符串 |
| 5 配置 | `ais_bench/benchmark/configs/datasets/<name>/*.py` | `reader_cfg` / `infer_cfg` / `eval_cfg` / `abbr` |
| 6 数据集 README | `<name>/README.md` + `README_en.md` | 简介、部署、可用任务表格；双语互链 |
| 7 文档 | `docs/source_zh_cn/**` + `docs/source_en/**` | 成对，语义一致 |
| 8 单测 | `tests/UT/datasets/test_<name>.py` | 覆盖加载、解析、评分边界 |
| 9 （可选）冒烟 | `smoke_tests/test-case/**` | `case.yml` + `run.sh` + 配置 |

## 规则

- 指标与数据集规模必须来自可核实来源；不得编造或估算后写成实测。
- 不能真实下载数据集时，如实声明“未验证真实数据”，并用合成/最小样本覆盖解析逻辑。
- 双语文档与 README 是硬性要求，不得以“后续补”通过。
- 新增数据集目录后，`ut_map.py suggest` 用于确认测试目录与覆盖率范围。

## 完成条件（`onboarded`）

- 上述清单逐项有产物或明确的“不适用 + 理由”；
- `tests/UT` 用例实际运行；
- `gate_check.py` 本地项无 `FAIL`，`unverified` 列出未验证的真实数据/硬件项。

---

## English

`dataset-onboard` lives in this skill and reuses its profile, gates, and ledger. The checklist is: dataset loader returning a `datasets.Dataset` (multimodal joined with the `<AIS_*>` markers), export from `datasets/__init__.py`, optional evaluator and post-processor, configuration files with `abbr`, bilingual `README.md`/`README_en.md` (intro, deployment, task table, cross-linked), paired `docs/source_zh_cn` and `docs/source_en` pages, unit tests under `tests/UT/datasets/`, and optionally a smoke case.

Metric values and dataset sizes must come from verifiable sources — never estimated and presented as measured. If the real dataset cannot be downloaded, say so explicitly and cover parsing with synthetic or minimal samples. Bilingual documentation is mandatory and cannot be deferred. `onboarded` requires every item to have an artifact or an explicit "not applicable" with a reason, unit tests actually executed, no failing local gate, and all unverified real-data or hardware items listed.
