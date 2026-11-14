# `develop`：分析、实现与本地验证

## 前置

- Issue 已核验（`verified`），范围已与用户确认；
- 工作树干净，或已明确本次要保留的本地改动；
- 已读 `references/gates.md` 与仓库画像。

## 步骤

1. **建分支**：从最新 canonical 默认分支创建 `feature/…`、`fix/…` 或 `docs/…`；禁止直接在默认分支开发。
2. **确认影响面**：列出将改动的文件、涉及的 `tests/UT/<sub>` 目录、是否需要双语文档。
3. **实现**：只做已确认范围内的改动；发现需要扩大范围时停止并回报，不擅自扩展。
4. **补测试**：
   ```bash
   python scripts/ut_map.py suggest --changed <改动文件...>
   ```
   按建议路径补 `tests/UT/<sub>/test_*.py`；feature 必补，bugfix 超过阈值必补。
5. **双语文档**：任何文档改动（新增/修改/删除/重命名）同步对侧语言，并逐项对齐语义。
6. **跑本地门禁并推进状态**：
   ```bash
   python scripts/devflow.py develop --pr-type <type> --issue <N>
   ```
   达标（无 blocker）即推进到 `locally-verified` 并写入账本；否则输出 blocker 与建议路径，按提示修复后重跑。
   子集覆盖率（可选，不安装依赖）：
   ```bash
   python scripts/ut_map.py coverage --test-dir tests/UT/<sub> --dry-run   # 先看命令
   ```
7. **记录证据**：命令、退出码、关键输出、`head_sha`、未运行项。

## 完成条件（`locally-verified`）

- 变更与确认范围一致；
- 受影响测试实际运行并有结果；
- 本地可判定门禁无 `FAIL`；
- 文档双语路径配对通过、内容已复核；
- `unverified` 列表已列出且如实说明（如 NPU 真机、全量覆盖率、外部构建）。

## 禁止

- push、建 PR（属于 `pr`）；
- 跳过 hooks、跳过测试、伪造覆盖率结论；
- 自动 `pip install` 补齐依赖；
- 为通过门禁而删测试、放宽阈值或改 CI 配置。

---

## English

`develop` requires a verified issue and a confirmed scope. Create a working branch from the latest canonical default branch (never work on the default branch), state the affected files and `tests/UT/<sub>` directories, implement only the confirmed scope, and add tests using `ut_map.py suggest`. Any documentation change is mirrored into the other language and reviewed item by item. Then run `devflow.py develop`, which judges the gates, records them in the ledger and advances to `locally-verified` only when no blocker remains; `ut_map.py coverage --dry-run` shows the scoped coverage command without installing anything.

`locally-verified` requires: changes match the agreed scope, affected tests actually ran, no locally decidable `FAIL`, documentation pairs present with reviewed content, and every unverified item disclosed (NPU hardware, full-suite coverage, external builds).

Prohibited: pushing, opening PRs, skipping hooks or tests, fabricating coverage, installing dependencies automatically, and weakening gates (deleting tests, lowering thresholds, editing CI) to make checks pass.
