# 门禁矩阵与判定语义

## 判定四值

| 值 | 使用条件 | 报告要求 |
|---|---|---|
| `PASS` | 有命令输出或远端事实支持 | 必须附 `value` 或 `evidence` |
| `FAIL` | 有证据表明不成立 | 必须附 `remedy`（可执行、单一动作） |
| `UNKNOWN` | 本地无法判定 | 必须附 `reason`，并进入 `unverified` |
| `NOT_APPLICABLE` | 不适用 | 必须附适用性理由 |

**硬规则**：`PASS` 不得用于“没检查”。退出码 `0` 只表示无已知 `FAIL`；推进（`locally-verified`、`pr-open`、`ready-to-merge`）要求 `promotion_blockers` 为空。

## 本地可预检项

| 门禁 | 判定方式 | 脚本 |
|---|---|---|
| 体量 ≤ 1000 additions | `git diff --numstat merge-base..head` 求和，二进制单列 | `gate_check` |
| 关联 Issue | 正文匹配 `Fixes #N` / `#N` | `gate_check` / `pr_body check` |
| UT 变更 | 变更路径含 `tests/UT/`；feature 必选，bugfix 且 additions>50 必选 | `gate_check` / `ut_map` |
| DCO | `base..head` 每条提交含 `Signed-off-by:` | `gate_check` |
| 标题格式 | `[type][scope]subject` | `pr_body check` |
| 标签命中 | 已勾选项逐字匹配 workflow 映射 | `gate_check` / `pr_body check` |
| 文档双语 | `docs/source_zh_cn` ↔ `docs/source_en` 双向配对；`README.md` ↔ `README_en.md` | `gate_check` |
| 文档内容同步 | 无法自动判定 | → `UNKNOWN` |

## 必须远端/运行时证据的项

| 门禁 | 为什么本地不行 |
|---|---|
| ≥1 条外部 review | GitHub 上的 review 事实，作者无法自造 |
| 行内 thread 全 resolve | 需要 PR 与 thread 状态 |
| 全量覆盖率 | 目标子集不能代表全量 80%/80%/60% |
| Read the Docs 构建 | 外部构建环境 |
| 自托管冒烟 | 需要 runner 上的服务与硬件 |

这些一律 `UNKNOWN`，在任何报告与最终答复中如实披露。

## 阈值来源

阈值**从仓库文件解析**，不硬编码：

- 体量上限、`tests/UT/` 前缀、bugfix 阈值 ← `.github/workflows/pr_quality_check.yml`
- 覆盖率阈值 ← `.github/workflows/run-ut-on-pr-py.yml`、`tests/pytest.ini`
- 标签映射 ← `.github/workflows/auto_label_pr.yml`
- 复选项 ← `.github/pull_request_template.md`

解析失败时使用默认值并写入 `warnings`，不允许静默猜测。

## 标签文案漂移

当模板复选项文案与 workflow 映射键不一致时：

1. 保留模板原文，只勾选用户选择的类型；
2. 输出 `pr_template_label_drift` 告警（受影响类型 + 双方文案）；
3. 受影响类型的 `labels` 判 `FAIL`，说明“按现有匹配规则无法获得期望标签”，而不是“GitHub 合入已失败”；
4. 不改模板、不改 workflow、不自动改标签；
5. 人工知情继续时，原始 `FAIL` 与告警必须保留在报告中。

## 文档双语

- 覆盖新增、修改、删除、重命名；双向检查。
- 路径配对与内容复核分开：前者脚本判定，后者 `UNKNOWN` 需逐项复核。
- 不提供跳过开关；`FAIL` 与 `UNKNOWN` 都阻止依赖文档证据的状态晋级。

---

## English

Verdicts are `PASS`, `FAIL`, `UNKNOWN`, `NOT_APPLICABLE`. `PASS` requires evidence and is never used for "not checked"; exit code `0` means "no known FAIL" only, and promotion requires an empty `promotion_blockers` list.

Locally decidable: size limit, issue reference, `tests/UT/` presence (feature, or bugfix above the threshold), DCO sign-off, title format, label mapping hit, and documentation path pairing. Remote- or runtime-only: external review, review-thread resolution, full-suite coverage, Read the Docs builds, and self-hosted smoke — all reported as `UNKNOWN` and disclosed.

Thresholds are parsed from the repository workflows and `pytest.ini`, never hardcoded; failed parsing falls back to defaults plus a warning.

Label drift is reported, not patched: the template wording is preserved, the affected type fails with an explicit "cannot obtain the expected label under current matching rules" message, and the original failure stays visible if a human chooses to continue. Documentation parity is enforced in both directions, and content sync stays `UNKNOWN` until reviewed item by item.
