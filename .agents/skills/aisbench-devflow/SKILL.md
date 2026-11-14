---
name: aisbench-devflow
description: AISBench 研发流程编排。把已授权 Issue 推进到本地实现、合规 PR、CI 通过与检视意见闭环，门禁满足后输出可合入清单交由人工合入。含 develop / pr / ci / feedback / review / dataset-onboard 等独立 action。只核验、不执行合入，不代替人工 Review，不代签 DCO/CLA，不改 PR 模板与标签工作流。
compatibility: 需要 Git 与 Python 3.10+；远端读写依赖已登录的 GitHub CLI（gh）或宿主提供的 GitHub 工具。运行期无第三方 Python 依赖。
---

# aisbench-devflow

## 定位与边界

本技能是 AISBench 仓库中**唯一覆盖写路径**的流程技能：建分支、改代码、跑本地门禁、提交、推送、创建 PR、回应检视意见。只读诊断与规范检查由既有技能负责，本技能只消费其结论：

| 技能 | 职责 | 本技能的关系 |
|---|---|---|
| `pr-checklist` | PR 规范检查（只读） | 规范判定以它为准，不重复实现 |
| `ci-failure-triage` | CI 失败诊断（只读） | `ci` action 的诊断引擎，本技能只做诊断后的修复 |
| `ascend-llm-deploy-eval` | 部署与压测执行 | 无交集 |

硬边界（任何模式、任何授权都不能突破）：

1. **永不执行合入**：不调用 merge API/CLI，不启用 auto-merge，不使用管理员绕过，不提供执行开关。
2. **永不代签**：DCO 使用用户已配置身份；CLA 由人签署；不伪造 reviewer 意见。
3. **永不改仓库规则**：不改 PR 模板、不改 `auto_label_pr.yml`、不自动改标签。
4. **永不把“没检查”写成通过**：判定只有 `PASS / FAIL / UNKNOWN / NOT_APPLICABLE`。

## Action 路由

| action | 输入 | 完成状态 | 主要禁止 | 必读 |
|---|---|---|---|---|
| `discover` | 仓库 + 标签/里程碑过滤 | `discovered` | 任何写 | `repo-profile.md` |
| `issue` | Issue URL/编号 | `verified` | 创建 Issue、代改他人负责状态 | `repo-profile.md`、`gates.md` |
| `plan` | 已核验 Issue | `planned` | 开始改代码 | `develop.md` |
| `develop` | 已确认方案 | `locally-verified` | push、建 PR | `develop.md`、`gates.md` |
| `pr` | 已本地验证分支 | `pr-open` | 触发 CI、处理 feedback | `pr.md`、`gates.md` |
| `ci` | PR 编号 | `ci-passed` | 扩大范围补开发（应回 `develop`） | `ci.md` |
| `feedback` | PR + 可写分支 | `feedback-resolved` / `waiting-for-review` | 批量 resolve、伪造回应 | `feedback.md` |
| `review` | 他人 PR | `review-findings` / `review-incomplete` | approve、改作者代码 | `review.md` |
| `merge` | PR 编号 | `ready-to-merge` / `waiting` / `blocked` | **一切合入操作** | `merge.md` |
| `dataset-onboard` | 数据集/后端规格 | `onboarded` | 编造指标、跳双语文档 | `dataset-onboard.md` |

`dataset-onboard` 是本技能内的 action，**不建第二个 Skill**：共享仓库画像、门禁与账本。

## 显式 action 契约

1. 只执行被指定的 action，达到完成条件立即停止。
2. 启动时按仓库事实重建状态，不相信调用方的阶段声明；ledger 只是缓存。
3. 前置不足 → `blocked` + 缺失证据 + 建议上游 action，**不自动补跑**。
4. 不执行下游 action：`pr` 创建成功后不触发 CI，`ci` 通过后不处理 feedback。
5. 为安全完成本 action 所必需的复验属于本 action（如 `pr` 提交前重跑受影响门禁）。

## 启动顺序

1. 确认仓库、reflog 干净的 remotes、源分支与目标分支。
2. `python scripts/gate_check.py --profile` 现读仓库画像（阈值、模板、标签映射、文档树）。
3. 校验能力梯度：`gh` typed 命令 → 宿主 GitHub 工具 → `gh api graphql` → 人工网页 → `blocked`。
4. 读 `.git/aisbench-devflow/ledger.json`（经 `git rev-parse --absolute-git-dir` 解析）并与事实交叉校验。
5. 判定当前门禁，按 §门禁执行。

## 门禁

完整矩阵、判定语义与补救方式见 [references/gates.md](references/gates.md)。可本地预检的项：

```bash
python scripts/gate_check.py --base origin/master --pr-type <type> --issue <N> \
  --title "<拟用标题>" --pr-body-file <body.md> --out gates.json
python scripts/pr_body.py check --body-file <body.md> --workflow .github/workflows/auto_label_pr.yml --issue <N>
python scripts/ut_map.py suggest --changed <改动源码...>
```

判定约定：

- `PASS` 必须有证据；`UNKNOWN` 表示本地不可判定（远端 review、thread 状态、全量覆盖率、外部构建、自托管冒烟）。
- 退出码 `0` 仅表示“无已知 FAIL”，**不代表可推进**；推进前要求 `promotion_blockers` 为空。
- 文档双语为强制项：`UNKNOWN` 不得被普通自主模式豁免。

## 状态机

```
discovered → verified → planned → branched → implemented → locally-verified
→ pr-open → ci-passed → waiting-for-review → feedback-resolved → ci-passed
→ review-approved → ready-to-merge
任一状态 → blocked（必须带 blocker 与建议上游 action）
```

每跳必须有证据，且绑定 `head_sha`；新 push 后旧结论立即作废。`onboarded` 与 `review-findings` 不进入作者主链。

## 全局不变量

- 不改 PR 模板与标签工作流；标签文案不一致只告警并保留原模板文案。
- 不 force push、不 `--no-verify`、不关 Issue、不删分支。
- 远端写前查幂等；写后回读校验；结果不确定时先查事实，不盲目重试。
- 生成正文末尾恰好追加一次 `<!-- aisbench-devflow -->`；机器协议 payload（`/pr_check`、`@issue_checker`）不加标记。
- 未验证项必须进最终报告 `unverified`，不许省略。
- 文档改动必须双语成对，且内容需逐项复核（路径配对不等于内容同步）。

## 输出

每次暂停或结束输出：`action`、当前状态、证据（命令 + 结果摘要）、`head_sha`、blocker、建议下一 action、需用户确认项、`unverified` 列表。

---

## English

`aisbench-devflow` is the write-path workflow skill for the AISBench repository: branch, implement, run local gates, commit, push, open a pull request and close review threads. Read-only skills stay authoritative for what they already own (`pr-checklist` for norms, `ci-failure-triage` for diagnosis, `ascend-llm-deploy-eval` for deployment).

Hard boundaries, regardless of mode or authorization:

1. Never merge — no merge API/CLI, no auto-merge, no admin bypass, no execute flag.
2. Never sign on the user's behalf — DCO uses the configured identity, CLA is human, reviews are never fabricated.
3. Never modify repository rules — the PR template and `auto_label_pr.yml` stay untouched.
4. Never report "not checked" as passing — verdicts are `PASS / FAIL / UNKNOWN / NOT_APPLICABLE` only.

Actions: `discover`, `issue`, `plan`, `develop`, `pr`, `ci`, `feedback`, `review`, `merge` (verify only), `dataset-onboard` (same skill, not a second one). Each action does exactly one thing, stops on completion, and returns `blocked` with the missing evidence plus the suggested upstream action instead of silently continuing.

Startup: resolve the checkout and remotes, read the repository profile with `gate_check.py --profile`, validate the capability ladder (`gh` → host tools → `gh api graphql` → manual → blocked), cross-check the ledger at the real Git directory, then judge the gates.

Verdict contract: `PASS` requires evidence; `UNKNOWN` means the gate cannot be decided locally (remote reviews, thread state, full-suite coverage, external docs builds, self-hosted smoke). Exit code `0` means "no known FAIL" only — promotion also requires an empty `promotion_blockers` list. Documentation changes must be bilingual as a hard requirement; `UNKNOWN` cannot be waived by an ordinary autonomous grant.

Global invariants: no template or workflow edits, no force push, no `--no-verify`, no closing issues, no branch deletion; check idempotency before remote writes and read back after them; append `<!-- aisbench-devflow -->` exactly once to authored bodies while keeping machine payloads clean; unfinished verification always appears in `unverified`.
