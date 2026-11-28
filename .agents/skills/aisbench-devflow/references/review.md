# `review`：检视他人的 PR（只读）

## 前置

- PR 作者不是当前账号（自审直接 `blocked: self-review-forbidden`）；
- PR 为 open 且非 Draft/WIP；
- 固定当前 `head_sha`，结论只对该 SHA 有效。

## 步骤

1. 读取完整 diff、全部变更文件、必要上下文、关联 Issue、仓库规范。
2. 只针对**本 PR 引入或暴露**的问题发 finding，每条包含：位置（path + 行）、现象、影响、依据、建议。
3. 已有 thread 覆盖的问题不重复发布。
4. 不 checkout、不执行 PR head 的脚本、不修改作者代码。
5. 输出分三类：`must-fix`、`suggestion`、`nit`（明确区分严重度）。

## 完成条件

- `review-findings`：已发布 finding，且每条可定位、可操作；
- `review-incomplete`：覆盖不足（例如 diff 被截断、必要文件无法读取），明确说明缺什么。

## 禁止

- approve / request changes / dismiss 他人 review；
- 代替作者回复或 resolve thread；
- 基于旧 `head_sha` 的结论；
- 把风格偏好当缺陷，或把无法验证的猜测写成 finding。

---

## English

`review` is read-only and only for PRs authored by someone else; self-review is blocked. Pin the current `head_sha` and read the full diff, changed files, needed context, the linked issue, and the repository norms.

Report only problems introduced or exposed by this PR. Each finding states location, symptom, impact, evidence, and a suggestion, and is graded `must-fix`, `suggestion`, or `nit`. Skip anything already covered by an existing thread, never check out or execute the PR head, and never modify the author's code. Close as `review-findings` when findings are published, or `review-incomplete` with the specific missing coverage — never approve, dismiss, reply on the author's behalf, or present style preferences and unverifiable guesses as defects.
