# `feedback`：处理自己 PR 的检视意见

## 前置

- PR 存在且当前账号是作者；
- 本地分支可写，且与 PR head 一致（不一致先说明，不擅自改写）。

## 步骤

1. 读取全部 review thread：`isResolved`、路径、行号、评论正文、提出者。
2. 逐条处理，不做批量操作：
   - 采纳 → 改代码；
   - 部分采纳 → 说明取舍理由；
   - 不采纳 → 给出依据（规范、兼容性、性能数据）。
3. 每条结论都要有**可核对的回应**：指向改动行、测试结果或明确理由。
4. 有代码变更新增提交并 push；随后重新计算受影响门禁。
5. 逐条回复并 resolve：只在“已处理并有回应”后 resolve，不得先 resolve 再解释。
6. 无新增意见且无未闭环 thread 时进入 `waiting-for-review`，不主动索取检视。

## 完成条件（`feedback-resolved`）

- 每条既有 thread 都已处理、回复、并在适用时 resolve；
- 新 head 的受影响门禁已重跑；
- 报告列出逐条处置与证据。

## 禁止

- 批量 resolve 而不处理；
- 用“已修复”回应但无对应改动或测试；
- 擅自 dismiss 他人 review、或伪造 reviewer 认可；
- 修改他人评论内容。

---

## English

`feedback` requires an open PR authored by the current account and a local branch matching the PR head. Read every review thread (`isResolved`, path, line, body, author) and handle them one at a time: accept and change code, accept partially and explain the trade-off, or decline with evidence from the norms, compatibility, or measurements.

Every thread needs a checkable response — the changed line, a test result, or an explicit reason. Push follow-up commits, re-run the affected gates, then reply and resolve only after the handling and its response exist. Enter `waiting-for-review` when nothing new is pending; never solicit reviews. Batch-resolving without handling, claiming a fix without a change, dismissing other people's reviews, or editing someone else's comment is prohibited.
