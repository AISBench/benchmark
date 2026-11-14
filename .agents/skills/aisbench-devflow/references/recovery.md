# 恢复、幂等与账本

## 账本位置

```bash
git rev-parse --absolute-git-dir     # 不能假设 .git 一定是目录（worktree / submodule）
```

账本路径：`<absolute-git-dir>/aisbench-devflow/ledger.json`

- 不进工作树、不随 commit 分发、不污染 diff；
- 原子写入（临时文件 + 替换）；并发时用锁文件拒绝覆盖，不静默丢失；
- 只存最小证据与标识：仓库、任务身份、`head_sha`、门禁状态、`unverified`；
- 不存 Token、凭证、完整日志或绝对路径。

## 幂等键

| 写操作 | 幂等键 | 检查 |
|---|---|---|
| 建分支 | repo + branch | `rev-parse --verify` + `ls-remote` |
| commit | 工作树内容 | `status --porcelain` |
| push | sha + remote + branch | `ls-remote` 比对 |
| 建 PR | head + base + repo | 列出同 head 的 PR（含 closed） |
| 评论 / 回复 | pr + head_sha + 正文哈希 | 回读现有评论 |
| resolve thread | thread id | 读 `isResolved` |
| 触发 `/pr_check` | pr + head_sha | 读该 SHA 最新一次运行的结论 |

## 不确定时

远端写超时或网络失败 → **先查事实，不重试**：

1. 查到已生效 → 继续，记录“写后确认”；
2. 查到未生效 → 允许重试一次；
3. 仍不确定 → `blocked: write-uncertain`，输出目标标识（PR 号 / 评论 ID / thread ID）交人判断。

## 恢复流程

1. 读账本，标记为待校验；
2. 探测本地：分支、HEAD、dirty、remotes；
3. 探测远端：Issue、PR、checks、reviews、threads；
4. 逐项比对账本记录的 SHA，不一致置 `STALE`；
5. 由“最后一个证据仍成立的状态”重新进入；
6. 报告：上次停在哪、为什么、这次从哪继续。

---

## English

The ledger lives at `<absolute-git-dir>/aisbench-devflow/ledger.json`, resolved through `git rev-parse --absolute-git-dir` because `.git` may be a file (worktrees, submodules). It never enters the work tree, is written atomically, uses a lock file instead of silently overwriting, contains only minimal evidence and identifiers, and never stores tokens, credentials, full logs, or absolute paths.

Idempotency keys: branch (`repo + branch`), commit (`worktree state`), push (`sha + remote + branch`), PR (`head + base + repo`), comments (`pr + head_sha + body hash`), thread resolution (`thread id`), and `/pr_check` triggering (`pr + head_sha`).

On an uncertain remote write, verify first and retry at most once; if still unknown, return `blocked: write-uncertain` with the target identifiers. Recovery re-reads local and remote facts, marks ledger entries whose SHA no longer matches as `STALE`, resumes from the last state still backed by evidence, and reports where it stopped, why, and where it continues.
