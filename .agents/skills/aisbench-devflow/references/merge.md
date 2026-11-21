# `merge`：只核验，永不执行

## 定位

`merge` 是本技能的**核验 action**，不是合入 action。这是产品边界，不是权限不足的降级。

以下任何情形都不得执行合入：具备写权限、用户授予普通 `autonomous`、用户已确认方案、CI 全绿、门禁全通过。

## 核验项

| 项 | 证据 |
|---|---|
| 当前 `head_sha` | PR 的 `headRefOid` |
| 必需 checks | 该 SHA 上的全部必需 check 结论 |
| 独立 review | 非作者提交的 review（不在本地伪造） |
| thread 状态 | 全部 `isResolved` |
| 本地门禁 | `gates.json`，且 SHA 与当前一致 |
| 文档双语 | 路径配对 `PASS`、内容已复核 |
| 规则可读性 | 分支保护/ruleset 若不可读，标 `UNKNOWN` |

## 结论

- `ready-to-merge`：核验范围内全部满足，且无 `UNKNOWN` 阻断项；输出可合入清单与证据链接。
- `waiting`：存在 pending check、等待 review 或未闭环 thread。
- `blocked`：存在已知失败，或必要证据不可取得。

三条都不表示 GitHub 最终许可，也不表示 PR 已合入。

## 禁止

- 调用合并 API/CLI，启用 auto-merge，使用管理员绕过；
- 提供 `--execute`、`--admin`、`--auto` 一类开关；
- 把 `ready-to-merge` 写成“已合入”；
- 代替维护者做出合入判断。

若外部已发生合入，可作为只读事实记录，但不得记为 DevFlow 执行的动作。

---

## English

`merge` verifies; it never merges. Having write permission, an ordinary `autonomous` grant, a confirmed plan, green CI, or all gates passing does not change this. The skill exposes no `--execute`, `--admin`, or `--auto` switch.

Verification covers: the current `head_sha`, required checks on that SHA, reviews submitted by someone other than the author, review-thread resolution, the local gate report with a matching SHA, bilingual documentation, and the readability of branch protection or rulesets (unreadable → `UNKNOWN`).

Conclusions are `ready-to-merge` (everything verifiable is satisfied with no blocking `UNKNOWN`), `waiting` (pending checks, awaiting review, or unresolved threads), or `blocked` (known failure or unavailable evidence). None of them means GitHub permits the merge or that the PR is merged. An externally completed merge may be recorded as a read-only fact but never as an action performed by this skill.
