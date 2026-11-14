# `ci`：监控、诊断与最小修复

## 分工

只读诊断由 `ci-failure-triage` 负责；本 action 负责**诊断之后的修复**与重跑判定。不得在本技能内复制诊断规则。

## 步骤

1. 固定当前 `head_sha`，读取该 SHA 上的 checks 与结论。
2. 分类：
   - 全部 `success` / `skipped`（必需项）→ `ci-passed`；
   - 有 `pending` → `waiting`，说明等待项与已等待时长；
   - 有 `failure` → 进入诊断（交给 `ci-failure-triage`）。
3. 诊断后分类根因：
   - 本次改动引入 → 最小修复；
   - 仓库基线或基础设施问题 → `blocked`，附证据，不扩大范围；
   - 证据不足 → `blocked`，说明需要哪些日志/权限。
4. 修复循环（guided 下每次确认）：改代码 → 跑受影响本地测试 → 新提交 → push → 等新 `head_sha` 的 checks。
5. 新 push 后旧结论立即失效，不得复用。

## 完成条件（`ci-passed`）

- 当前 `head_sha` 上的必需 checks 全绿；
- 本地复验记录（改了哪些文件、跑了哪些测试）；
- `unverified` 列出无法本地等价的项（自托管冒烟、外部文档构建）。

## 禁止

- 用重跑代替根因分析；
- 在 CI action 内补做产品实现（应回 `develop`）；
- 绕过失败项（跳检查、改阈值、删测试、`--no-verify`）；
- 对超时/网络错误盲目重试——先查事实再决定。

---

## English

Diagnosis belongs to `ci-failure-triage`; this action only repairs and re-judges. Pin the current `head_sha`, read its checks, and classify: all required checks green → `ci-passed`; pending → `waiting` with the waiting items; failure → diagnose, then decide whether the root cause is this change (minimal fix), the repository baseline or infrastructure (blocked, with evidence), or insufficient evidence (blocked, naming what is missing).

The repair loop is: change code, run affected local tests, commit, push, wait for the new `head_sha`. Conclusions from the previous SHA are void after a push. Never use retries instead of root-cause analysis, never extend scope into product development here, never bypass a failing check, and never blindly retry timeouts or network errors — check the facts first.
