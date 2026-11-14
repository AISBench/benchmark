# `pr`：提交、推送与创建 PR

## 前置

- `locally-verified` 已达成，且证据绑定的 `head_sha` 仍是当前 `HEAD`；
- 目标分支与 fork 远端已确认；
- 已读 `references/gates.md`。

## 提交

```bash
git add <本次涉及的文件>
git commit -s -m "[type][scope]subject"
```

- `-s` 只用**用户已配置的** `user.name` / `user.email`；身份缺失时停止询问用户，不猜、不伪造。
- 不跳过 hooks；hook 失败则修复后创建新提交，不改写已推送历史。
- 提交信息按目标仓库贡献规范，不假定必须与 PR 标题完全一致。

## 推送

```bash
git push -u <remote> HEAD:<branch>
```

- 默认不 force push；确有必要时先征得用户明确同意。
- push 后记录新的 `head_sha`，此前所有门禁证据作废并按需重跑 `pr` 自身的复验。

## 生成正文

```bash
python scripts/pr_body.py render \
  --template .github/pull_request_template.md \
  --workflow .github/workflows/auto_label_pr.yml \
  --pr-type <type> --issue <N> --sections <sections.json> --out <body.md>
```

- 保留模板全部章节与检查项，只填内容；
- **只勾选一个** PR 类型，且使用模板原文；
- 标签映射不一致时保留原文并告警，不替换文案；
- 正文末尾恰好一次 `<!-- aisbench-devflow -->`；
- 未填的模板占位段落会在注释中列出，不得留作“已完成”的证据。

自检：

```bash
python scripts/pr_body.py check --body-file <body.md> \
  --workflow .github/workflows/auto_label_pr.yml --issue <N> --title "<标题>"
```

## 创建 PR

- 使用普通 PR（非 Draft）；
- 标题符合 `[type][scope]subject`；
- 正文关联 Issue；
- 创建后回读：标题、正文哈希、Issue 关联、`headRefOid`、base 分支。

## 完成条件（`pr-open`）

- PR 已创建并回读一致；
- 本地门禁无 `FAIL`，`promotion_blockers` 中只剩必须远端判定的项；
- `unverified` 已列出（review、thread、全量覆盖率、外部构建、冒烟）。

## 禁止

- 创建 Draft PR 或主动请求检视、触发检视机器人；
- 在 `pr` 内监控或修复 CI（属于 `ci`）；
- 关闭 Issue、删除分支；
- 为凑标签而改写模板文案或再勾选一个类型。

---

## English

`pr` starts from `locally-verified` whose evidence still matches the current `HEAD`. Stage only the intended files, commit with `-s` using the user's configured identity (stop and ask if it is missing; never fabricate it), and never bypass hooks or rewrite pushed history.

Render the body with `pr_body.py render` so the template structure is preserved, exactly one PR type is ticked using the template's own wording, the label drift is reported instead of patched, and `<!-- aisbench-devflow -->` appears exactly once. Unfilled placeholder sections are listed in a comment and are never treated as completed evidence.

Open a normal (non-draft) PR with a conforming title and an issue reference, then read back the title, body hash, issue link, `headRefOid` and base branch. `pr-open` completes when the read-back matches and no locally decidable gate fails. Do not monitor CI here, do not request reviews, and never rewrite template wording to force a label.
