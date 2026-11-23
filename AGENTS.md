# AGENTS.md

本仓库为 Agent 提供技能（.agents/skills/），索引见 .agents/README.md。
技能分两类，按需加载对应目录的 SKILL.md，不要预读全部。

## 业务技能（使用 AISBench 完成任务时）
- 昇腾部署/压测/评测/释放 → skills/ascend-llm-deploy-eval 及新版五技能

## 贡献流程技能（向本仓库提交代码时）
- 提交 PR 前/复审时 → skills/pr-checklist/SKILL.md
- PR、分支差异或本地变更的证据驱动代码检视 → .agents/skills/evidence-pr-review/SKILL.md
- CI 失败排查 → skills/ci-failure-triage/SKILL.md
- 客户端协议适配兼容性疑问 → skills/check-api-compat/SKILL.md
- Issue→PR→CI→检视 全流程编排 → skills/aisbench-devflow/SKILL.md

## 贡献规则入口
PR 标题/DCO/模板/质量门禁以 CONTRIBUTING.md 与 .github/pull_request_template.md 为准。
技能只做检查与诊断，不替代质量门禁，不执行合入，不代签 DCO/CLA。
