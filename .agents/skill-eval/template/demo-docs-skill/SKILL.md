---
name: demo-docs-skill
description: 演示用「纯文档型 SKILL」：没有可执行脚本，只有规范与检查清单，用于验证测评框架不依赖 CLI 也能评「内容表达与格式规范」。
---

# Demo Docs Skill

## 适用场景

当某个 SKILL 的产出主要是**书面规范**（编写约定、评审清单），而不是可执行脚本时，
测评只能评「写得对不对、规不规范」，不需要、也不应该跑命令。

## 约定

1. `SKILL.md` 必须带 YAML frontmatter，且 `name` 与目录名一致。
2. 正文引用到的 `references/` 文件必须真实存在（断链会让 Agent 拿到空规则）。
3. 每条规则要可判定：写「必须显式给出 X」，不要写「尽量注意 X」。

## 检查清单

- [ ] frontmatter 完整（name / description）
- [ ] 相对链接可解析
- [ ] 规则可判定、无「视情况而定」这类模糊表述

细节见 [检查清单](references/checklist.md)。
