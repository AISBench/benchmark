# 仓库画像（repo-profile）

启动阶段现读，不记忆、不假设、不复用上一次会话的结论。

## 读取顺序与提取项

| 顺序 | 文件 | 提取 |
|---|---|---|
| 1 | `docs/source_zh_cn/develop_guide/contributing.md` | 标题格式、DCO 要求、测试命令、覆盖率目标、重大变更阈值 |
| 2 | `.github/pull_request_template.md` | 章节、复选项原文、Checklist |
| 3 | `.github/workflows/auto_label_pr.yml` | `typeToLabel` 映射（只用于预测，不覆盖模板文案） |
| 4 | `.github/workflows/pr_quality_check.yml` | 体量上限、`tests/UT/` 前缀、bugfix 阈值 |
| 5 | `.github/workflows/run-ut-on-pr-py.yml` | 测试命令、覆盖率阈值、runner 标签 |
| 6 | `.github/workflows/run_smoke_test_pr_and_daily.yml` | 触发路径、environment 审批、超时 |
| 7 | `tests/pytest.ini`、`.coveragerc` | `fail_under`、omit、branch |
| 8 | `docs/source_zh_cn` / `docs/source_en` | 双语目录结构、README 互链形态 |
| 9 | `tests/UT/**` | 一级目录分类与命名风格 |

`gate_check.py --profile` 输出结构化画像：

```bash
python scripts/gate_check.py --profile --out /tmp/profile.json
```

## 画像失效条件

出现以下任一情况必须重读，不得复用：

- `HEAD` 变化；
- 上述任一文件在工作树中为 dirty；
- 目标仓库的 ruleset/分支保护发生变更（只能读远端判断）。

## 禁止

- 不得把画像写死进技能文档或脚本常量。
- 不得根据旧画像断言“当前仓库仍然是这么规定的”。
- 画像解析失败时**降级并告警**，不得静默套用默认值后当作事实。

---

## English

Read the repository profile at startup, every time: contribution guide, PR template, label workflow, quality gate workflow, test workflow, smoke workflow, `pytest.ini`/`.coveragerc`, bilingual documentation trees, and `tests/UT` layout. `gate_check.py --profile` emits the structured result.

Invalidate the profile whenever `HEAD` moves, any source file is dirty, or remote rulesets may have changed — then re-read instead of reusing cached facts. Never hardcode profile values in the skill, never assert current rules from a stale profile, and always degrade with a warning when parsing fails rather than silently trusting defaults.
