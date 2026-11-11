# Agent Skills

本目录收录用于 AISBench 相关工作流的 Agent skills。每个 skill 独立维护说明、脚本、示例、测试与文档。

| Skill | 用途 | 使用入口 |
|---|---|---|
| [ascend-llm-deploy-eval](skills/ascend-llm-deploy-eval/SKILL.md) | 昇腾模型部署、预检、性能测试与资源释放 | [快速开始](skills/ascend-llm-deploy-eval/README.md) |
| [pr-checklist](skills/pr-checklist/SKILL.md) | PR 贡献规范、说明材料与复审记录检查，不重复质量门禁 | [快速开始](skills/pr-checklist/README.md) |

## 加载与使用

选择需要的 skill，按照所用 Agent 的加载机制安装或引用完整的 `skills/<skill-name>/` 目录，再按该 skill 的快速开始准备依赖和运行环境。`.agents` 是本仓库的组织目录，不保证被所有 Agent 自动发现。

## 目录约定

每个 skill 的 `SKILL.md` 描述触发条件与执行流程；`scripts/`、`references/`、`assets/`、`schemas/` 和 `examples/` 按需提供运行资源；`tests/` 与 `docs/` 保存该 skill 自身的测试和文档。新增 skill 使用独立目录，并在上表添加入口。

公共层仅保留导航与跨 skill 的通用约定。日志、现场绑定、模型权重和任务产物保存在仓库外。
