# Agent Skills

本目录收录用于 AISBench 相关工作流的 Agent skills。每个 skill 独立维护说明、脚本、示例、测试与文档。

| Skill | 用途 | 使用入口 |
|---|---|---|
| [ascend-llm-deploy-eval](skills/ascend-llm-deploy-eval/SKILL.md) | 昇腾模型部署、预检、性能测试与资源释放 | [快速开始](skills/ascend-llm-deploy-eval/README.md) |
| [ascend-llm-router](skills/ascend-llm-router/SKILL.md) | 新版任务入口、意图识别与阶段交接 | [CLI 安装与使用](ascend_llm_cli/README.md) |
| [ascend-llm-plan](skills/ascend-llm-plan/SKILL.md) | 现场绑定、审阅与输入冻结 | [计划与绑定](skills/ascend-llm-plan/references/plan-contract.md) |
| [ascend-llm-deployment](skills/ascend-llm-deployment/SKILL.md) | 部署、健康检查、冒烟与服务保留 | [部署技能](skills/ascend-llm-deployment/SKILL.md) |
| [ascend-llm-eval](skills/ascend-llm-eval/SKILL.md) | 原生性能评测、证据核验与独立评测收尾 | [评测技能](skills/ascend-llm-eval/SKILL.md) |
| [ascend-llm-release](skills/ascend-llm-release/SKILL.md) | 按资源归属核验并释放自有服务 | [释放技能](skills/ascend-llm-release/SKILL.md) |

五个新版技能共用 [ascend_llm_cli](ascend_llm_cli/README.md)，通过 `deploy-eval-v2` 的公开 JSON 契约协作；架构见 [DESIGN](ascend_llm_cli/DESIGN.md)。稳定版 `ascend-llm-deploy-eval` 独立保留，不随新版任务迁移或改写历史产物。

## 加载与使用

选择需要的 skill，按照所用 Agent 的加载机制安装或引用完整的 `skills/<skill-name>/` 目录，再按该 skill 的快速开始准备依赖和运行环境。`.agents` 是本仓库的组织目录，不保证被所有 Agent 自动发现。

## 目录约定

每个 skill 的 `SKILL.md` 描述触发条件与执行流程；`scripts/`、`references/`、`assets/`、`schemas/` 和 `examples/` 按需提供运行资源；`tests/` 与 `docs/` 保存该 skill 自身的测试和文档。新增 skill 使用独立目录，并在上表添加入口。

公共层仅保留导航与跨 skill 的通用约定。日志、现场绑定、模型权重和任务产物保存在仓库外。
