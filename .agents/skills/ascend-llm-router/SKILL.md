---
name: ascend-llm-router
description: 昇腾模型部署、性能测试及资源释放的任务入口；识别意图，选择 plan、deployment、eval、release，并根据阶段边界继续任务。
---

# Ascend LLM Router

依赖 deploy-eval-v2 CLI >=0.1.0a6、协议 1.0。先运行 `deploy-eval-v2 --version`。五个技能平级安装；缺少依赖时报告，不隐式切换到历史 ascend-llm-deploy-eval。

| 用户意图 | 阶段 |
| --- | --- |
| 部署并测试 | ascend-llm-plan → ascend-llm-deployment → ascend-llm-eval → ascend-llm-release |
| 仅部署 | ascend-llm-plan → ascend-llm-deployment，按授权保留 |
| 测试已有服务 | ascend-llm-plan → ascend-llm-eval，不取得外部服务管理权限 |
| 分析已有结果 | ascend-llm-eval |
| 释放已有任务 | ascend-llm-release |

由当前 Agent 加载下一技能，不递归调用技能或自行启动协调 Agent。已有授权持续有效。阶段交接保留 task_ref、request_id、operation_id、next_cursor、输入摘要、service/case 身份及证据引用。

收到 deployment.ready 后核对当前状态，再进入 eval；evaluation.completed 后按冻结策略进入下一 case 或 release。RUNNING 按游标等待；WAIT_TIMEOUT 不代表失败。结果未知时恢复查询，不更换请求身份盲重跑。具体参数、预检和清理规则由对应阶段技能负责。

共享命令格式和返回语义见 [CLI 契约](references/cli-contract.md)。仅在用户要求无模型验收时读取 [Linux socket 验证](references/socket-validation.md)。最终分别报告执行、证据完整性、性能有效性和清理状态，不把模拟结果当实机性能。
