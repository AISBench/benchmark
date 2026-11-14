---
name: ascend-llm-router
description: 昇腾模型部署、性能测试及资源释放的任务入口；识别意图，选择 plan、deployment、eval、release，并根据阶段边界继续任务。
---

# Ascend LLM Router

依赖 deploy-eval-v2 CLI >=0.1.0a7、协议 1.0。先运行 `deploy-eval-v2 --version`。五个技能平级安装；缺少依赖时报告，不隐式切换到历史 ascend-llm-deploy-eval。

| 用户意图 | 阶段 |
| --- | --- |
| 部署并测试 | ascend-llm-plan → ascend-llm-deployment → ascend-llm-eval → ascend-llm-release |
| 仅部署 | ascend-llm-plan → ascend-llm-deployment，按授权保留 |
| 测试已有服务 | ascend-llm-plan → ascend-llm-eval，不取得外部服务管理权限 |
| 分析已有结果 | ascend-llm-eval |
| 释放已有任务 | ascend-llm-release |

由当前 Agent 加载下一技能，不递归调用技能或自行启动协调 Agent。已有授权持续有效。阶段交接保留 task_ref、request_id、operation_id、next_cursor、输入摘要、service/case 身份及证据引用。

收到 deployment.ready 后核对健康及冒烟证据；仅部署任务交付 service.json 并保留服务，只有用户要求评测时才进入 eval；evaluation.completed 后按冻结策略进入下一 case 或 release。RUNNING 按游标等待；WAIT_TIMEOUT 不代表失败。结果未知时恢复查询，不更换请求身份盲重跑。具体参数、预检和清理规则由对应阶段技能负责。

共享命令格式和返回语义见 [CLI 契约](references/cli-contract.md)。仅在用户要求无模型验收时读取 [Linux socket 验证](references/socket-validation.md)。最终分别报告执行、证据完整性、性能有效性和清理状态，不把模拟结果当实机性能。

从第一条现场/CLI命令开始保存命令、原始stdout/stderr和实际退出码。PowerShell Start-Transcript可能漏记原生命令stdout，不能只凭transcript存在宣称响应完整；把CLI JSON直接保存为独立文件并核对可解析性。阶段身份、游标、技能加载与下一CLI调用同步写检查点。停止runtime前归集日志；新CLI支持停止后的read-result只读回收，但不能把未回收完整的日志标为完整。

## 阶段边界驱动的技能加载
执行当前阶段时仅加载该阶段技能及其公开引用。不要在入口一次性预加载deployment/eval/release。plan完成冻结与readiness后实际读取deployment/SKILL.md再调用deploy；收到并核验deployment.ready之后，若用户要求评测，再实际读取eval/SKILL.md并调用eval；收到并核验evaluation.completed之后，按授权实际读取release/SKILL.md并调用release。仅部署成功保留服务，不读取eval启动负载或自动释放。外部评测不取得部署/服务释放权限。
即使某技能之前因其他原因读过，阶段交接时仍必须重新读取目标SKILL.md并记录当时的边界事件、游标、技能路径/摘要和随后的CLI请求，不能把预加载或事后写的JSON摘要当作实际加载动作。边界错误需要收尾时，实际读取release技能后按授权处理，不能假定之前读过即可。检查点保留真实工具调用证据。
