---
name: ascend-llm-deployment
description: Deploy and Eval v2 的部署阶段交接与事件处理，检查冻结输入、健康和冒烟证据；支持经授权的开发验收，实机验证待完成。
---

# Deploy

依赖 deploy-eval-v2 CLI 0.1 与协议 1.0，先执行 `deploy-eval-v2 --version` 核对。共享 CLI 契约随 ascend_llm_cli 包提供；缺失依赖时明确报告，不切换到旧版。执行器已接入，先通过 check-ready 和现场预检；max_retries 是已确认失败且清理核验后允许的最大重试次数；未知结果不重试，不能静默改动参数。按已有授权执行，不绕过 server 门控，不借用旧 FULL 代替本阶段。

部署前核对 plan_ref 的冻结输入、审阅摘要、绑定和 runtime 版本；首次 expected_version=0。不能覆盖已有任务产物。用唯一 request_id 保存请求文件，恢复时保留原业务内容和幂等键。

目标执行顺序：核验输入 → 主机级资源声明 → 分发与共享存储核验 → 占用检查 → 全部容器创建 → 全部依赖检查 → 模型/代理启动 → 健康 → 冒烟 → 保存部署证据。代理门控、完整 KV 参数、挂载与容器基线不能因拆分而丢失。

工具返回 RUNNING 就保存操作和游标继续等待。收到 deployment.ready 后核对 service_id、endpoint、model、ownership、部署摘要、健康/冒烟时间及保留策略。仅事件名不足以证明真实可用；模拟结果必须标为模拟。

失败检查 cleanup_status 和逐资源证据；部分创建的自有资源由 runtime 收尾，Agent 不按容器名称或 PID 猜测归属。结果未知时不重新创建容器。仅部署任务按已授权策略保留并交付释放方式；端到端任务把 service handle 交回路由选择 eval。

## 本阶段参考与交付

依赖 CLI >=0.1.0a6、协议 1.0。发布前读取 [现场验收](references/host-acceptance.md)，启动及身份恢复见 [Python runtime](references/launcher.md)。输入 Schema 由已安装 ascend-llm-plan 的 references/schemas/ 提供；共享请求格式见 ascend-llm-router 的 references/cli-contract.md。

接收冻结输入、runtime 包、control 配置及授权证据。依次执行 check-ready、publish-task、start-task、get-status、deploy；保存原 nonce 与摘要。runtime 使用独立 Python 进程，日志在任务目录，不依赖 systemd。返回 deployment.ready 后交付 service_id、健康/冒烟证据、当前版本及游标。未知启动结果查回执和状态，不重复启动。
