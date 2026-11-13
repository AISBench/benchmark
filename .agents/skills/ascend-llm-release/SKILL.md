---
name: ascend-llm-release
description: Deploy and Eval v2 的资源释放审阅、归属核验及结果交接；通过受归属守卫约束的 release/cancel 接口处理自有资源。
---

# Release

依赖 deploy-eval-v2 CLI 0.1 与协议 1.0，先执行 `deploy-eval-v2 --version` 核对。共享 CLI 契约随 ascend_llm_cli 包提供；缺失依赖时明确报告，不切换到旧版。使用已接入的 release/cancel 接口，实机释放仍需验收；不以手工 docker rm 绕过归属守卫。

先读取当前 task/service 身份、版本、活动操作及归属记录。只释放本任务明确拥有的资源；外部服务、共享权重/镜像及未知归属资源不进入默认释放范围。名称或 PID 相同不能证明所有权，须结合任务标签、容器身份和远端命令证据。

同服务测试与释放互斥；如果仍有活动操作，按授权使用 cancel 并等待真实清理结果，不把取消已接纳当释放成功。保留原 request_id 做幂等恢复；版本冲突先重读事实。

release.completed 后检查逐资源结果及 resource_results_ref。release.unverified、通信未知或部分失败时保留现场，停止复用资源，不删锁、不重放部署来试探。任务目录和证据默认保留，不随容器释放删除。

交付已释放、未释放、未知资源清单及证据位置。整体 CLI 退出码 0、operation.cancelled、task.expired 都不能代替资源释放核验。

## 本阶段参考与交付

依赖 CLI >=0.1.0a7、协议 1.0。资源归属与终态语义见 [生命周期边界](references/lifecycle-boundaries.md)，部分创建失败见 [创建前清理](references/precreate-cleanup.md)。共享命令格式见已安装 ascend-llm-router 的 references/cli-contract.md；停止观察 runtime 见 ascend-llm-deployment 的 references/launcher.md。

get-status 后按当前版本提交 release，活动测试需先按授权 cancel 并等待清理证据。资源释放核验完成后，使用原 task_ref 和 nonce 执行 stop-runtime，核对 runtime_stopped=true。runtime 停止通过认证 socket 请求完成，不猜 PID 或运行系统服务命令。保留任务日志、请求、冻结包和结果，交付已释放/未知资源清单。

失败任务仅在无活动操作且终态持久证据cleanup_status=VERIFIED时可stop-runtime；FAILED不自动代表仍占资源，也不代表已释放。RECOVERY_REQUIRED仍需先恢复核验，不能删锁或猜PID。
