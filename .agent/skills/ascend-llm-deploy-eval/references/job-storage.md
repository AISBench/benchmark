# 任务存储

任务目录保存 job_spec.json、bindings.json、status.json、scripts、logs、results 和最小 .runtime。服务日志按部署运行编号组织，性能结果按 subtask 与 attempt 分开。目录必须未使用，拒绝覆盖。共享目录须验证可见性与摘要，不能仅凭相同路径判断共享。任务目录与运行时目录保留，资源释放不删除结果、镜像或权重。

完整参数与操作见[统一 CLI](../docs/compact-cli.md)。
