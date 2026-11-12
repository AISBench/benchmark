# 服务生命周期

启动前检查设备与端口占用；创建全部容器后完成全部依赖检查，再启动模型、代理，健康和冒烟通过后执行性能测试。

默认 after_case=keep_service、task_end=stop_container；可在 case 后停止容器，当前不支持 stop_service。复用服务前仍检查健康，重建服务有独立运行编号。缓存默认不清理；reset_cache=true 时逐模型实例核验空闲和清理成功。预热由 AISBench num_warmups 控制，不证明所有路由的缓存已覆盖。

仅释放任务身份匹配的资源。失败或中断进入收尾，分别记录释放、采集和状态写入结果；释放不明时不继续任务。任务目录不自动删除。

完整参数与操作见[统一 CLI](../docs/compact-cli.md)。
