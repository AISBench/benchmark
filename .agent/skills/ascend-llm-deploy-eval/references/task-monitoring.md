# 进度与日志观察

查看 status.json 获取 case、资源释放和产物汇集状态；logs/coordinator.log 保存协调决策，logs/DEPLOYMENT-runNN/NODE-INSTANCE.log 保存服务输出，results/SUBTASK-tryNN 保存 AISBench 输出和 worker 日志。

使用 `python scripts/watch_events.py --run-dir /work/JOB` 只读观察时间线；关闭观察器不控制任务。资源释放须根据 release 和 finalization 证据判断，不能由进程消失或中断信号推定成功。

完整参数与操作见[统一 CLI](../docs/compact-cli.md)。
