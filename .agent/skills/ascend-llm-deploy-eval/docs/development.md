# 开发与测试

实际容器依赖屏障会读取本节点生成的代理启动脚本末尾 `exec python3` 命令，核验已绑定的代理程序为可读文件，包括 `proxy_program_key` 选择的路径。此检查不执行代理或 bootstrap，不验证代理协议兼容性；缺文件、目录或读取失败会在模型启动前阻塞。相关覆盖位于 `tests/unit/test_container_dependencies.py`。

[返回快速开始](../README.md)

本 skill 的实现、文档和测试均位于当前目录内，可整体复制。以下命令从 skill 根目录执行：

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests/unit -v
```

测试通过模拟节点、临时目录与子进程检查规划、配置、打包、共享存储、执行生命周期和资源归属，不需要真实服务器或模型。Windows 会跳过标注为 Linux 专用的测试；完整检查应在 Linux 上执行。

真实部署和性能回归需要 Linux 昇腾服务器、Docker、NPU、可用镜像与权重以及免密 SSH，按使用文档进行有人值守的预检与执行。离线测试通过不代表真实模型性能测试通过。

`run.py --run-dir` 指定仓库外目录；旧公开提交入口已停用。

设计和决策文档来自原 AutoBenchmark 项目，带日期的验证记录及提交编号是历史背景，不表示本仓库的提交或本次验证结果。运行接口以 `SKILL.md` 和脚本为准。

## 标准 CLI 适配验证

测试覆盖外部 JSONL/CSV 数据路径、原生配置与命令生成、成功/失败退出、请求数量和长度校验、日志落盘索引及只读日志观察器。测试不需要 Prefix Cache 插件；不使用真实模型、数据集或服务器。Linux 专用进程检查与真实硬件回归需在相应环境执行。

2026-11-04，Windows / Python 3.11：110项测试，107项通过，3项Linux专用测试跳过。当前环境未安装完整AISBench运行依赖，尚未进行真实CLI压测、模型部署或硬件性能回归。

本轮修复、回归覆盖及实机验证边界见 [2026-11-05 修复记录](audit-repairs-20261105.md)。

## Startup ownership evidence

Generated start/probe scripts append host UTC time, NPU status, fuser PIDs for the deployment devices, PID cgroups and running container IDs/names to `logs/DEPLOYMENT-runNN/NODE-ownership.jsonl`. Each diagnostic command has a 3-second timeout. Missing tools, vanished processes and timeouts remain explicit unknowns, not proof of idle devices. No process arguments or environment are read, no resources are stopped, and diagnostic failure does not replace the deployment result. Synchronous sampling adds at most about 9 seconds per probe; no background monitor is created.
