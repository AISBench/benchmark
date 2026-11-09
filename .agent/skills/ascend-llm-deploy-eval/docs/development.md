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

历史记录（2026-11-04）：Windows 110 项中 107 项通过、3 项跳过，当时尚未完成真实 CLI 压测。此记录不表示当前验证状态。

历史修复、回归覆盖及当时验证边界见 [2026-11-05 修复记录](audit-repairs-20261105.md)。

## Startup ownership evidence

Generated start/probe scripts append host UTC time, NPU status, fuser PIDs for the deployment devices, PID cgroups and running container IDs/names to `logs/DEPLOYMENT-runNN/NODE-ownership.jsonl`. Each diagnostic command has a 3-second timeout. Missing tools, vanished processes and timeouts remain explicit unknowns, not proof of idle devices. No process arguments or environment are read, no resources are stopped, and diagnostic failure does not replace the deployment result. Synchronous sampling adds at most about 9 seconds per probe; no background monitor is created.


## 2026-11-09 综合检视修复验证

本轮全量 unittest：Linux / Python 3.11 共 214 项全部通过；Windows / Python 3.11 共 214 项，203 项通过、11 项平台专用测试跳过。Windows 新增跳过两项 Linux flock 分发验证；Linux 已执行这两项。新增回归覆盖显式 privileged、最终 shell 参数边界/JSON/布尔值、日志积压与轮转、事件游标、零输出与 reset 聚合证据、分发失败/强杀/竞争发布。命令为 `python -m unittest discover -s tests/unit -q`。

开发过程中 V1–V6 已完成真实 AISBench 小负载流程验证（含原生预热）；V6 实际请求只覆盖 P1，P2/P3 仅健康验证。上述场景不是本次最新代码的完整实机重跑。本轮只执行离线测试，不重新部署模型、不占用 NPU。

Ruff 0.17.4 已扫描 scripts/ 与 tests/：`python -m ruff check --isolated --select E9,F63,F7,F82 scripts tests` 正确性规则通过。另执行开发机现有广泛规则的 `python -m ruff check scripts tests`，仍存在导入排序、宽泛异常捕获、未使用导入等问题，不能标记完整 lint 全绿；没有为消除报告而批量改写历史生命周期实现。该广泛规则依赖开发机配置，不作为仓库统一 lint 基线。完整 lint 整理仍是待办。Git 空白检查及 skill 元数据校验单独执行，不能替代 lint 或真实部署验收。
