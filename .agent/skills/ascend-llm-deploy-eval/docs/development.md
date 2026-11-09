# 开发与测试

实际容器依赖屏障会读取本节点生成的代理启动脚本末尾 `exec python3` 命令，核验已绑定的代理程序为可读文件，包括 `proxy_program_key` 选择的路径。此检查不执行代理或 bootstrap，不验证代理协议兼容性；缺文件、目录或读取失败会在模型启动前阻塞。相关覆盖位于 `tests/unit/test_container_dependencies.py`。

[返回快速开始](../README.md)

本 skill 的实现、文档和测试均位于当前目录内，可整体复制。以下命令从 skill 根目录执行：

```sh
python -m pip install -r requirements.txt
python tests/run_unit_tests.py --output /path/outside/repo/unit-test-report.json
```

测试通过模拟节点、临时目录与子进程检查规划、配置、打包、共享存储、执行生命周期和资源归属，不需要真实服务器或模型。Windows 会跳过标注为 Linux 专用的测试；完整检查应在 Linux 上执行。

真实部署和性能回归需要 Linux 昇腾服务器、Docker、NPU、可用镜像与权重以及免密 SSH，按使用文档进行有人值守的预检与执行。离线测试通过不代表真实模型性能测试通过。

`run.py --run-dir` 指定仓库外目录；旧公开提交入口已停用。

设计和决策文档来自原 AutoBenchmark 项目，带日期的验证记录及提交编号是历史背景，不表示本仓库的提交或本次验证结果。运行接口以 `SKILL.md` 和脚本为准。

## 标准 CLI 适配验证

测试覆盖外部 JSONL/CSV 数据路径、原生配置与命令生成、成功/失败退出、请求数量和长度校验、日志落盘索引及只读日志观察器。测试不需要 Prefix Cache 插件；不使用真实模型、数据集或服务器。Linux 专用进程检查与真实硬件回归需在相应环境执行。

当前测试结果只维护在[机器生成的验证记录](unit-test-results.md)。旧计数不再作为当前验收依据；实际跳过项随操作系统与 Bash 可用性变化，以报告中的测试 ID 和原因核对。

历史修复、回归覆盖及当时验证边界见 [2026-11-05 修复记录](audit-repairs-20261105.md)。

## Startup ownership evidence

Generated start/probe scripts append host UTC time, NPU status, fuser PIDs for the deployment devices, PID cgroups and running container IDs/names to `logs/DEPLOYMENT-runNN/NODE-ownership.jsonl`. Each diagnostic command has a 3-second timeout. Missing tools, vanished processes and timeouts remain explicit unknowns, not proof of idle devices. No process arguments or environment are read, no resources are stopped, and diagnostic failure does not replace the deployment result. Synchronous sampling adds at most about 9 seconds per probe; no background monitor is created.


## 2026-11-09 综合检视修复验证

当前全量结果、Python/操作系统及每项跳过原因见[验证记录](unit-test-results.md)。报告脚本保留 unittest 的失败退出码，输出 JSON 包括源码摘要、Git 状态、计数及跳过原因；使用 `python tests/render_unit_test_report.py WINDOWS.json LINUX.json --output docs/unit-test-results.md` 生成展示记录，两份报告的测试源码摘要必须一致。

开发过程中 V1–V6 已完成真实 AISBench 小负载流程验证（含原生预热）；V6 实际请求只覆盖 P1，P2/P3 仅健康验证。上述场景不是本次最新代码的完整实机重跑。本轮只执行离线测试，不重新部署模型、不占用 NPU。

Ruff 0.17.4 已扫描 scripts/ 与 tests/：`python -m ruff check --isolated --select E9,F63,F7,F82 scripts tests` 正确性规则通过。另执行开发机现有广泛规则的 `python -m ruff check scripts tests`，仍存在导入排序、宽泛异常捕获、未使用导入等问题，不能标记完整 lint 全绿；没有为消除报告而批量改写历史生命周期实现。该广泛规则依赖开发机配置，不作为仓库统一 lint 基线。完整 lint 整理仍是待办。Git 空白检查及 skill 元数据校验单独执行，不能替代 lint 或真实部署验收。


## CI 回归保护

关联 benchmark 仓库的 `.github/workflows/run-ut-on-pr-py.yml` 对 `.agent/**` 及 workflow 自身变更触发。独立的 `skill_unit_tests` job 在 Linux runner 的临时虚拟环境安装本 skill 的 requirements.txt（含 jsonschema/packaging），并调用同一个 tests/run_unit_tests.py；测试失败直接使 job 失败，不依赖 AISBench 核心测试 job 的结果。JSON 报告通过 artifact 保存，runner 上未安装或无法安装依赖会明确失败，不静默跳过测试。

Bash 相关测试共享可执行性探测：优先 Windows Git Bash，再检查 PATH 候选，实际读取 BASH_VERSION。启动失败、无有效版本或超时均以原因明确的 SkipTest 记录，WSL launcher 存在不等于 Bash 可用。Linux CI 必须提供可执行 Bash；测试 runner 支持 --require-bash，CI 启用该选项以防 shell 覆盖静默缺失。
