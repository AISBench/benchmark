# 开发与测试

[返回快速开始](../README.md)

本 skill 的实现、文档和测试均位于当前目录内，可整体复制。以下命令从 skill 根目录执行：

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests/unit -v
```

测试通过模拟节点、临时目录与子进程检查规划、配置、打包、共享存储、执行生命周期和资源归属，不需要真实服务器或模型。Windows 会跳过标注为 Linux 专用的测试；完整检查应在 Linux 上执行。

真实部署和性能回归需要 Linux 昇腾服务器、Docker、NPU、可用镜像与权重以及免密 SSH，按使用文档进行有人值守的预检与执行。离线测试通过不代表真实模型性能测试通过。

`submit_job.py --local-jobs-dir` 可指定仓库外产物目录，默认使用用户主目录下的 `.autobenchmark/jobs`。

设计和决策文档来自原 AutoBenchmark 项目，带日期的验证记录及提交编号是历史背景，不表示本仓库的提交或本次验证结果。运行接口以 `SKILL.md` 和脚本为准。

## 原生 CLI 适配验证（2026-11-04）

配置映射测试使用本仓库 Prefix Cache Scenario 解析器验证生成配置；结果测试覆盖原生产物、失败退出、请求数量、token 长度及 DP/预热证据。已移除属于工具副本的单元测试，由原仓库插件测试维护其实现。真实 Linux 昇腾部署及性能回归仍需现场执行。

本次 Windows / Python 3.11 检查：107 项测试中 104 项通过，3 项 Linux 专用测试跳过；本地文档链接、Python 语法、JSON 解析与 Git 空白检查通过。未运行真实模型部署或性能回归。
