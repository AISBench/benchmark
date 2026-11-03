# VABench轻量UT与覆盖率

在AISBench开发环境中执行，不向评分环境强制添加测试依赖：

```bash
python3 -m pip install -r tests/UT/vabench/requirements.txt
python3 tests/UT/vabench/run_coverage.py
```

需要完整FFmpeg/ffprobe；可用`VABENCH_FFMPEG`和`VABENCH_FFPROBE`指定可执行文件。测试以真实临时媒体、真实评分子进程协议、模拟模型及系统包管理边界验证接入行为，不下载模型、不运行pip/apt安装、不要求GPU。

覆盖率包含分层后的`datasets/vabench.py`、`tasks/vabench_eval.py`、`tasks/vabench/*.py`、`summarizers/vabench*.py`、`utils/vabench/*.py`、`tools/vabench/*.py`以及旧`benchmark/vabench/*.py`兼容层，共23个Python模块（21个含可执行语句）。不是整个AISBench仓库或外部官方评分模型的覆盖率。环境准备、设备分支、CLI入口及兼容层均在分母中，没有因迁移而排除它们。

`coverage_bootstrap/sitecustomize.py`仅在本命令设置`COVERAGE_PROCESS_START`时启动测量，使真实worker子进程也计入数据；产品运行不导入这个测试钩子。运行器清除本范围旧数据、收集本次父子进程数据并合并，避免重复累计旧报告。

门槛：总语句覆盖率≥80%、总分支覆盖率≥80%、每个模块的语句＋分支合并覆盖率≥80%；任一未达到或UT失败则退出非零。

报告位于：

```text
test_reports/vabench/coverage.json
test_reports/vabench/coverage.xml
test_reports/vabench/htmlcov/index.html
```

2026-11-03本地测量：139 passed；总语句覆盖率94.56%、总分支88.49%、合并92.81%，最低模块89.83%。新增分层测试验证新旧命令、导入隔离、来源指纹和protocol打包。测试模型/设备接口的替身只证明控制流与协议，不证明H200推理、NPU支持或官方<1%对齐。
