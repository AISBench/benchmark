# VABench轻量UT与覆盖率

在AISBench开发环境中执行，不向评分环境强制添加测试依赖：

```bash
python3 -m pip install -r tests/UT/vabench/requirements.txt
python3 tests/UT/vabench/run_coverage.py
```

需要完整FFmpeg/ffprobe；可用`VABENCH_FFMPEG`和`VABENCH_FFPROBE`指定可执行文件。测试以真实临时媒体、真实评分子进程协议、模拟模型及系统包管理边界验证接入行为，不下载模型、不运行pip/apt安装、不要求GPU。

覆盖率包含分层后的`datasets/vabench.py`、`tasks/vabench_eval.py`、`tasks/vabench/*.py`、`summarizers/vabench*.py`、`utils/vabench/*.py`、`tools/vabench/*.py`，共14个Python模块（12个含可执行语句，2个仅为包文档）。不是整个AISBench仓库或外部官方评分模型的覆盖率。环境准备、设备分支及CLI入口均在分母中；旧兼容模块已删除，而非通过覆盖率排除配置隐藏。

`coverage_bootstrap/sitecustomize.py`仅在本命令设置`COVERAGE_PROCESS_START`时启动测量，使真实worker子进程也计入数据；产品运行不导入这个测试钩子。运行器清除本范围旧数据、收集本次父子进程数据并合并，避免重复累计旧报告。

门槛：总语句覆盖率≥80%、总分支覆盖率≥80%、每个模块的语句＋分支合并覆盖率≥80%；任一未达到或UT失败则退出非零。

报告位于：

```text
test_reports/vabench/coverage.json
test_reports/vabench/coverage.xml
test_reports/vabench/htmlcov/index.html
```

2026-11-03本地测量：154 passed；总语句覆盖率95.77%、总分支89.70%、合并94.01%，最低模块91.18%。新增分层测试验证实际分层实现、新命令、无旧包、导入隔离、来源指纹和protocol打包。测试模型/设备接口的替身只证明控制流与协议，不证明H200推理、NPU支持或官方<1%对齐。

### 与仓库 PR CI 的统计区别

仓库 CI 使用 `python3 tests/run_tests.py tests/UT -p 16` 运行整个 UT，按全仓范围检查80%门槛；本目录运行器仅统计上述14个VABench模块。两者不能用同一个百分比替代。PR #636 的旧运行曾报告全仓79.95%，不能用独立VABench报告宣称该CI已经达标。

CI不调用本目录的子进程覆盖率启动器。只在评分子进程中经过的代码，不能保证被普通pytest-cov计入；同时CI未必存在相邻的官方VABench源码或FFmpeg，相关集成测试可能跳过。

因此新增`test_worker_execution.py`直接在pytest进程内验证模型调用、官方包装、异常/缺失/重复输出及设备参数；`test_catalog_offline.py`使用临时JSON/mapping/协议指纹验证精确匹配，不依赖外部仓库。这些测试由普通CI自动发现，无须修改CI配置或排除生产代码。真实子进程/媒体集成测试仍保留。

本地采用pytest-cov 7.0.0及仓库`tests/pytest.ini`普通父进程收集复测：154 passed，worker模块58.22%→95.96%；VABench模块合并93.91%，比补测前新增覆盖121条生产代码语句。该数值仍是VABench范围诊断，不是全仓覆盖率；全仓是否超过80%须以相同PR的后续CI结果确认。
