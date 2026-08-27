# AISBench 接入 Harbor Agent 测评通用方案

> 目标：在 AISBench 中新增一条独立的 Agent 测评链路，通过 Harbor 框架执行任意 agent 的测评，
> 支持 Harbor 定义的全部数据集格式，统一用户参数入口，并实现依赖隔离。
>
> 状态：已确认设计决策
> - 监控服务：使用标准库 `http.server`
> - workflow：仅 `agent` / `agent_viz` 两个 mode

## 0. 版本对齐说明（重要）

- **参数定义唯一依据**：当前 harbor 源码 `/d:/group_dev/adapt_harbor/harbor`（**0.21.0**，见其 `pyproject.toml`）。所有 `AgentConfig` / `JobConfig` / `DatasetConfig` / `EnvironmentType` / 任务扫描行为均以该版本源码为准。
- **旧 `harbor_task.py` 基于 harbor 0.6.1**，实现时仅参考其结果落盘与断点续跑的外层流程，**不得沿用其已过时的 API 调用**，典型差异：
  - 旧实现 `AgentName(self.model_cfg.get("agent_name", "oracle"))` 直接构造枚举——0.21.0 中 `AgentConfig.name` 为 `str`，且需支持 `module:ClassName` 自定义 import path；新实现把原始字符串传给 `AgentConfig`，由 harbor `AgentFactory` 解析，仅对不含 `:` 的值校验是否属于 `AgentName.values()`。
  - 旧实现只用了 `AgentConfig(name/model_name/kwargs/env)` 四个字段——0.21.0 新增 `import_path / n_concurrent / skills / mcp_servers / include_logs / exclude_logs / resume_trajectory / load_trajectory / deps_path / override_timeout_sec` 等，需按需透传。
  - `requirements/datasets/harbor.txt` 锁定 `harbor==0.6.1`，**不适用于本方案**；`requirements/agent.txt` 改为安装最新 `harbor`（如需精确对齐本地 0.21.0 源码，可单独 `pip install -e <harbor绝对路径>`）。

---

## 1. 设计约束与兼容性要求

| 编号 | 要求 | 落点 |
|---|---|---|
| 1 | 在 `benchmark/ais_bench/benchmark/cli/workers.py` 的 `WORK_FLOW` 中新增 agent 独立 workflow，并继承 `BaseWorker` 创建 agent 专属 worker | 新增 `AgentEval` worker；`WORK_FLOW` 新增 `agent=[AgentEval, AccViz]`、`agent_viz=[AccViz]` |
| 2 | `benchmark/ais_bench/benchmark/runners` 新增 agent 专属 runner：拉起多个 harbor task + 定时监测 + 提供实时查询服务 | 新增 `HarborRunner` + `HarborMonitor`（存储）+ `HarborMonitorServer`（HTTP 服务） |
| 3 | `benchmark/ais_bench/benchmark/tasks/custom_tasks` 新增 harbor task，支持 harbor 全部 agent | 新增 `HarborAgentTask`（继承 `HarborTask`） |
| 4 | 因不同 agent 对同一含义参数（如 base url）传入方式不同，必须提供参数适配器 | 新增 `AgentParamAdapter` |
| 5 | 支持命令行直接配置常用参数 | `cli/argument_parser.py` 新增 `agent_args` 分组 |
| 6 | 依赖隔离：agent 测评独立一套依赖，仅安装 agent 所需依赖即可使用 | 新增 `requirements/agent.txt`；`runtime.txt` 与默认安装不动 |
| 7 | 结果落盘格式与现有 `harbor_task.py` 一致 | `HarborAgentTask` 复用 `_dump_eval_results` |
| 8 | 配置文件遵循 `models`（agent/模型服务参数）与 `datasets`（测评任务参数）划分 | 沿用并扩展原示例结构 |
| 9 | 改造不得影响 AISBench 其他功能 | 全部 harbor import 为函数级 lazy import；`runtime.txt` 不变 |

---

## 2. 总体架构

```
ais_bench config.py --mode agent
  └─ TaskManager → ConfigManager（加载配置、合并 CLI agent 参数）
       └─ WORK_FLOW["agent"] = [AgentEval, AccViz]
            ├─ AgentEval.update_cfg
            │     • 合并 CLI agent 参数到 cfg（CLI 优先）
            │     • eval.runner.type = HarborRunner（未显式设置时）
            │     • eval.runner.task.type = HarborAgentTask（取配置，沿用 Eval 的 custom_task 模式）
            │     • eval.partitioner = NaivePartitioner，out_dir = work_dir/results/
            │     • 注入 monitor_port / jobs_dir / refresh_interval / max_num_workers / debug
            ├─ AgentEval.do_work
            │     • partitioner(cfg) → clear_repeat_tasks → RUNNERS.build(cfg.eval.runner)(tasks)
            │     • 每个 task = model × dataset 组合 → 一个 HarborJob
            │           └─ HarborRunner.launch(tasks)
            │                 ├─ 单任务（tasks==1）：进程内直跑
            │                 │     └─ TASKS.build(HarborAgentTask).run(无子进程，日志直接打印)
            │                 │           ├─ AgentParamAdapter: 统一参数 → 各 agent 私有 kwargs/env
            │                 │           └─ harbor Job.create(config) + job.run() → 周期落盘 result.json
            │                 ├─ 多任务（tasks>1）：子进程拉起 + 主进程看板
            │                 │     ├─ HarborAgentTask 子进程（get_command + param file 机制）
            │                 │     ├─ TasksMonitor 看板（守护线程，读 status_tmp 打印每任务进度条）
            │                 │     └─ HarborMonitor（守护线程）读 status_tmp + job_dir/result.json
            │                 └─ HarborMonitorServer（守护线程，标准库 http.server，两种模式均启动）
            │                       └─ GET /api/health、/api/tasks、/api/tasks/{name}、/api/tasks/{name}/cases、/api/jobs
            └─ AccViz: HarborSummarizer 汇总 results/{model}/{dataset}.json
```

---

## 3. 模块设计

### 3.1 Workflow 与专属 Worker（修改 `cli/workers.py`）

`WORK_FLOW`（[workers.py#L771-L780](file:///d:/group_dev/adapt_harbor/benchmark/ais_bench/benchmark/cli/workers.py#L771-L780)）新增：

```python
WORK_FLOW = dict(
    all=[Infer, JudgeInfer, Eval, AccViz],
    infer=[Infer],
    judge=[JudgeInfer],
    infer_judge=[Infer, JudgeInfer],
    eval=[JudgeInfer, Eval, AccViz],
    viz=[AccViz],
    perf=[Infer, PerfViz],
    perf_viz=[PerfViz],
    agent=[AgentEval, AccViz],       # 新增：拉起 HarborJob + 汇总
    agent_viz=[AccViz],              # 新增：仅对已有结果做汇总
)
```

新增 `AgentEval(BaseWorker)`（继承 [BaseWorker](file:///d:/group_dev/adapt_harbor/benchmark/ais_bench/benchmark/cli/workers.py#L65-L78)）：

- `update_cfg(cfg)`：
  - 将 CLI agent 参数合并进 `cfg["models"][0]` 与 `cfg["datasets"][0]["args"]`（见 3.5）；
  - 若 `cfg["eval"]["runner"]["type"]` 未设置 → 置为 `HarborRunner`（字符串形式）；
  - `task.type` 取配置中的 `HarborAgentTask`（若配置未给 `eval` 块则用默认 `HarborAgentTask`）；
  - 设置 `eval.partitioner = NaivePartitioner`、`out_dir = work_dir/results/`；
  - 注入 runner 参数：`max_num_workers`、`monitor_port`、`refresh_interval`、`jobs_dir`、`debug`。
- `do_work(cfg)`：`PARTITIONERS.build → partitioner(cfg) → clear_repeat_tasks → RUNNERS.build(cfg.eval.runner)(tasks)`，与 `Eval.do_work` 骨架一致。

同步修改 `cli/argument_parser.py`：`--mode` 的 `choices` 增加 `agent`、`agent_viz`。

### 3.2 Agent 专属 Runner + 监控服务（新增 `runners/harbor_runner.py`、`runners/harbor_monitor.py`）

#### `HarborRunner(BaseRunner)`（注册到 `RUNNERS` registry）

不依赖 GPU 池。按任务数量分两种执行模式：

**模式 A：单任务（`len(tasks) == 1`）——进程内直接执行**

- **不起子进程**：runner 直接 `TASKS.build` 构建 `HarborAgentTask`，在主进程内调用 `task.run(task_state_manager=None)` 执行 harbor job；
- **日志直接打印**：harbor 自身日志与 rich 进度（以及 `_run_with_tqdm` 的 tqdm）直接输出到主进程 stdout，不做文件重定向；
- **每 case 明细仍通过监控服务呈现**：启动 `HarborMonitor` + `HarborMonitorServer`（守护线程），以 `job_dir` 落盘文件为唯一信息源（in-process 无子进程状态文件，故**不注册** `status_tmp`，`job_dir` 仍按 `work_dir/results/{model}/{dataset}/details` 计算）；
- 不启动 `TasksMonitor` 看板（避免与直接日志输出重复）。

```python
def _launch_inline(self, task) -> Tuple[str, int]:
    monitor = HarborMonitor(work_dir, refresh_interval=self.refresh_interval)
    monitor.register_task(task_name, status_file=None, job_dir=self._task_job_dir(task))
    monitor.start()
    server = HarborMonitorServer(monitor, port=self.monitor_port); server.start()
    try:
        built = TASKS.build(dict(cfg=task, type=self.task_cfg['type']))
        built.run(task_state_manager=None)   # 进程内直跑，日志直接打印
        return built.name, 0
    except Exception:
        return built.name, 1
    finally:
        server.stop(); monitor.stop()
```

**模式 B：多任务（`len(tasks) > 1`）——子进程 + 主进程看板**

- 每个 task 以子进程方式拉起（复用 `task.get_command(cfg_path, template)` + 临时 param file + out 日志重定向，与 `LocalRunner._launch` 一致），子进程负责写 `status_tmp`；
- **主进程启动看板**：复用现有 `TasksMonitor.launch_state_board()`（curses 交互屏，后台模式退化为进度条），读取各子进程 `status_tmp`，打印每个 harbor 任务的进度条；看板在主进程的守护线程中运行，任务结束即退出；
- 同时启动 `HarborMonitor` + `HarborMonitorServer`：为每个任务注册 `status_file`（进程级状态）+ `job_dir`（case 级明细），对外提供每 case 实时信息；
- 任务结束统一清理临时文件，返回 `[(task_name, exit_code)]`。

```python
def launch(self, tasks) -> List[Tuple[str, int]]:
    if len(tasks) == 1:
        return [self._launch_inline(tasks[0])]     # 模式 A
    monitor = HarborMonitor(work_dir, refresh_interval=self.refresh_interval)
    for task in tasks:
        monitor.register_task(task_abbr, status_file=..., job_dir=self._task_job_dir(task))
    monitor.start()
    server = HarborMonitorServer(monitor, port=self.monitor_port); server.start()
    board = self._start_task_board(task_names, work_dir, self.debug)  # TasksMonitor 看板线程
    try:
        status = self._run_tasks(tasks)            # 子进程并发拉起（模式 B）
    finally:
        board.join(); server.stop(); monitor.stop()
    return status
```

#### 看板实时指标（正确/错误/异常/平均分）

模式 B 的 `TasksMonitor` 看板 "Extend Parameters" 列实时显示每个 harbor 任务的四项指标，数据由任务子进程定时从 `job_dir/result.json` 读取并写入 `status_tmp` 的 `other_kwargs` 字段（看板直接展示该字段，数据流不变）：

| 指标 | 口径（读自 `result.json` 的 `trial_results`） |
|---|---|
| `correct` 正确 | `reward >= 1.0` 的条数 |
| `wrong` 错误 | `reward < 1.0` 且无异常的条数 |
| `exception` 异常 | 带 `exception_info` 的条数 |
| `avg_score` 平均分 | 已完成 trial 的 reward 均值（4 位小数） |

实现：
- `HarborTask._refresh_progress_metrics()`：无操作钩子，在 `_run_with_tqdm` 的 `monitor_progress` 线程中每轮调用（对旧 HarborTask 流程零影响）；
- `HarborAgentTask._refresh_progress_metrics()`：覆盖实现，**每 2s** 调 `_job_metrics()` 读 `result.json`，结果写入 `task_state["other_kwargs"]`（`update_task_state` 合并进状态文件，看板每 `refresh_interval` 刷新展示）。
- 模式 A（in-process，`task_state_manager=None`）不写入，模式 A 本无看板。

#### 中断处理（Ctrl+C 优雅回收）

模式 B 下 Ctrl+C 的完整处理链路（进程组内子进程同样收到 SIGINT，自行回收容器）：

1. **立即停看板**：`_launch_multi` 捕获 `KeyboardInterrupt` → `_stop_board(board)`（`TasksMonitor.stop_state_board()` 设置停止标志，curses/后台循环退出，终端恢复）；
2. **等待回收 + 心跳**：`_wait_for_cleanup(timeout=cleanup_timeout)` 轮询 `_active_popens`，首条心跳**立即打印**，之后每 5s 一条 `Waiting for harbor cleanup (N subprocess(es), Xs)...`，避免误以为卡死；
3. **超时兜底**：超过 `cleanup_timeout`（默认 120s，可用 runner 配置 `cleanup_timeout` 调整）→ 对残留子进程 SIGTERM → 10s 宽限 → SIGKILL，父进程必然退出；
4. **干净退出**：`raise SystemExit(130)`，不打印 KeyboardInterrupt 调用栈。

子进程侧配套修复（否则容器已回收、子进程却因线程卡死不退）：`harbor_agent_task.py` / `harbor_task.py` 的 `__main__` 中 `except Exception` 改为 `except BaseException`，使 Ctrl+C（KeyboardInterrupt）时同样把 `task_state_manager` 状态置为 `"error"`，令非 daemon 的 `manager_t` 线程退出循环，子进程在容器回收后立即正常退出。

> 备注：`_launch` 使用 `Popen + wait()` 而非 `subprocess.run()`，避免中断时被 SIGKILL 打断容器回收；并行模式 `_run_tasks` 中断时不阻塞等待（`executor.shutdown(wait=False)`）。

#### `HarborMonitor`：两级监控快照（任务级 + 每 case 级）

除 harbor 进程本身的状态外，harbor job 落盘文件携带大量执行信息。监控按「任务级聚合 + 每 case（trial）明细」两级结构组织，全部信息只读自落盘文件。

**一级：任务级快照**（结构可扩展，新增来源写入 `harbor` / `extra` 字段）

```json
{
  "task_name": "HarborAgentTask_terminus-2_harbor_terminal-bench-2",
  "status": "running",
  "process_id": 1234,
  "finish_count": 3,
  "total_count": 20,
  "progress_description": "Running Harbor Job",
  "start_time": 1720000000.0,
  "log_path": "logs/eval/xxx.out",
  "job_dir": "results/terminus-2/harbor_terminal-bench-2/details",
  "harbor": {
    "n_total_trials": 20,
    "n_running_trials": 3,
    "n_completed_trials": 3,
    "n_errored_trials": 0,
    "n_pending_trials": 14,
    "evals": {
      "terminus-2__qwen3__terminal-bench-2": {
        "n_trials": 3, "n_errors": 0,
        "metrics": [{"mean": 0.85}],
        "pass_at_k": {"1": 0.85}
      }
    }
  },
  "extra": {}
}
```

信息源三路合并：
1. `status_tmp/tmp_<task>.json`：TaskStateManager 周期写入的进程级状态（`status/finish_count/total_count/process_id`）；
2. `job_dir/result.json`：harbor `Job.run()` 运行中持续写入，含 `JobStats` 进度计数与 `evals`（其中 `exception_stats` 即聚合级失败原因分布）；
3. `job_dir/trial_*` 目录扫描（见二级）。

**二级：每 case（trial）执行状态与成败原因** —— 由 `HarborMonitor` 直接扫描 job 目录下 `trial_*/` 落盘文件推导（目录布局以 0.21.0 `TrialPaths` 为准）：

```json
{
  "trial_name": "task1__AbC1234",
  "task_name": "task1",
  "status": "completed",
  "reward": 1.0,
  "rewards": { "reward": 1.0, "pass": 1.0 },
  "exception": {
    "type": "VerifierTimeoutError",
    "message": "Verifier timed out after 300s",
    "occurred_at": "..."
  },
  "verifier": {
    "has_reward_json": true,
    "reward_json": { "reward": 1.0 },
    "stdout_tail": "...",
    "stderr_tail": "...",
    "ctrf": { "passed": 5, "failed": 0, "skipped": 0, "failures": [] }
  },
  "timings": {
    "started_at": "...", "finished_at": "...",
    "agent_execution_sec": 120.5, "verifier_sec": 3.2
  },
  "agent": {
    "name": "terminus-2", "version": "0.21.0", "model": "qwen3",
    "has_trajectory": true,
    "tokens": { "input": 12345, "output": 678, "cost_usd": 0.0 }
  },
  "extra": {}
}
```

**case 状态推导规则**（扫 `trial_*` 目录，`result.json` 存在且 mtime 变化才重读）：

| 条件 | status |
|---|---|
| 尚无 trial 目录且已建 trial 数未达 `n_total_trials` | `pending` |
| 有 trial 目录但无 `result.json`（agent / verifier 正在写入） | `running` |
| `result.json.exception_info.exception_type == "CancelledError"` | `cancelled` |
| `result.json.exception_info` 非空 | `errored` |
| `result.json.verifier_result.rewards` 非空 | `completed`（含 reward） |
| 多步任务（`steps/` 存在）：按各 step 的 agent/verifier 落盘文件递归推导并聚合为 trial 级 status | 同上规则 |

**成败原因信息源**（0.21.0 `TrialPaths` 定义）：
- `trial_*/result.json`：`exception_info`（type / message / traceback）、`verifier_result.rewards`、时间戳、`agent_info`、token/cost；
- `trial_*/exception.txt`：harbor 单独落盘的异常消息文本；
- `trial_*/verifier/reward.json`（奖励明细）、`reward.txt`（文本奖励）；
- `trial_*/verifier/test-stdout.txt` / `test-stderr.txt`：验证器输出（尾部截断）；
- `trial_*/verifier/ctrf.json`：CTRF 格式，每测试用例 passed / failed / skipped + message；
- `trial_*/agent/trajectory.json`：agent 轨迹存在性（运行中信号）；`agent/analysis.md`（可选分析）；
- `trial_*/trial.log`：trial 日志尾部（调试用）。

**性能与增量策略**：维护「已解析 trial 目录 + 文件 mtime」缓存，仅 mtime 变化时重读 `result.json` / `ctrf.json`；stdout / stderr 只取尾部固定行数；新 trial 目录首见时全量解析一次。

#### `HarborMonitorServer`（标准库 `http.server.ThreadingHTTPServer`）

| 端点 | 说明 |
|---|---|
| `GET /api/health` | 存活探测 |
| `GET /api/tasks` | 全部 task 任务级快照列表 |
| `GET /api/tasks/{模型}/{数据集}/` | **任务总览 = `job_dir/result.json` 原文**（尾斜杠触发） |
| `GET /api/tasks/{模型}/{数据集}/{case}` | **单个 case 的 `trial_*/result.json` 原文**（case 可为 trial 目录名 `trial_00000`、数字序号 `0`、或 harbor task 名如 `astropy__astropy-12907`） |
| `GET /api/tasks/{模型}/{数据集}/cases` | 单个 task 的每 case 执行状态与成败原因明细（派生快照） |
| `GET /api/tasks/{模型}/{数据集}` | 单个 task 的任务级派生快照（向后兼容） |
| `GET /api/jobs` | 各 harbor job 进度（聚合计数 + case 状态统计） |

- 端口由 `--monitor-port` 指定（默认 `0` = 不启动服务）；
- 只读服务，无写入端点，不跨域写请求；
- 采用标准库实现，零新增依赖；
- `{模型}/{数据集}` 即任务名 `task_abbr_from_cfg`（`模型abbr/数据集abbr`），case 结果未落盘时返回 404。

### 3.3 支持全部 agent 的新 Task（新增 `tasks/custom_tasks/harbor_agent_task.py`）

`HarborAgentTask(HarborTask)`（继承 [harbor_task.py](file:///d:/group_dev/adapt_harbor/benchmark/ais_bench/benchmark/tasks/custom_tasks/harbor_task.py)）：

- **复用**：`get_command / run / _run_with_tqdm / _resume_job / _dump_eval_results`（保证结果落盘格式与断点续跑行为与现有实现完全一致）；
- **看板实时指标**：覆盖 `_refresh_progress_metrics()`，每 2s 读 `job_dir/result.json` 汇总 正确/错误/异常/平均分 写入状态文件 `other_kwargs`（详见 3.2「看板实时指标」）；
- **中断可退出**：`__main__` 的 `except Exception` 改为 `except BaseException`，Ctrl+C 时也置状态 `"error"`，非 daemon 的 `TaskStateManager` 线程正常退出（详见 3.2「中断处理」）；
- **重写** `_run_harbor_job` 的 `JobConfig` 构建（严格对齐 harbor **0.21.0** 的 `JobConfig` / `AgentConfig` / `DatasetConfig` 定义）：
  - **Agent 构建（不再直接构造 `AgentName` 枚举）**：`AgentConfig.name` 传原始字符串（任意 `AgentName` 值或 `module:ClassName` 自定义 agent），`AgentConfig.import_path` 单独透传；仅当值不含 `:` 时校验其属于 `AgentName.values()`，非法值报清晰错误；
  - **0.21.0 `AgentConfig` 字段透传**：`name / import_path / model_name / n_concurrent / concurrency_group / skills / override_timeout_sec / override_setup_timeout_sec / max_timeout_sec / resume_trajectory / load_trajectory / extra_allowed_hosts / include_logs / exclude_logs / kwargs / env / deps_path / mcp_servers`（按配置可选）；
  - **0.21.0 `JobConfig` 字段透传**：`n_attempts / timeout_multiplier / agent_timeout_multiplier / verifier_timeout_multiplier / agent_setup_timeout_multiplier / environment_build_timeout_multiplier / debug / n_concurrent_trials / quiet / retry(max_retries, include_exceptions, exclude_exceptions) / install_only / environment(type, force_build, delete, env, kwargs, override_*) / verifier(disable, env, import_path, kwargs) / metrics / artifacts / extra_instruction_paths`；
  - **数据集全部来源（0.21.0 `DatasetConfig`）**：
    - 本地路径 `args.path`：自动识别「单个 task 目录 → `config.tasks`」/「数据集目录 → `config.datasets`」；
    - registry：`args.dataset_name_version`（`name@version`）；
    - package：`org/name@ref`；
    - 过滤：`task_names / exclude_task_names / n_tasks`；
  - **环境类型**以当前 `EnvironmentType` 枚举为准（docker / daytona / e2b / modal / runloop / gke / novita 等）。

### 3.4 参数适配器（新增 `utils/agent_params.py`）

`AgentParamAdapter`：把统一语义参数 → 各 agent 的 `AgentConfig.kwargs` / `AgentConfig.env`。

```python
class AgentParamAdapter:
    EXPLICIT_MAP = {
        "terminus-2": {"api_base": ("kwarg", "api_base")},
        # 其余 installed 系 agent 动态发现
    }

    @classmethod
    def translate(cls, agent_name, unified: dict) -> dict:
        """unified: {api_base, api_key, llm_kwargs, model_info, ...}
        返回 {"kwargs": {...}, "env": {...}}"""
```

转换优先级：

1. **显式映射** `EXPLICIT_MAP`（如 terminus-2 → `kwargs["api_base"]`）；
2. **动态发现（基于 0.21.0 描述符）**：读取 `harbor.agents.installed.base.BaseInstalledAgent.ENV_VARS`（`EnvVar(kwarg, env, env_fallback, ...)`）与 `CLI_FLAGS`（`CliFlag(kwarg, cli, env_fallback, ...)`），匹配 `kwarg` 含 `api_base/base_url` 或 `env` 含 `BASE_URL` 的项，自动映射到对应环境变量（如 `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL`）；`api_key` 同理匹配 `API_KEY`；
3. **兜底约定**：`api_base → OPENAI_BASE_URL`、`api_key → OPENAI_API_KEY`；
4. **按 agent 注入默认 kwargs**（`AGENT_DEFAULT_KWARGS`）：如 mini-swe-agent 未提供 `config` 时自动注入 `{"model": {"model_class": "litellm"}}`，用户显式 `--ak config` 仍优先；
5. **向后兼容**：配置已含原始 `agent_kwargs`（如现有示例的 `agent_kwargs.api_base`）时原样直通，不重复转换。

### 3.5 命令行常用参数（修改 `cli/argument_parser.py`）

新增 `agent_args` 分组（避开既有 `-m/-r/-s/-w` 短参数冲突）：

| 参数 | 说明 | 映射目标 |
|---|---|---|
| `-a/--agent` | agent 名或自定义 import path | `models[0].agent_name` |
| `--model` | 模型名（可多值） | `models[0].model_names` |
| `--api-base` | 模型服务 base url（统一语义） | `models[0].api_base` |
| `--api-key` | API key | `models[0].api_key` |
| `--ak/--agent-kwarg` | agent 私有参数 `key=value`（可多值） | `models[0].agent_kwargs` 追加 |
| `--ae/--agent-env` | agent 环境变量 `KEY=VALUE`（可多值） | `models[0].agent_env` 追加 |
| `--agent-deps` | 离线 agent 依赖包路径（`<agent>.tar.gz` 或目录） | `models[0].deps_path` |
| `-p/--path` | 本地数据集 / 单 task 路径 | `datasets[*].args.path` |
| `-d/--dataset` | registry 数据集 `name@version` | `datasets[*].args.dataset_name_version` |
| `-n/--n-concurrent` | 并发 trial 数 | `datasets[*].args.n_concurrent_trials` |
| `-k/--n-attempts` | 每 trial 尝试次数 | `datasets[*].args.n_attempts` |
| `-e/--environment` | 环境类型（docker/daytona/e2b/modal…） | `datasets[*].args.environment_type` |
| `--timeout-multiplier` | 超时倍数 | `datasets[*].args.timeout_multiplier` |
| `--max-retries` | 最大重试次数 | `datasets[*].args.max_retries` |
| `--include-task-name` / `--exclude-task-name` / `--n-tasks` | 任务过滤 | `datasets[*].args.*` |
| `--force-build/--no-force-build`、`--delete/--no-delete` | 环境构建/清理策略 | `datasets[*].args.*` |
| `--host-network` | 所有 task 容器共享宿主机网络 | `datasets[*].args.environment_kwargs["host_network"]=True` |
| `--disable-verification` | 禁用验证器 | `datasets[*].args.disable_verification` |
| `--env-file` | .env 文件路径 | `datasets[*].args.env_file` |
| `-q/--quiet` | 静默模式 | `datasets[*].args.quiet` |
| `-y/--yes` | 自动确认 | `datasets[*].args.yes` |
| `--monitor-port` | 监控服务端口（0=关闭，默认 0） | runner 参数 |

`AgentEval.update_cfg` 统一将上述 CLI 值合并进 cfg，CLI 优先于配置文件。

### 3.6 依赖隔离（新增 `requirements/agent.txt`，修改 `setup.py`、`runners/local.py`）

目标：agent 测评仅安装极简依赖即可运行，主要依赖收束到 harbor；不影响 AISBench 其他功能。

**新增 `requirements/agent.txt`**（核心 CLI + **harbor**，不含 torch/transformers/datasets/opencv 等重依赖）：

```
harbor
mmengine-lite
numpy
Pillow
tqdm>=4.64.1
tabulate
orjson
psutil
pyyaml
requests
python-dotenv
windows-curses; sys_platform == "win32"
```

- `harbor` 从 PyPI/镜像直接安装（**不沿用** `requirements/datasets/harbor.txt` 的 `harbor==0.6.1` 锁定）；如需精确对齐本地 0.21.0 源码，可单独执行 `pip install -e <harbor绝对路径>`；
- `orjson` / `psutil` / `Pillow` / `windows-curses` 为 CLI 导入链（`runners/base.py`、`utils/file/file.py`、`partitioners/base.py → datasets/utils/datasets.py`）所需；
- pydantic / typer / rich / fastapi / uvicorn / litellm 等由 harbor 自带。

**`setup.py`**：新增 `extras_require["agent"] = parse_requirements("requirements/agent.txt")`；`install_requires=runtime.txt` 与默认安装保持不变。

**隔离前置改造（仅影响导入时机，行为不变）**——CLI 导入链 torch 顶层依赖仅 3 处，已全部 lazy 化：

- `runners/local.py`：顶层 `import torch` / `mmengine.device.is_npu_available` 移入 `launch()`；
- `openicl/icl_inferencer/icl_base_local_inferencer.py`：顶层 `from torch.utils.data import DataLoader` 移入 `get_dataloader()`；
- `tasks/__init__.py`：改为 PEP 562 惰性 `__getattr__` 再导出，避免包导入即拉入 swebench/oneig/openicl 等重型后端；
- `cli/workers.py`、`utils/config/run.py`：`tasks` / `openicl` / partitioner / runner 相关导入函数内化；
- `cli/config_manager.py`：`datasets.custom`（依赖 HuggingFace `datasets`）导入函数内化；`try_fill_in_custom_cfgs` 对 agent 风格数据集（含 `args` 字段）跳过 dummy 填充，避免加载配置时拉入 openicl / `datasets`；
- `datasets/__init__.py`：各数据集后端星号导入改为逐模块 try/except 守卫，缺失可选依赖（`datasets`/torch/transformers）时仅跳过该后端，registry 按点号路径加载不受影响；
- `utils/file/__init__.py`：`load_tokenizer`（顶层依赖 `transformers`）改为 PEP 562 惰性再导出；`load_tokenizer.py` 内 `transformers` 亦改为函数内惰性导入；
- `summarizers/__init__.py`：改为 PEP 562 惰性再导出（`default_perf` 依赖 plotly，vbench/swebench/oneig 依赖重型后端）；`workers.py` 中 `DefaultPerfSummarizer` 移入 `PerfViz.update_cfg` 惰性导入；
- 所有 harbor 相关 import 保持函数级 lazy import（现有 `harbor_task.py` 已是此风格）。

**独立使用方式**：

```bash
python -m venv .venv-agent && .venv-agent\Scripts\activate
pip install -e . --no-deps
pip install -r requirements/agent.txt
ais_bench configs/agent_example/harbor_agent_task.py --mode agent
```

---

## 4. 兼容性保障清单

1. **结果格式**：`HarborAgentTask` 复用 `_dump_eval_results`，落盘字段与 [harbor_task.py#L326-L344](file:///d:/group_dev/adapt_harbor/benchmark/ais_bench/benchmark/tasks/custom_tasks/harbor_task.py#L326-L344) 一致（`total_count / n_errors / avg_score / reward_distribution / exception_distribution / n_total_trials / pass_at_k`），`HarborSummarizer` 无需改动。
2. **配置格式**：沿用 `models` / `datasets` 划分；原 `agent_kwargs.api_base` 写法继续可用（适配器直通），新增统一字段为可选增强。
3. **其他功能零影响**：`runtime.txt`、默认安装、`all/infer/eval/perf` 等既有 workflow 完全不动；新增组件均通过 `RUNNERS` / `TASKS` registry 注册并按需 lazy import。
4. **断点续跑**：沿用 harbor job 目录 `config.json` 检测 → `_resume_job`。

---

## 5. 改动文件清单

### 新增

| 文件 | 说明 |
|---|---|
| `benchmark/ais_bench/benchmark/runners/harbor_runner.py` | `HarborRunner`：多 task 拉起 + 监控调度 |
| `benchmark/ais_bench/benchmark/runners/harbor_monitor.py` | `HarborMonitor`（状态快照存储）+ `HarborMonitorServer`（`http.server`） |
| `benchmark/ais_bench/benchmark/tasks/custom_tasks/harbor_agent_task.py` | `HarborAgentTask`：支持全部 agent / 全部数据集来源 |
| `benchmark/ais_bench/benchmark/utils/agent_params.py` | `AgentParamAdapter`：统一参数 → 各 agent 私有参数 |
| `benchmark/requirements/agent.txt` | agent 测评独立依赖集 |
| `benchmark/ais_bench/configs/agent_example/harbor_agent_task.py` | 新示例配置（展示统一参数与 `--mode agent`） |

### 修改

| 文件 | 说明 |
|---|---|
| `benchmark/ais_bench/benchmark/cli/workers.py` | 新增 `AgentEval` worker；`WORK_FLOW` 增加 `agent` / `agent_viz`；task 类导入函数内化 |
| `benchmark/ais_bench/benchmark/cli/argument_parser.py` | `--mode` choices 增加 `agent` / `agent_viz`；新增 `agent_args` 参数组 |
| `benchmark/ais_bench/benchmark/cli/config_manager.py` | `datasets.custom` 导入函数内化（依赖隔离） |
| `benchmark/ais_bench/benchmark/runners/__init__.py` | 导出 `HarborRunner`（保持无 harbor 顶层依赖） |
| `benchmark/ais_bench/benchmark/runners/local.py` | torch / is_npu_available lazy import |
| `benchmark/ais_bench/benchmark/tasks/__init__.py` | 改为 PEP 562 惰性再导出（依赖隔离） |
| `benchmark/ais_bench/benchmark/openicl/icl_inferencer/icl_base_local_inferencer.py` | `DataLoader` lazy import（依赖隔离） |
| `benchmark/ais_bench/benchmark/datasets/__init__.py` | 数据集后端星号导入改为逐模块守卫（依赖隔离） |
| `benchmark/ais_bench/benchmark/utils/file/__init__.py` | `load_tokenizer` 惰性再导出（依赖隔离） |
| `benchmark/ais_bench/benchmark/summarizers/__init__.py` | 改为 PEP 562 惰性再导出（依赖隔离） |
| `benchmark/ais_bench/benchmark/utils/config/run.py` | openicl/tasks 导入函数内化；agent 数据集跳过 dummy 填充 |
| `benchmark/setup.py` | 新增 `agent` extra |

---

## 6. 使用示例

```bash
# 配置方式（沿用 models/datasets 划分，配置文件读入与 AISBench 原有方式一致）
ais_bench configs/agent_example/harbor_agent_task.py --mode agent

# 命令行覆盖常用参数 + 开启监控服务
ais_bench configs/agent_example/harbor_agent_task.py --mode agent \
  -a terminus-2 --model hosted_vllm/qwen3 --api-base http://0.0.0.0:8080/v1 \
  -p /path/to/terminal-bench-2 -n 5 -k 1 --monitor-port 8787

# 外部实时查询所有 harbor task 执行信息
curl http://127.0.0.1:8787/api/tasks

# 仅汇总已有结果
ais_bench configs/agent_example/harbor_agent_task.py --mode agent_viz
```

---

## 7. 验证方式

1. **依赖隔离验证**：在仅安装 `agent.txt` 的 venv 中执行
   `python -c "from ais_bench.benchmark.cli.workers import WORK_FLOW"`，无 torch 报错。
2. **全量回归**：现有环境跑 `--mode all -c <既有配置>` 冒烟，确认 infer / eval / perf 行为不变。
3. **agent 端到端**：跑 `--mode agent`，验证：
   - 多 dataset 并发拉起多个 harbor job；
   - `GET /api/tasks` 返回实时进度，`harbor` 快照随 `result.json` 更新；
   - `GET /api/tasks/{name}/cases` 返回每 case 的 `status`（pending/running/completed/errored/cancelled）与成败原因（`exception` / `verifier.ctrf` / `reward`），并随落盘文件 mtime 更新；
   - 构造一个失败 case（如超时/验证失败），确认 `exception.type + message` 与 `ctrf.failures` 正确上报；
   - 结果 json 字段与旧版 `harbor_task.py` 一致，`HarborSummarizer` 汇总正常；
   - 中断后 `--mode agent` 重跑自动 resume。
4. **多 agent 适配**：分别以 terminus-2（kwargs）与 claude-code（env 变量）验证 `AgentParamAdapter` 转换正确。
