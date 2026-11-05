> 历史设计记录：其中旧入口、快照目录和提交流程不适用于新任务。当前规则以 SKILL.md 和 docs/compact-cli.md 为准；旧公开入口已停用。

# 宿主机整组执行与事件

单机任务的执行进程放在该机宿主上，多机任务放在选定主节点宿主上，通过已有 SSH 节点执行器推进。整组 group_run.py 也运行在主节点宿主机，无常驻调度服务。模型服务仍在任务容器内；任务结束删除任务容器，保留宿主产物。

## 有人值守准备

1. 按现有五阶段为每个独立任务生成 job_spec、上传相同版本包并完成 preflight。单机尽量合并同服务器的多个部署以复用容器；同机不同卡并行暂不规划。
2. 在仓库外生成 group-plan.json，格式见 examples/group-plan.example.json。每个 task 提供 id、host（任务协调进程所在主机）、run_dir、冻结 plan.json 中的 job_spec_sha256、depends_on。它是运行编排清单，不改变 job_spec v3。group_id 与各 job_id 使用不同编号，run_dir 不与子任务目录重叠。
3. 每台机器在所有 bindings 中使用同一规范地址，不能用不同 SSH 别名/IP 将同一主机伪装成两项资源。协调器从各任务 resolved-cluster.json 的节点登记推导资源集合，不接受手填的空资源集合。当前资源键是规范主机地址，不支持别名自动归一。
4. 整组主节点必须能免密 SSH 到全部任务协调节点；子任务协调节点必须能连接各自执行节点。所有子任务 task_end 必须为 stop_container。当前主机不作为自己的 SSH 例外，提前核验自连接。
5. **实际提交前**按 [任务监控](task-monitoring.md#提交前的三类观察命令) 提供整体进展、服务日志、测试日志三类命令；以下为整体进展命令，填入真实主节点与路径并正确 shell 引号。用户可以在终端观察进度，不要求其开启终端才运行任务。

```sh
ssh -t user@main.example 'python3 /work/GROUP/common/skill-snapshot/scripts/watch_events.py --run-dir /work/GROUP'
# 或原始事件视图：
ssh -t user@main.example 'tail -n 30 -F /work/GROUP/logs/events.jsonl'
```

## 提交

在主节点使用已上传的 Skill 包：

```sh
python3 /path/to/skill/scripts/group_run.py --start --plan /work/group-plan.json --run-dir /work/GROUP
```

入口检查全部任务已预检且尚未执行，冻结整组计划和自身代码快照，再以新会话后台启动，标准输入关闭、输出写 logs/coordinator.log。子任务由 group_worker.py 同样脱离 SSH 启动，记录 PID/start_ticks 与退出记录。提交后重新连接核验 group-launch.json、进程身份和 group-state.json；SUBMITTED 不等于任务成功。启动记录存在时拒绝重复 --start。

协调器每30秒检查一次远端状态并汇集增量事件，不调用 Agent、不消耗模型推理。不同服务器任务并行，共享服务器任务互斥；depends_on 全部 DONE 且 cleanup_verified 后启动后继。DONE 表示执行终止且资源释放核验通过，不代表测量成功。测量 FAILED 但释放通过允许后继继续，最终整组仍 FAILED。

SSH 不可达记为 UNKNOWN，占用资源保持保留，继续检查；连续3次记录 blocked 事件供监控通知，不自动重跑。已确认中断但释放缺失记 BLOCKED，相关后继阻塞，独立任务继续。原任务记录、计划摘要或归属不明时不自动接管。主机重启/协调进程被杀后不提供自动恢复服务；可由 Agent 审阅身份与记录后单独处理，禁止删除启动记录来强行重跑。

## 事件结构

每条事件有 event_id、UTC time、stage、status；case 事件附 job_id、subcase_id（对应 sub_tasks.id）、case_id（复用负载定义）、deployment_id、phase_id，重试附 attempt。状态：doing、done、fail、skipped、blocked、retry。未开始阶段没有事件；doing 之后中断没有终态时保持未知，不推断完成。

| 范围 | 阶段 | 语义 |
|---|---|---|
| 整组/任务 | group、task | 启动、终止或阻塞 |
| subcase | deploy、deploy_attempt、smoke | 部署健康检查、部署尝试、真实请求冒烟；服务复用时 smoke=skipped |
| subcase | testing | 包含全部尝试，最终结果以验证产物为准 |
| attempt | cache_reset、benchmark、validate | 可选缓存重置、AISBench原生预热与正式请求、结果验证；预热细节见原生日志 |
| attempt | client_cleanup、attempt | 客户端释放及尝试结果；重试单独记 retry |
| subcase | reconcile、release | 失败后的服务状态核验、case资源释放策略 |
| 任务 | job_release | 全部登记节点的任务级释放 |

节点事件保存在各服务目录 logs/events.jsonl；整组进程将新增事件汇集到整组 logs/events.jsonl，附 task_id、source_stream、source_event，保留源 event_id。不同机器时钟可能有偏差，跨流显示顺序不是严格因果顺序。事件写入/收集失败不应阻断清理，写入错误保留在进程日志；终态必须读 summary、各 subtask 结果、release-results，不能只凭 done 事件判成功。崩溃窗口可能产生重复汇集，观察器/通知按源事件ID去重。

终端每条进展以北京时间时间戳开头，精确到秒并标注 +08:00；原始事件继续保存 UTC 时间。关闭观察终端不影响执行。watch_events.py 是只读显示，不负责通知和操作。整组结束保留全部事件、计划、状态、日志和子任务产物；无需常驻的整组进程自行退出。

## Agent 兜底

整组只绑定一个应用监控，覆盖全部子任务；创建、检查顺序、通知条件及停止方式统一遵循[任务监控](task-monitoring.md)。远端协调器负责自动推进，应用监控只读。
