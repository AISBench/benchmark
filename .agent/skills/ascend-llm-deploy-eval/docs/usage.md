# 使用与验证说明

[返回首页](../README.md)

本文保留操作细节及带日期的验证记录；历史验证不代表当前版本已完成真实模型回归。

## 当前执行方式

单个 job_spec 内按 sub_tasks 串行执行。需要跨任务衔接时，在主节点宿主机启动 group_run.py：独立服务器上的单机任务可并行；依赖任务结束且资源释放核验通过后，自动启动后继双机任务。单机执行进程放本机宿主，多机执行进程放主节点宿主，模型服务仍在任务容器内。主进程与子任务脱离 SSH 会话运行，关闭观察终端不影响执行。

五阶段保持 plan_run、preflight、deploy、evaluate、finalize。默认每 case 正式请求数=并发×4、运行1轮、最多额外重试3次；显式用户要求优先。同部署有效服务可复用，case 前重置缓存并预热，切换部署优先复用兼容容器；任务结束默认停止服务并删除任务容器，保留宿主产物、镜像与权重。

容器默认使用 root、host 网络/PID、500g 共享内存、privileged、八卡及管理设备和完整默认挂载。bindings.mounts 按目标路径覆盖或追加，runtime.container 支持已建模的参数覆盖。缺失挂载来源时阻塞预检，不让 Docker 创建空目录替代配置文件。

## 生成和提交

以下命令在本 skill 根目录执行。Python 3.10+，规划依赖 jsonschema。现场需要 Linux、Docker、curl、npu-smi 和免密 SSH；AISBench 从镜像动态发现，前缀工具随 Skill 分发。

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests/unit -v
python scripts/plan_run.py --job-spec /input/job_spec.json --run-dir /work/JOB_ID
# 上传完整任务包及相同版本 Skill 后，在任务协调节点进行有人值守预检：
python3 /work/JOB_ID/skill-snapshot/scripts/preflight.py --run-dir /work/JOB_ID --bindings /input/bindings.json
# 单任务前台执行入口：
bash /work/JOB_ID/run.sh
```

需要后台整组执行时，对每个子任务完成上述规划与预检，填写独立的 group-plan.json，之后在主节点执行（不要先运行子任务 run.sh）：

```sh
python3 /path/to/skill/scripts/group_run.py --start --plan /input/group-plan.json --run-dir /work/GROUP_ID
```

Agent 在提交前给出真实主节点和目录的观察命令，例如：

```sh
ssh -t user@main.example 'python3 /work/GROUP_ID/skill-snapshot/scripts/watch_events.py --run-dir /work/GROUP_ID'
```

事件按 subcase、阶段和 attempt 记录；终端显示北京时间并精确到秒，原始事件保留 UTC。远端每30秒轮询属于普通 Python 执行，不调用模型；当前聊天的 Agent 兜底监控默认每60分钟一次，正常静默。监控不可用时明确告知；默认通知可能延迟约一小时，应用离线可能更久。

## 实现与验证边界

- 仅接受 job_spec v3；支持混部及一个 P、一个 D、一个 proxy 的单机/双机 PD 结构。
- 整组并发以互不重叠的服务器为边界；所有 bindings 必须对同一主机使用统一地址。不支持同机分卡并发、自动别名归一或 job_spec 内 phase 并行屏障。
- 进程退出不等于成功。case 结果和全部登记节点释放记录共同决定终态；测量失败但释放通过可以继续后继，整组仍记录失败。
- 未实现多 P/D（如3P1D）、精度、配置库、复杂故障恢复和主机重启后的自动恢复。配置存储仅有埋点。
- 截至2026-11-01，Linux 72项检查通过；真实 SSH、后台进程、自动后继启动和终端观察已用合成任务验证。该验证没有启动模型或容器，不代表新版整组性能测试已通过。应用通知实际投递、完整 Agent 及强弱模型复现仍待验证。

`docs/` 只存设计、决策与工程计划；`tests/` 存可复用检查。日志、报告、现场绑定、脚本快照和真实基础设施信息保存在仓库外任务目录，不提交推送。示例只使用占位资源，不能作为执行授权。

## Quickstart：复制给Agent的任务提示词

下面示例已填好测试需求；执行前将示例主机、镜像、权重和脚本路径替换为实际资源。

```text
使用 $ascend-llm-deploy-eval 执行性能测试。
服务器：root@node-a.example，SSH免密，工作目录/work/benchmark。
镜像：registry.example/vllm-ascend:verified。
模型：GLM53 Flash，权重/mnt/models/GLM-5.3-Flash。
部署：单机混部，参考脚本/path/to/start.sh。
测试：输入256K、输出5K（1K=1024 tokens），并发32，前缀目标比例50%、90%。
```

## 检测阶段与当前检查项

```mermaid
flowchart LR
 A[需求齐备] --> B[脚本审阅与参数规范化]
 B --> C[现场预检]
 C --> D[部署前即时资源复查]
 D --> E[健康检查与真实请求冒烟]
 E --> F[测试与结果验证]
 F --> G[subtask释放与后继条件核验]
```

| 阶段 | 当前检查内容 | 证据/边界 |
|---|---|---|
| 规范化 | 部署角色、卡数与DP/TP、模型上下文上限、测试长度、参数绑定、case对应关系 | job_spec与plan；脚本内容需Agent审阅 |
| 连接与任务归属 | SSH节点调用、任务目录归属、冻结计划和代码快照一致性 | 节点登记和摘要 |
| 镜像与工具 | 镜像解析、探测容器内AISBench及依赖工具 | discovery；探测容器清理 |
| 权重与挂载 | 模型目录、索引引用权重分片、tokenizer文件、挂载源和设备路径存在 | preflight-evidence；不是全量权重校验和检查 |
| 网络与端口 | 可用IPv4、网卡速率及选择依据、端口范围分配、端口可绑定 | discovery/resolved；跨机业务通信由部署阶段进一步验证 |
| 现场资源 | NPU进程占用，或已有本任务READY服务身份核验 | npu快照；健康告警的解释由Agent核验 |
| 部署后 | 全部实例健康、入口真实推理冒烟 | 属于deploy，不应在preflight中提前标为通过 |
| 测试与收尾 | 缓存重置、预热、请求/输出和前缀指标验证、客户端与服务释放 | 每attempt结果与release-results |

节点预检保存preflight-report.json，记录检查时间、各组状态及失败原因；实际值保存在preflight-evidence.json与discovery.json。失败后的未执行项目保持NOT_RUN，终端只输出总体结论和错误。部署前再次核验即时资源；服务健康和真实请求冒烟仍属于部署阶段。

## 存储与异常收尾（初版）

任务节点预算为1GiB公共空间加4GiB×该节点subtask数，另保留10GiB文件系统空闲余量。预检/部署前检查预算，测试入口在每次尝试前及客户端运行期间每30秒检查；超限结束客户端并走服务释放。磁盘检查基于快照，不是文件系统硬配额；初版运行中周期检查在压测入口节点，其他节点在预检/部署时检查。结果保留不等于磁盘已释放，后继必须重新核验空间。

同一任务节点内缓存相同配置的数据集，按tokenizer相关文件、源数据、生成代码及生成参数校验；重试复用只读数据，执行结果独立保留。不同节点不共享缓存。细粒度算子日志变量在配置校验中拒绝显式赋值。

资源释放采用弱写盘保护；已核验归属的残留NPU进程按PID与启动标识有限次终止，未知身份不清理。协调器无法在任务分区记录中断时，尝试写入系统临时目录autobenchmark-interrupted-<job_id>.json；不自动恢复任务。

验证：2026-11-02 Linux 77项单元/进程测试通过；Windows通过、3项Linux专用检查跳过。尚未执行修改后的真实模型性能回归。
