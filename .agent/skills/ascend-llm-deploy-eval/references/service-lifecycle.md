# 串行 case 与服务生命周期

## 准备与循环

plan_run 固定 sub_task 顺序并生成 run.sh；preflight 在有人值守时完成现场绑定。run.sh 开始后，每个 sub_task 调用 deploy → evaluate → finalize。deploy 每次检查服务状态；调用该阶段不代表必须重新部署。

协调端在 .internal/services/<deployment_id>/service-state.json 保存整套服务状态；节点各容器目录的 run_state.json 保存本地服务状态，执行入口节点在其中的 cases[sub_task_id] 单独记录 case 状态。整组任务状态在 task-state.json，日志与结果按部署、case 和 attempt 分开。

| 状态 | 行为 |
|---|---|
| ABSENT / RELEASED | 检查即时资源后建立服务 |
| DEPLOYING / DEPLOYED | 等待启动与默认冒烟，不重复启动 |
| READY | 核验容器/进程身份与健康检查，匹配则跳过部署和冒烟 |
| TESTING | 当前 case 执行中，不启动另一 case |
| FAILED / UNKNOWN | 保留失败原因，按已实现的有限重试或收尾处理，不盲目复用 |

实例部署摘要只包含部署与绑定内容，不包含并发、请求数等 case 负载。容器归属使用独立的任务容器摘要与实际容器 ID 核验，服务身份仍按部署分别记录；case 状态、结果和重试预算独立。服务重启后旧冒烟失效。

## 资源分层

| 资源 | 当前策略 |
|---|---|
| case 客户端与进程组 | 测试结束/异常时结束，核验无活动成员；清理不明阻止下一 case |
| case 配置与外部数据集 | 保留配置和结果，仅引用外部数据路径，不创建或删除数据链接 |
| 前缀缓存 | 默认不清理；显式 reset_cache=true 时在原生预热前重置并核验；case 收尾标记 NEEDS_RESET，避免重复清理 |
| 相同部署的服务 | 连续 case 默认保留有效 READY 实例 |
| 部署切换 | 默认停止旧部署服务，释放核验后复用兼容任务容器；显式 stop_container 则停止但保留容器对象待复用 |
| 任务结束 | 按 task_end 处理所有登记部署；一个释放失败不跳过其他部署，失败单独报告 |

每台机器首个兼容容器以总 job_id 命名，额外不兼容配置使用数字子编号。进程组登记在 service-processes.json；客户端身份、释放证据保存在各 attempt 的 client.json / cleanup.json。已完成结果保留，重复任务入口不会重新发送成功/失败终态 case 的请求，仍执行任务收尾。

## 锁、重试和失败

整组循环持有任务锁，各部署操作另持有容器锁。前缀重置、工具配置与负载都在锁内执行。不是并行任务调度器，也不自动抢占其他用户资源。

部署和 case 运行异常默认最多额外重试3次。case 预算按 sub_task_id 分离；显式释放后重新建立服务时记录新的部署尝试批次。重试前核验服务身份和客户端清理；显式要求清理缓存时，每次重试执行清理。未启用时不保证缓存状态一致。耗尽后当前子任务失败保留，统一全量释放本任务部署容器、服务及客户端；释放核验通过后继续下一 subcase 并重新部署。清理未核验时阻塞相关资源。

遇到客户端清理不明或未知状态，停止相关循环并进入适用的任务收尾。中断后仍为 RUNNING 的 case 不自动恢复，避免重放未知请求。服务与容器停止不删除共享镜像、权重或历史证据。

## 配置存储埋点

logs/config-storage-events.jsonl 记录 task_normalized、bindings_resolved、deployment_verified、case_finished，附任务/部署/case 标识和必要摘要。storage 固定标为 not_implemented；没有配置库写入、版本管理或自动匹配。

## 可选 Prefix Cache 清理的执行边界

仅 reset_cache=true 时执行下述检查，不是标准 AISBench 测试的默认前置条件。

缓存属于 case 隔离资源，服务可以继续复用。case finalize 标记 NEEDS_RESET；下一次 evaluate 的 case_runner 在预热和正式客户端启动之前调用 cache_reset.reset_instances，首次 case 和每次重试同样执行。任务结束停止服务时不再重复请求重置接口。

- 收集启动配置时，为需要 reset 的每个 vLLM 模型进程设置 instances[].launch.env.VLLM_SERVER_DEV_MODE="1"，模板在 bootstrap 之后导出该环境变量。保留已审阅的 Prefix Caching 参数；不为实现清理而关闭缓存。当前部署示例已为 P/D 模型进程启用该变量。
- 拓扑编译器从模型实例 checks 生成 service.reset_urls，列出全部需清理的模型实例 IP 与动态端口。PD 场景包括 P 和 D，混部列出 mixed 实例。不要默认填 Proxy 端口，也不要套用固定端口号。
- 在本任务独占锁与服务身份核验下，先读取全部实例 /metrics，核验 running/waiting 请求数均为零（含各 DP 标签）。指标缺失、不合法或仍忙则本次尝试失败；空闲快照不等于异步 KV 传输已结束，仍以 reset 返回值为准。
- 逐实例 POST /reset_prefix_cache，不传强制终止运行请求参数。必须 HTTP 成功且 JSON success 严格为 true，才进入预热。false、非 JSON、HTTP 错误或超时均阻止本次测试；记录其余实例结果，复用 case 最多额外3次的重试预算，不新增嵌套重试。
- 每个 attempt 保存 cache-reset.json、空闲指标、原始响应或 HTTP 错误。失败记录不会被后续成功覆盖。404 需核对运行版本与 Dev Mode，不能当成已清理；接口返回空响应的版本需要单独核验适配，不能仅凭 HTTP 200 放行。

此动作只证明各实例本地 Prefix Cache 重置得到确认，不保证全程零命中率，也不声称清除了外部 connector 缓存。当前不自动添加 reset_external 或调整 Hybrid 模型兼容性配置。Dev Mode 只对已授权的本任务测试服务启用。

官方接口参考：[vLLM reset_prefix_cache](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/serve/dev/cache/api_router/)。接口明确区分本地和可选外部缓存，并说明在运行请求或异步传输持有块时可能返回 false。

## 多节点协调

每个节点按镜像与容器配置创建或复用本任务容器，独立保存身份、脚本摘要、服务进程和释放证据。协调端通过固定 node_prepare.py / node_rpc.py 发起 JSON 数据请求；实际命令仍在节点本地执行。模型进程先启动并通过健康检查，再启动 proxy，最后从正式入口进行一次非流式 curl 真实推理冒烟，并保存请求、响应和服务身份。全体节点身份在正式测试前后核验，任何节点不明均不能宣称整套服务 READY。

协调端任务锁覆盖串行循环，节点操作使用容器锁；不提供跨协调端分布式调度。节点通信失败或释放失败逐项记录，继续尝试其他登记节点的释放。仅确认全部登记节点完成指定策略后报告任务释放通过。节点日志保留在各自任务目录，协调端保存节点位置与汇总；任务包和节点包由 Agent 上传相同版本。

首次执行将 sub_task 标记 RUNNING 后才调用节点。已完成终态子任务不重放；中断留下 RUNNING 时要求先收尾、诊断后新建任务，不自动恢复。不同 deployment 可共享兼容任务容器，切换前停止旧部署全部服务并核验。

预热由 AISBench --num-warmups 执行，按用户数据顺序发送；不做逐 DP 缓存覆盖或前缀命中率校验。

任务结束默认 stop_container：停止容器、核验占卡进程和监听端口释放，然后 docker rm 并核验删除。共享容器只实际删除一次，后续释放读取任务登记返回已删除；日志、结果、冻结脚本均保留在宿主挂载目录。任务内 stop_service 不停止容器，服务死亡也优先清理登记进程后复用容器。归属不明、当前仍登记其他服务或清理失败时禁止切换。

## 数据与存储初版

各 attempt 只引用用户指定的外部数据集，不生成数据或维护自有缓存。重试保留独立配置、日志和性能结果，不修改源数据。

每节点任务限额=1GiB+4GiB×本节点subtask数，单subtask结果限额4GiB，文件系统空闲保留10GiB。预检/部署检查剩余预算，压测入口在attempt前及客户端运行时每30秒检查；其他节点运行中尚无独立周期检查。超限结束客户端，停止服务后按record_only处理；后继重新检查空间，不自动删除保留的结果。外部数据集不在任务产物预算内，轮询检查不是硬配额。

释放前的状态、快照写入采用弱保护，OSError记录为recording_errors，不阻止资源操作；归属与身份检查不能跳过。容器停止后残留NPU进程只在本次捕获的PID和start_ticks一致时有限次TERM/KILL并复查，身份不明或清理失败仍阻塞。此保护不承诺在任意损坏状态下自动恢复。

## 执行环境回归检查

修改容器内 runner 依赖时，在只包含实际分发脚本的隔离目录执行导入和存储检查，不能用源码目录的 PYTHONPATH 掩盖缺失模块。重新部署的端口探测使用 SO_REUSEADDR 兼容连接回收状态；绑定仍失败时保存 ss 监听与归属证据，不把 errno=98 直接认定为其他任务占用或自动杀进程。

端口复用检查最多等待60秒，失败以可重试的 PortUnavailable 上报，不能包装为不可重试的配置 ValueError。容器停止后，NPU进程与监听进程分别核验；监听残留仅在本次容器快照PID和start_ticks匹配时TERM/KILL并复查，记录全部TCP状态以区分LISTEN与TIME_WAIT。未知监听不按“空进程”清理。真实外部监听持续冲突以结构化 PortUnavailable 传回协调器；当前部署全部节点释放核验通过后，在原部署重试预算内为全部节点重新分配端口并重新编译拓扑。新现场配置保存在 port-generations/<generation>，成功后更新 resolved-cluster.json 与内存登记，重新启动并冒烟；旧配置和释放证据保留，不修改冻结需求或旧脚本。分配失败、预算耗尽或释放不明则停止，不清理外部监听。
