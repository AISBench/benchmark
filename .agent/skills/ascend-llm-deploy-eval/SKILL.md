---
name: ascend-llm-deploy-eval
description: 以 job_spec 引导昇腾模型部署与性能测试，将自然语言需求规范化并完成现场预检，再执行部署、冒烟、测试和分级收尾。支持多模型、跨服务器整组任务及显式阶段并行，分支内部串行；多P/D需验证代理和连接兼容性，不包含精度或复杂自动恢复。
---

## 设计与保真边界

交互规划与执行分开：Agent 结合权重信息、机器环境、依赖规则和模型推理，把参考脚本整理成可审阅的 job_spec 与 bindings；确认后按这两份配置生成确定脚本。保真指确认后的任务参数，不是照搬输入脚本中失效的 IP、端口或权重路径。

交互阶段读取 [审阅规则](references/job-spec-review.md)。`scripts/review_job.py` 采集只读现场证据：权重 config.json、权重文件、网卡地址、设备概况、镜像及依赖路径。程序检查结构和确定性规则；Agent 据证据审阅架构、量化、上下文长度、并行配置、显存及连接依赖。探测成功只表示证据可读，不等于语义已确认；不足之处保留 unresolved。

确认前展示修改依据、默认值与派生参数；用户指定值优先。执行中不静默纠错、不降低负载、不换端口、不删除未知参数。环境、KV 参数与绑定矛盾时阻塞并返回交互阶段。命令强校验可交给上游专门 Agent，本技能仍检查结构、绑定、依赖一致性与运行证据。

规划时按审阅规则为每个当前参考脚本建立完整参数清单，覆盖 export、unset、bootstrap、CLI、完整 JSON 和代理参数，并逐项核对规范化 job_spec 与 scripts 模式生成内容。现场绑定差异必须展示原值、解析值及依据；遗漏或未解释差异未解决前不得进入 FULL。`preserve_reference` 标志、Schema 校验通过或生成成功都不证明参考参数已经完整转录。清单保存在仓库外审阅目录或生成后任务的 `.runtime`，不增加任务根目录杂项。

## 唯一执行入口

Python >=3.10，安装 requirements.txt。新任务仅用 [统一 CLI](docs/compact-cli.md)：

```sh
python3 scripts/review_job.py --job-spec /input/job_spec.json --bindings /input/bindings.json
python3 scripts/run.py --job-spec /input/job_spec.json --bindings /input/bindings.json --run-dir /work/JOB_ID --mode scripts
# FULL 用未使用的任务编号/目录，不覆盖 scripts 输出。
python3 scripts/run.py --job-spec /input/job_spec.json --bindings /input/bindings.json --run-dir /work/OTHER_JOB_ID --mode FULL
```

scripts 不访问服务器，生成可手工检查、执行与复现的脚本。FULL 在 Linux 主节点运行一个协调进程，按相同脚本分发、创建容器、部署、冒烟、测试、有限重试、释放和汇集产物；远端没有第二个协调进程。启动前检查设备占用和端口，失败不抢占其他任务资源。

旧 plan_run/submit_job/run_task/节点 RPC 等公开命令已停用。历史冻结任务使用其自带代码。仓库保留的旧内部实现供回归比较，不构成第二套公开运行协议。

## 配置与运行约束

- job_spec 是唯一任务需求源，bindings 是物理机器、路径、网络与端口映射。默认值在交互确认阶段呈现；生成不会引入额外调参策略。
- `execution.proxy_start_after_models_healthy` 默认 true，生成代理启动前的有界逐模型健康门控；审阅须展示默认值和路由对应的健康 URL。显式 false 保留提前启动行为，须说明参考时序差异，详见审阅规则。
- `execution.enable_cache_reset_api` 默认 true，为非 proxy 实例补充 `VLLM_SERVER_DEV_MODE=1`；显式 0 与 true 冲突时报错。false 不注入，也不删除已有值。`case.reset_cache` 独立控制清理，默认 false；开启后逐模型实例核验空闲、直接重置并核验成功，再执行原生预热和测试。
- `launch.kv_transfer_config` 表达完整原始 KV JSON，保留额外字段；connector、角色、engine_id 和端口须与拓扑一致。省略时沿用结构化连接的确定生成规则，交互确认时展示生成结果。
- case.mode=perf 使用原生 AISBench；accuracy 仅预留接口，当前拒绝执行，不回退成性能测试。原生配置保存原文，运行前核验模型、端点和负载，运行后核验预期模型×数据集结果组。
- 全部服务健康且入口冒烟通过后才测试。缺失结果、worker 日志或部署日志不能报告整体通过。指标采集失败记录 UNVERIFIED，不把未知命中率当成零。
- 只清理由本次执行标识确认拥有的容器。远端任务目录保留独占执行声明，禁止另一协调器重放同一任务。释放无法核验时停止推进；不删除锁后盲目重跑。
- 支持 after_case=keep_service/stop_container、task_end=stop_container；stop_service 与复杂自动恢复当前阻塞。默认子任务串行；显式 stages 支持阶段内独立分支并行、分支内串行，阶段间强制释放和产物屏障（见统一 CLI）。NODE 表示物理机器，不是 P/D 角色。

## 目录、日志与完成条件

根目录只有 job_spec.json、bindings.json、status.json、scripts、logs、results 与最小 .runtime，没有逐阶段整套技能快照。status 只由协调器写入，并由 status.schema 校验。bindings.schema 在生成前生效，shared_storage.schema 校验共享声明，FULL 分发前验证真实共享。

logs/coordinator.log 记录协调决策；logs/DEPLOYMENT-runNN/NODE-INSTANCE.log 记录服务输出；results/SUBTASK-tryNN 保留 AISBench 打屏、原生结果、worker 日志、清缓存和指标证据。手动执行不伪造协调器完成状态。

提交前告知进展、服务日志和结果的实际路径。按用户授权推进，不因内部流程重复索要许可；设计讨论和手工执行不自动创建监控。目录不自动删除，资源释放已核验、所需结果已保留且无异常后可由外层释放；异常证据保留。

本地 mock 和 Bash 测试不代表真实 Ascend/Linux/Docker/AISBench 版本验证。报告区分本地回归、Linux 集成与实际模型测试。
`run.py` 支持节点 `mount_profiles` 模板及显式 mounts 覆盖，另自动挂载任务目录和模型目录。规划时根据现场依赖选择模板，展示展开与覆盖结果；见 [挂载模板](references/mount-profiles.md)。数据集、代理和诊断程序仍须单独审阅。FULL 创建全部部署容器后，依赖预检全部通过才启动模型；dependency_timeout_seconds 默认120秒。预检使用镜像ENV和已确认launch.env，不执行bootstrap，规划须显式审阅依赖环境。

同一任务的不同部署需要不同代理程序时，在对应 proxy 实例设置 `launch.proxy_program_key`（默认 `proxy_program`，只允许字母/下划线开头的标识符），选择 `bindings.nodes[节点].paths[该键]` 的物理路径。例如 A/B 使用 `reference_proxy`，C/D 使用 `image_proxy`；同一节点同时绑定两个路径，无须切换文件或 wrapper。该字段仅供 proxy 使用，模型实例声明会被拒绝；缺少所选路径时生成阻断。它不改变代理参数、路由或健康门控。
