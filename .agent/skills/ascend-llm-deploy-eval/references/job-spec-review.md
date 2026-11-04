# 分层部署、可复用 case 与现场绑定

当前只接受 job_spec v3 的拓扑结构，不兼容开发阶段旧平铺字段。参考 examples/job_spec.serial.json 和 examples/bindings.example.json；示例不包含真实服务器、私有路径或镜像摘要。

配置生成后，先运行 plan_run.py --check-dependencies，再用 --validate-only 检查默认值补齐后的 Schema 和业务约束；依赖缺失或校验失败不得报告计划已冻结。仅校验允许 draft，但冻结仍要求 ready 且无未决项。仅校验不写任务目录或快照、不保留编号、不检查远端资源；使用 python3 -B 避免 Python 写入字节码缓存。

新任务省略 job_id，使用 --jobs-dir 指定统一任务父目录、--date 指定用户时区日期（YYYYMMDD）。脚本分配 YYYYMMDD-NNN（001–999），跳过已有目录，冻结时原子创建目录；校验显示的候选编号可能在冻结时变化，以 PLANNED 输出为准。冻结失败留下的已保留目录不自动删除或复用。此分配只保证同一父目录内不重复；同一执行环境应使用统一父目录。显式 job_id 仍由 Schema 校验，使用 --run-dir。负载说明放 case ID，不拼入任务编号。

## 三类定义

- deployments：每项为完整服务拓扑，包含 id、topology、model、runtime、instances、connections、endpoint。
- cases：仅描述负载：数据集标识、最大输出 token、并发、请求数及可选输入长度校验，不引用部署。默认值见下节。
- sub_tasks：显式串行执行清单，id、phase_id、deployment_id、case_id、after_case、on_failure。phase_id 默认 pp1，仅为分组标识，执行仍按清单顺序；on_failure 当前仅支持 record_only。一个 case 可以被多个子任务引用；结果用 sub_task ID 隔离。当前不支持 phase 并行屏障。

## 配置默认值

数据集通过 dataset_id 引用现场路径；不生成前缀或测量目标命中率。默认 formal_requests=concurrency×4、request_rate=0、num_warmups=1、reset_cache=false，用户显式要求优先。steady_state只选择正式结果的原生统计器。

重试、缓存、冒烟和服务释放的运行规则集中在[生命周期](service-lifecycle.md)。

## deployment

model 存逻辑模型 id 与 served_name。runtime 存共用镜像和容器要求。instances 中每个进程有角色 mixed/prefill/decode/proxy，以及 placement（逻辑 node、devices）、launch（模型 args/env/bootstrap 或外部 proxy adapter）、ports、checks。

同一部署同一节点的实例共用容器。容器按任务和现场配置管理，不以 deployment ID 分配：相同主机、镜像及容器配置复用，预检汇总各部署的设备与模型挂载；不兼容配置另建子编号容器。单机 PD 的 P/D 设备互不重叠，双机 PD 的 P/D 分别放到不同 node。当前 PD 路径只实现一个 P、一个 D 和一个 proxy；混部只有一个 mixed。DP/TP 必须匹配实例设备数。模型参数以实例 launch.args 为唯一来源，max-model-len 在每个模型实例定义，测试必须满足所有相关实例的上限。

connections.kv_transfer 关联 producer/consumer 和 connector；connections.routing 关联 proxy、prefillers、decoders。KV 配置中的 P/D DP/TP 与角色由实例关系生成。endpoint 引用对外服务实例和逻辑端口。禁止在 launch.args 重复填写由编译器管理的 host、port、served-model-name、device-ids、kv-transfer-config。

launch.args 中 true/false 渲染为开关及 no- 开关，数值/字符串为参数值，对象/数组编码为单个 JSON 参数；配置必须经过 Agent 审阅，不能把字符串 shell 当任意可执行命令。bootstrap 保留已审阅的环境加载步骤。当前以 vllm serve 和现场提供的 PD proxy 为执行适配器，不承诺任意服务框架。

## bindings

bindings.nodes 按逻辑 node 记录 host、work_dir、package、model_paths、paths、mounts，可选 network_interface。真实现场信息仅放仓库外。

容器按 [默认容器基线](container-baseline.md) 补齐参数和挂载；bindings.mounts 是按目标路径覆盖/追加的现场配置，省略不代表无挂载。runtime.container 的显式参数覆盖默认值，完整结果写入 resolved 并参与容器复用指纹。源路径不存在时阻塞，不能让 Docker 自动创建空目录代替通信配置文件。

Agent 通过 prepare 上传冻结包：独立存储分别上传，显式声明且验证通过的共享组仅上传一份，各节点 package 使用自身挂载路径访问；共享验证在上传前完成，证据供有人值守审阅，详见[Job存储](job-storage.md)，协调端需要能免密 SSH 到全部节点（包含自身作为节点的情形）。preflight 通过固定节点脚本和 JSON 请求发现环境并生成脚本，不远程执行输入中的 Python 代码。各节点权重、镜像解析、设备、端口与工具均分别检查。

默认不填写 network_interface：从 UP 且有 IPv4 的接口按当前协商速率选唯一最高者，保留 VLAN，排除 bridge/veth 等虚拟接口。通过 ip JSON 与 sysfs/ethtool 获取信息，不按 data0 名称选择。速率未知、并列或多地址需有人核验；网卡速率不代表跨机可达性。ip 命令损坏时尝试系统路径，不修改服务器 PATH 或网络。端口从系统临时客户端端口区间之外动态选择，启动前再检查；这不是永久端口租约。

## 执行产物

plan.json 固定需求和 sub_task 顺序；resolved-cluster.json 保存拓扑编译结果及节点注册信息。节点保存 discovery.json、各容器的 resolved.json、启动脚本和脚本摘要。resolved-1 是内部节点执行产物，不是另一种用户输入格式。归属摘要与实际容器/进程身份一并核验。配置事件定义见[生命周期](service-lifecycle.md#配置存储埋点)。

容器归属摘要与服务摘要分离：前者包含任务计划、镜像与容器配置，后者包含各部署命令和绑定。节点任务目录的 container-pools/<容器名>.json 登记实际容器 ID、当前服务目录及删除状态。旧服务释放核验后才能切换登记；协调器锁与容器锁使用不同名称。

## 命令与检查报告

规范化阶段移除参考命令中的ASCEND_SLOG_PRINT_TO_STDOUT与ASCEND_GLOBAL_LOG_LEVEL赋值；计划校验拒绝在launch.env/bootstrap中显式设置，避免启动细粒度算子日志。环境加载脚本仍需审阅，不保证外部脚本没有间接设置。

节点preflight-report.json记录检查时间、分组结果、失败原因及未执行项目；实际资源值查preflight-evidence.json、discovery.json。检查分组覆盖磁盘、镜像、挂载、设备、权重、tokenizer、工具、网络、资源占用或服务复用。权重检查为索引分片存在性，不是全文件内容校验。独立检查项全部执行并汇总异常，依赖缺失项目标为阻塞，不能计为通过。现有总体提交条件保持：冻结计划与节点登记一致、预检成功、任务未重复执行；终端只输出总体结论与失败原因。

### 原生统计口径

steady_state=true 使用 AISBench stable_stage 统计器；默认使用 default_perf。正式请求数有限，不自动开启 pressure 模式，不承诺一定形成稳态区间。输入长度与前缀特征取决于用户数据集，不采集或承诺目标前缀命中率。性能结果保留 AISBench 原始统计口径。

## 标准 CLI 现场输入

确认容器内已安装可运行的 AISBench，不要求额外插件。用户指定数据集绝对路径、JSONL/CSV格式及字段映射，在节点 bindings.benchmark.datasets 中登记，由 case.dataset_id 引用。数据集不进入交付件，不自动生成、复制或下载。PD 部署另需现场代理路径。使用默认日志落盘，并提供原生日志跟踪命令；详见[CLI适配](../docs/native-cli.md)。
