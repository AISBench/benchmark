# Job 包与存储

本机使用 submit_job.py prepare 校验 JSON、在 --local-jobs-dir 原子分配编号并冻结任务，再生成 job_id.tar.gz。统一使用同一本地 jobs 目录；远端编号冲突时拒绝覆盖，保留证据，重新分配新任务。任务包以文件清单和 SHA256 校验完整性。协调端与节点的 host/work_dir 相同时只上传一次；主机别名应在 bindings 中统一，脚本不推断不同别名是否属于同一机器。

父任务目录为 work_dir/job_id；公共内容位于父目录，节点可变产物位于 nodes/<node_id>/：

- common/inputs/：原始需求与现场 bindings；common/plan.json、common/job_spec.json：规范化冻结计划。
- common/skill-snapshot/：唯一运行包，含执行代码、Schema、测试原始数据和代理。同一节点内直接复用，节点部署不再复制包。排除测试代码、MD指引、示例及字节码。
- .internal/：现场绑定、生成脚本、服务身份和协调器诊断；deployments/ 保存服务日志，cases/ 保存负载定义，sub_tasks/ 保存测试结果。公共工具源码直接从 common 执行，不为 attempt 改写源码。此处节点运行状态、服务日志及测试结果位于 nodes/<node_id>/ 下，父目录保留协调状态与冻结负载。
- dataset-cache/：按参数及内容摘要复用的生成数据。
- tmp/、cache/：容器通用临时文件和缓存。
- 状态、summary.json、release-results.json、storage-state.json：执行及占用证据。

上传时先独占创建 job 目录，压缩包也落在目录内。压缩包和解压文件校验成功后删除远端压缩包；失败保留目录，不自动覆盖或重试未知任务。现场预检在该目录执行，prepare 不启动测试。全部归档均由单个 job 目录覆盖，没有 inputs-*、packages-* 旁路暂存目录。

节点执行的 disk_guard 按节点 storage.root 统计常规文件的逻辑字节数，不跟随软链接；storage-state.json 保存统计及限额。每节点分别监控，按现有 admission/attempt/客户端运行中周期检查执行。此值不是文件系统块占用，也不包含共享镜像、权重、驱动日志和 Docker 元数据。容器关闭 Docker 日志驱动，服务日志仍写入任务目录；TMPDIR=/bench-tmp、XDG_CACHE_HOME=/bench-cache 通过 bind mount 映射到节点目录 runtime/<容器摘要12位>/tmp 和 cache，但第三方程序硬编码的外部输出不保证纳入统计。容器可写层随任务容器释放，不声称目录大小等于主机全部增量。

## 主要产物：deployment 与 sub_task

以下为 nodes/<node_id>/ 下的主要产物结构；父任务保留协调汇总及 common/node-map.json 节点映射：

```text
deployments/<服务配置摘要前16位>-<运行编号>/
  deployment.json
  <服务名>.log
sub_tasks/<sub_task_id>/attempt-01/
  sub_task.json
  tool/  # 原工具产物及 aisbench.log 打屏日志
```

摘要由实际service定义计算（含命令、地址及端口），完整SHA256存于deployment.json，不作为跨机器模型配置标识。每次模型启动分配新编号；proxy加入同次运行。服务日志从启动持续记录至服务停止，多个sub_task复用服务时不切断、不复制日志。sub_task.json保存case参数、attempt、部署运行编号和路径、UTC起止时间、状态与错误。通过部署编号和时间定位服务日志。

重试独立保存，失败结果不覆盖。tool保留原工具结果结构；aisbench.log 保存表格和日志，子进程结束后仅去除 ANSI 控制符、修复表格底边粘行，不改变数值。不以自制汇总替代原工具产物。

新任务的 attempt 直接保存于 sub_tasks，不生成旧 results/tool/console.log 兼容软链接。工具日志以 aisbench.log 为唯一完整记录，不再写 aisbench_all.log 或向 outputs 复制日志。runner-diagnostics.log 仅失败时保留。脚本、探测记录、进程登记、数据集缓存均为辅助材料，执行和资源释放核验完成前不得删除。当前脚本不自动删除辅助目录或远端工作区；Agent归档以主要产物为准，校验归档完整性与资源释放后按用户授权清理。数据集缓存策略不变。


## 公共内容与结果入口

新任务上传的代码快照、原始公共输入、冻结计划及上传清单统一保存在 common/。模型进程管理、case runner、PD proxy 直接执行公共快照；测试工具公共模块通过模块路径复用，不为每个 attempt 创建模块和 GSM8K 软链接。每个 attempt 保留 tool-config.json、AISBench 模型配置薄模板和工具原始结果，不生成工具源码副本或包装脚本。

cases/<case_id>/case.json 保存负载定义；deployments/ 保存连续服务日志；sub_tasks/ 保存各次测试结果。节点运行状态放在 nodes/<node_id>/.internal/；父任务协调状态及提交日志放在父目录 .internal/。上文 deployments/、sub_tasks/ 为节点目录下的相对路径，cases/ 为父目录下的冻结负载。根目录保留运行入口与汇总状态，便于查询。旧冻结任务继续使用自身快照，不自动迁移历史绝对路径。

### 显式共享挂载与有人值守验证
默认每个host为独立存储；相同绝对路径不是共享声明。bindings可添加：
```json
{"shared_storage_groups":[{"id":"weights-share","nodes":["node_p","node_d"]}]}
```
节点名称引用bindings.nodes；每组至少两个节点、节点不能重复参与组。协调端若需复用，必须与某个声明节点具有完全一致的host/work_dir。组内允许不同绝对work_dir；它们必须映射同一共享目录。探针使用同一相对文件名，各节点从自身work_dir读写，不能混用其他节点的绝对路径。
prepare冻结本地任务后、任何远端公共包上传前验证声明：各节点轮流在work_dir写独占随机临时探针，所有成员读取核对内容；每轮清理探针，失败阻塞、不静默回退。work_dir须已存在且可写。证据保存本地job目录shared-storage-verification.json，供有人值守审阅。验证通过仅允许上传去重，仍须完成其余预检和有人值守提交决策；prepare不启动服务。
上传回执只核验已声明且验证通过的组在本次上传后的可见性，不能替代共享声明；回执不可见阻塞。未声明的物理共享目录会在第二次独占创建时失败，不自动启用共享。
逻辑父job_id保持不变。节点身份为<job_id>-node-<node_id>，node_id限制1–32个ASCII字母、数字、下划线或连字符且以字母数字起始。节点物理目录为<work_dir>/<job_id>/nodes/<node_id>，包含部署日志、sub_tasks、磁盘统计、临时缓存和.internal运行状态；容器身份带node_job_id。common/node-map.json关联父任务、节点身份、本机绝对路径、归档相对路径及公共包。公共包只保留父任务common/skill-snapshot一份，各节点使用自身挂载路径访问。父任务根仍为容器挂载范围，容器登记保留主机隔离。重试与端口generation均留在对应节点目录内。保持失败证据，不覆盖未知任务目录。


### 代码、现场绑定和运行配置边界
本地仓库维护固定工具源码；同步安装前备份并核验摘要，冻结后通过公共common/skill-snapshot直接执行。prepare_benchmark_tool仅创建输出目录及源码摘要清单，不复制或字符串改写aisbench_test.py。
每attempt的tool-config.json是执行器传入工具的解析配置，包含case负载、现场service参数、DP和设备数。runtime_config读取它并映射已有CLI字段；该模式优先于CLI默认值。独立工具不设置BENCH_CASE_CONFIG时继续使用config.py及CLI。正式统计由case.steady_state选择，前缀填充固定default_perf。AISBench要求的temp_api.py仅是模型配置薄模板，保留阶段并发/输出长度转换，不是工具源码副本。
事件、pipefail/失败传播、日志规范化、metrics快照直接维护在公共aisbench_test.py；事件通过BENCH_EVENT_ROOT启用，链接登记通过BENCH_LINK_MANIFEST启用。输出及快照落在attempt工作目录。现场start-*.sh仍在IP/端口/路径绑定后生成并记录摘要，不能与固定工具代码混淆。
