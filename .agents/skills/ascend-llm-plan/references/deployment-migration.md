# Deployment contract migration

The deployment item Schema is extracted from the immutable baseline commit
74a7feffb76c32b23658d37ea681784b3a9b2474. Exact provenance is recorded in
deployment-schema-source.json. No mutable stable module is imported. The test
fixture is the deployment item of that commit's job_spec.compact.json; it is
illustrative data, not authorization to access any server.

| Historical input | v2 deploy plan |
| --- | --- |
| job_id | task_id (v2 identity syntax) |
| deployments[i] | deployment (one reviewed deployment per stage plan) |
| deployments[i].instances[].launch | unchanged, including bootstrap/env/args/full KV JSON |
| execution | execution, with timing/gating/retry choices explicit before review |
| unresolved | unresolved |
| cleanup | explicit policy; no automatic equivalence mapping |
| cases/sub_tasks/stages | separate eval/routing inputs; not silently discarded by a converter |

This is a contract, not an automatic old-job converter. Multiple deployments must
be planned as separate stage operations; existing groups/stages have not yet been
migrated. There is no claim that old FULL lifecycle behavior has been verified.

Current validation covers Schema, unique instance identities, endpoint references,
KV/routing roles, proxy/model launch distinction and basic topology completeness.
It preserves extensible launch.args and complete KV payloads without mutation.
Bound compilation now validates binding Schema/IPv4, references, port spans,
environment-vs-network conflicts and device overlap through the frozen topology
helper. It expands full KV JSON, argv/env/bootstrap and mount profiles without
modifying inputs. Image-specific KV compatibility and host prerequisites must still
be checked on site by the guarded deployment executor. Provenance is in topology-migration-source.json. Input freezing
checks this plan but still reports production_ready=false; it is not deployment
approval. Existing review digest checks remain necessary.

## 公开参考脚本绑定规则（CLI 0.1.0a8）

参考脚本是用户提供的部署配置来源，不是要由Agent直接运行的旁路。将启动前shell内容放在 launch.bootstrap；将确定的export放在launch.env；vllm serve后的选项放在launch.args（键不带--）。unset和source保留在bootstrap；依赖shell展开的export（如LD_LIBRARY_PATH）保留bootstrap，不能在env中写$变量并期待展开。最终执行顺序为：日志重定向、bootstrap、env导出、exec编译后的argv。bootstrap中使用的绑定变量必须在bootstrap自身赋值，不能依赖随后才导出的env。

编译器对args/env/完整kv_transfer_config递归展开双花括号；bootstrap中的替换会执行shell安全引用，无须再给占位符套引号。整串占位符在JSON中保留值类型，嵌入字符串只允许标量。支持的上下文如下：

| 占位符 | 公开来源 |
| --- | --- |
| {{model.path}} | nodes[placement.node].model_paths[deployment.model.id] |
| {{model.served_name}} | deployment.model.served_name |
| {{network.ip}} / {{network.interface}} | nodes[slot].deployments[deployment.id].network |
| {{ports.http}} / {{ports.rpc}} / {{ports.kv}} / {{ports.hccl}} | nodes[slot].deployments[deployment.id].ports[instance.id]中同名端口基址 |
| {{paths.KEY}} | nodes[slot].paths.KEY，须为已确认绝对路径 |
| {{task_dir}} | nodes[slot].work_dir + '/' + task_id |
| {{run_dir}} | task_dir + '/logs/' + instance.id |

参考脚本LOCAL_IP→{{network.ip}}、NIC_NAME→{{network.interface}}、MODEL_PATH→{{model.path}}、HTTP_PORT→{{ports.http}}、RPC_PORT→{{ports.rpc}}、KV_PORT→{{ports.kv}}、HCCL_BASE_PORT→{{ports.hccl}}、RUN_DIR→{{run_dir}}。例如bootstrap可为 `LOCAL_IP={{network.ip}}\nRUN_DIR={{run_dir}}\nmkdir -p "$RUN_DIR"\ncd "$RUN_DIR"`（JSON中的\n解析为换行）；env可含 `"HCCL_IF_BASE_PORT":"{{ports.hccl}}"`；args可含 `"data-parallel-size":4,"tensor-parallel-size":1,"max-model-len":1048576`。这里只说明语法，数值必须来自用户参考，不能用示例替代实际参数。

完整KV JSON可放launch.kv_transfer_config，保留所有参考键和值，仅将现场绑定值换为上述占位符；不得只保留connector名称。模型launch以preserve_reference语义编译：不隐式增加性能环境默认值；部署拓扑仍管理设备可见性、HTTP/RPC地址端口、served model和KV一致性。显式参数与绑定冲突会拒绝冻结；不要用删参数解决冲突。布尔true产生--key，false产生--no-key；例如参考--no-disable-hybrid-kv-cache-manager对应disable-hybrid-kv-cache-manager=false。JSON对象/list参数保持结构，最终作为一个JSON argv值传递。

placement.devices是物理设备ID；同一物理主机只使用一个node身份，各模型实例设备集合不可重叠。实例ports值是端口跨度宽度，bindings同名值是基址，所有跨度必须无冲突；不要将基址误写成跨度。

### 容器设备与挂载的实际契约

CLI 0.1.0a13 的 runtime.container 默认值为 user=root、pid_mode=host、shm_size=500g；privileged 必须显式审阅为 true/false。未指定 container.devices 时，默认设备路径为 /dev/davinci_manager、/dev/hisi_hdc、/dev/ummu、/dev/uburma 和 /dev/davinci0 至 /dev/davinci7。/dev/devmm_svm 不属于此默认集合；不能仅因历史平台的设备路径不存在就推断当前默认集合缺失。通过 draft-review 的 effective_container/container_defaults 及冻结 compiled.json 核对本次实际值，再按实际值现场检查。显式 devices 替换默认集合时必须有平台和现场依据，不为避开预检任意删设备。

container.devices 是容器可见设备集合，placement.devices 是模型实例使用的物理卡集合；仍须完整保持用户 DP/TP 拓扑且禁止抢占在用卡。设备路径可为目录（如 ummu/uburma），以现场存在性及执行器预检为准。

nodes[slot].mount_profiles 的 ascend-runtime 展开为 /usr/local/Ascend/driver 和 /usr/local/dcmi 的同路径只读挂载；ascend-hixl 展开为 /etc/hixlep 的同路径只读挂载。显式 mounts 中同 target 的条目覆盖对应 profile 默认值。权重、代理资产和参考所需其他路径另行明确挂载并核验，不能假设旧服务器目录存在。审阅和编译产物给出有效绑定；现场必要路径确实缺失时报告具体缺口，不修改驱动或共享环境。

proxy实例不占设备。launch.proxy_program_key指向nodes[slot].paths中的代理文件键（默认proxy_program）。编译器从connections.routing和各实例绑定生成--host、--port、--prefiller-hosts/ports、--decoder-hosts/ports；这些参数不得再写入proxy.launch.args。workers、重试等其余参考参数需保留在args，参考source/unset保留bootstrap。P_IP/D_IP及P_HTTP_PORT/D_HTTP_PORT由路由引用的实例绑定表示，不支持跨实例{{ports.other.http}}语法。

用户提供的proxy.py若现场不存在，可在授权工作根中的本轮独立assets目录用文件传输发布原文件并核验本地/远端SHA256，将该绝对路径记入paths和只读mounts。这是部署输入准备，不是启动旁路；不要直接运行参考脚本或手工拉起模型。任务根本身保留给publish-task，避免预先填入文件导致发布冲突。镜像/权重/通信目录与用户资产的挂载都须在bindings明确记录；comm/hixl路径不能从历史现场推断。

先按上述规则形成完整plan/bindings，再draft-review、freeze-inputs、check-ready；核对冻结编译产物中的实际argv、env、bootstrap、设备、挂载、端口及参考差异。部署+评测可以使用deploy-plan的evaluation与deploy-bindings的testers；prepare-inputs的stage=deploy专供只部署，拒绝内嵌评测。是否收到deployment.ready后进入eval由用户意图和router交接决定。

### 模型参数完整迁移表
以下参数必须迁移到结构化字段，而不是从部署行为中删除。审阅时逐项确认编译后的argv仍具有相同语义；此迁移不违反保留参考参数的要求。

| 原始命令参数 | 结构化位置（不要重复放launch.args） |
| --- | --- |
| --host | bindings.nodes[slot].deployments[deployment.id].network.ip |
| --port | bindings.nodes[slot].deployments[deployment.id].ports[instance.id].http |
| --served-model-name | deployment.model.served_name，须保留参考实际名称 |
| --kv-transfer-config | instance.launch.kv_transfer_config，保留完整JSON |

以上四个键在launch.args中出现即拒绝冻结。data-parallel-rpc-port/data-parallel-address可以省略，由rpc绑定生成；若显式保留则必须与绑定一致。device-ids或ASCEND_RT_VISIBLE_DEVICES/ASCEND_VISIBLE_DEVICES如果参考显式给出，需要与placement.devices一致；否则编译器按placement补设备选择，不改DP/TP/PCP规模。preserve_reference会保留其余环境及参数，不取消上述结构化字段的一致性管理。

完整KV JSON中的kv_port必须匹配本实例ports.kv，kv_role和connector必须匹配connections.kv_transfer；engine_id在launch.engine_id与完整KV JSON之间必须保留一致，且遵循参考。不要将JSON参数同时放在launch.args和专用launch.kv_transfer_config中。所有受管理参数迁移都需在审阅差异中记录原参考值、目标字段和值及编译结果。
