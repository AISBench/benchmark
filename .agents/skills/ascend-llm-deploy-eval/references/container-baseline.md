# 昇腾服务容器基线

当前公开入口为 `scripts/run.py`。容器参数由 `container_baseline.py` 提供默认值，挂载由 `mount_profiles.py` 和显式 bindings 确定；旧 plan_run、task_preflight 的挂载清单不应用于当前 FULL。

- 默认 root 用户、host 网络、host PID、500g 共享内存；privileged 必须显式选择，示例默认/推荐 true；创建时使用后台容器和交互终端。
- 默认暴露 /dev/davinci0–7、davinci_manager、hisi_hdc、ummu、uburma。设备暴露不代表允许模型超出 placement 指定的卡运行。
- `runtime.container` 可覆盖 shm_size、privileged、user、pid_mode、devices；网络保持 host。现场审阅应展示实际容器参数及权限。即使使用推荐值，也必须写出 `"privileged": true`；省略时校验失败，不由 normalize 或 Docker 参数生成器隐式提权。

## 当前挂载规则

节点配置为 `bindings.nodes.<node>`：

- `mount_profiles` 显式选择驱动和通信挂载模板，具体路径见 [挂载模板](mount-profiles.md)。省略时不追加模板挂载。
- `mounts` 按 target 覆盖模板的 source 和 read_only；重复显式 target 报错。
- 生成器额外挂载任务目录（可写）和模型目录（默认只读）。显式声明同 target 时，source 必须与绑定路径一致，任务目录不能只读；模型目录保留用户显式指定的访问模式。冲突时生成报错。
- 数据集、宿主代理文件、rootinfo、日志目录及需要的 npu-smi 工具按现场需求显式绑定。不自动挂载整盘 /mnt、/home、/usr/lib64，也不同时要求两处 npu-smi 路径存在。

完整示例见 [bindings.example.json](../examples/bindings.example.json)，与 [job_spec.compact.json](../examples/job_spec.compact.json) 配套；快速开始使用的 [bindings.compact.json](../examples/bindings.compact.json) 提供相同示例。示例路径、镜像、网络和端口均需现场审阅，不代表可直接在任意机器启动。job_spec 保持 draft，完成审阅后才设置 ready 并解决 unresolved。

容器脚本在创建前检查挂载来源及设备存在，缺项时阻塞，不自动创建空文件。FULL 创建全部部署容器后，在实际容器内检查模型依赖、选定代理文件和测试依赖；全部通过后才启动模型。

自动执行以 `sleep infinity` 保持容器运行，通过 `docker exec` 启动服务。任务日志和归属证据写入任务目录。启动前检查端口与设备占用，冲突不构成清理其他任务资源的授权。HCCL 拓扑错误需结合现场 rootinfo、hixlep、驱动及网络证据核验。
