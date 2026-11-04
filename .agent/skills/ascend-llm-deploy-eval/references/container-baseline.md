# 昇腾服务容器基线

此环境的模型服务容器遵循用户提供的 docker run 基线，由 scripts/container_baseline.py 统一维护，deploy.py 与现场绑定生成共用它。

- root 用户、交互终端、后台运行、host 网络、host PID、500g 共享内存、privileged=true。
- /dev/davinci0–7、davinci_manager、hisi_hdc、ummu、uburma 全部必需。设备暴露不代表允许模型超出 placement 指定的卡运行。
- 同路径挂载：/root/host、/usr/local/sbin、/usr/local/sbin/npu-smi、/usr/local/bin/npu-smi、/usr/local/dcmi、/mnt、/usr/lib64、/home、/etc/hixlep、/etc/hccl_rootinfo.json、/usr/local/Ascend/driver。
- /var/log/npu 挂载为 /usr/slog。用户原稿重复的 npu-smi 挂载只保留一次。
- 保留任务工作目录和模型路径的额外挂载；容器工作目录继续使用任务服务目录以便保存归属和阶段证据。

上述为默认值，用户特殊指定优先。job_spec 的 runtime.container 支持 shm_size、privileged、user、pid_mode、devices 覆盖；plan_run 补齐并冻结默认值。网络保持 host。设备覆盖必须包含部署分配的卡。

bindings.mounts 按 target 合并：未指定的目标补默认挂载，已指定的目标保留用户 source 和 read_only；重复目标报错。默认清单见 examples/bindings.example.json。预检检查合并后全部来源及设备存在，缺项明确报出路径并阻塞，不自动创建空文件或静默跳过。容器配置与完整设备挂载进入身份摘要，禁止复用不兼容容器。

自动执行保留任务目录作为工作目录，使用 sleep infinity 保持容器运行，再通过 docker exec 启动服务；用户原稿的 /home 工作目录与交互 bash 不替代任务归属和生命周期管理。其他尚未建模的特殊容器要求需明确适配，不能静默忽略。

镜像工具发现使用的 -000 临时探测容器仍为无网络、无设备的只读工具检查路径；它不启动模型服务，不套用服务容器权限基线，检查结束后删除探测容器。

端口由现场动态分配，启动前复核。端口绑定失败须报告端口和 errno，不能只记录 Address already in use；不得据此清理未知资源。HCCL 拓扑错误需独立核验 rootinfo、hixlep 和驱动拓扑，不把它直接归因于 HTTP 端口。
