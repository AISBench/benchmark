# 容器挂载模板

节点 bindings 可设置 `"mount_profiles": ["ascend-runtime", "ascend-hixl"]`。

- `ascend-runtime`：只读挂载 `/usr/local/Ascend/driver` 和 `/usr/local/dcmi`。
- `ascend-hixl`：只读挂载 `/etc/hixlep`；当前连接实现需要时选择。
- 省略或空数组保持原有显式挂载行为。模板不随本地路径存在与否变化，scripts 模式无需访问服务器。

根据现场和参考脚本识别依赖，显式选择模板并展示展开与覆盖结果。显式 `mounts` 以 target 覆盖模板 source 和 read_only，支持不同宿主路径；重复显式 target 拒绝。review_job 探测展开后来源；容器脚本在 docker run 前检查全部来源，缺失即失败。

npu-smi 在镜像中缺失而参考 CPU 绑定等逻辑需要它时，探测宿主实际可执行路径并显式只读绑定到容器调用路径。不要同时猜测挂载 /usr/local/bin 与 /usr/local/sbin 两处。仅用于可选诊断时，不将其缺失视为模型运行必需依赖。

模板不推断硬件代际，不挂载整盘 /mnt、/home 或 /usr/lib64。数据集、代理文件、rootinfo 等按实际来源另行绑定；task/model 仍由生成器添加。模板不证明镜像依赖可用，FULL 实际容器依赖屏障仍须通过。
