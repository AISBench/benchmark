# 使用说明

从[快速开始](../README.md)提供服务器、工作目录、镜像、权重、参考脚本和负载需求。前缀测试使用 AISBench 原生合成；真实数据测试提供文件路径及字段映射。

1. 按[审阅规则](../references/job-spec-review.md)核对参考参数和现场绑定，形成 job_spec 与 bindings。
2. 按[统一 CLI](compact-cli.md)生成并检查脚本；远程执行前发布冻结运行时，使用新任务目录启动 FULL。
3. 查看 status.json、服务日志和 AISBench 原始输出。完成后分别核验请求结果、资源释放和产物汇集，不将生成成功等同于执行成功。

[原生 CLI 适配](native-cli.md)说明配置与结果语义；[原生前缀数据](generated-data.md)说明目标长度和前缀比例；[设计说明](design.md)说明组件职责与执行边界。
