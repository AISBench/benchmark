# 交互审阅与确认

1. 提取源脚本的模型环境、启动参数、完整 KV JSON、代理参数、拓扑和测试目标。源脚本是待审阅输入，不直接执行。
2. 读取权重 config.json、tokenizer 与权重位置，采集给定机器的网卡/IP、设备、镜像、代理及通信文件和数据集路径。运行 scripts/review_job.py 获取证据；REQUIRES_AGENT_REVIEW 表示还需 Agent 判断。
3. 规则核对 DP×TP×PCP 与设备、路由覆盖、connector 与并行依赖、地址和端口。Agent 审阅架构/量化支持、显存和上下文/rope 配置；证据不足保留 unresolved，不臆造兼容结论。设备形态固定，不另行猜测硬件方案。
4. 在交互阶段纠正源脚本中失效 IP、端口和权重路径，呈现修改依据。模型负载、并行和其他语义参数的调整在确认前明确说明。
5. 展示规范化默认值（包括 enable_cache_reset_api）、完整 KV 生成结果与脚本；确认后 job_spec 成为唯一任务需求源，bindings 仅解析显式现场字段。执行遇到冲突报错，不静默覆盖。

命令强校验由上游专门 Agent 负责，本技能仍检查结构、依赖、资源归属与结果证据。原生 AISBench 配置保存原文，目标/负载不一致时回到交互阶段修正后重新确认。
`launch.args.device-ids`、`ASCEND_RT_VISIBLE_DEVICES` 和 `ASCEND_VISIBLE_DEVICES` 是显式设备选择，必须保留并与 placement.devices 的设备及顺序一致；冲突时报错，不把 CLI 替换成环境变量。未指定任何选择时，保真模式默认生成 ASCEND_RT_VISIBLE_DEVICES，其他模式默认生成 device-ids；交互规划须展示派生值及选择机制。
