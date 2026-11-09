# 交互审阅与确认

1. 读取当前部署参考的每个模型、代理和配套启动脚本及 NOTES；建立逐脚本完整参数清单，记录来源路径、摘要、行号或片段及目标实例。覆盖全部 export（包括未被命令直接引用的环境变量）、unset、bootstrap 操作及顺序、CLI 参数和值、完整 JSON（含嵌套和额外字段）、代理启动参数与路由。追踪当前配置实际引用的配套文件和变量展开；无法确定的有效值保留 unresolved。区分有效命令与注释、停用材料，不自动执行历史补丁。源脚本是待审阅输入，不直接执行。
2. 读取权重 config.json、tokenizer 与权重位置，采集给定机器的网卡/IP、设备、镜像、代理及通信文件和数据集路径。运行 scripts/review_job.py 获取证据；REQUIRES_AGENT_REVIEW 表示还需 Agent 判断。
3. 规则核对 DP×TP×PCP 与设备、路由覆盖、connector 与并行依赖、地址和端口。Agent 审阅架构/量化支持、显存和上下文/rope 配置；证据不足保留 unresolved，不臆造兼容结论。设备形态固定，不另行猜测硬件方案。
4. 在交互阶段纠正源脚本中失效 IP、端口和权重路径，呈现修改依据。模型负载、并行和其他语义参数的调整在确认前明确说明。
5. 用清单逐项对照规范化 job_spec、bindings 解析值及 scripts 模式实际生成的实例和代理脚本，记录每项落点与结论。export/unset 的作用域与先后顺序、bootstrap 工作目录和依赖、CLI 重复参数及 JSON 嵌套字段均需核对，不能仅比较已提取到配置中的键。反向检查生成脚本新增的默认值和派生参数（包括 enable_cache_reset_api），展示完整 KV 生成结果。
6. 每项现场绑定差异列明参考原值、确认值、解析依据和生成落点；仅将已明确展示的现场映射记作绑定差异。语义调整不能伪装成绑定修正，须按第 4 步审阅。遗漏、无法表达的命令或无法解释的增删项须记录 unresolved，在解决并重新核对前不得进入 FULL；不以环境默认值、镜像隐式设置或 preserve_reference=true 代替显式参数保真证据。确认后 job_spec 成为唯一任务需求源，bindings 仅解析显式现场字段，执行遇到冲突报错。

参数清单是规划证据，不是第二份执行配置。保存到仓库外审阅目录，或在 scripts 输出生成完成后保存到该任务 `.runtime`；不要预建 CLI 将拒绝覆盖的 run-dir，也不要在任务根目录新增清单文件。记录所核对输入与生成脚本摘要；FULL 使用独立新目录，执行输入如有改变须重新生成并核对。Schema 校验、review_job 探测或 scripts 生成成功均不能替代这一逐项核对。

NOTES 中的跨实例启动顺序也须列入清单：逐项核对前置实例、就绪条件、失败阻断及其在正式 job_spec 和生成脚本中的落点。需要启动前健康门控时，可在 `launch.bootstrap` 中显式表达有界等待与失败退出；不能把进程已启动当作健康，也不能只依赖生成脚本隐式启用 `set -e`。顺序差异如拟判定等价，须在 FULL 前展示源码或现场证据、适用条件和审阅结论；无法证明或未解释的差异保留 unresolved，不得执行。最终报告事后披露不能替代执行前审阅。

命令强校验由上游专门 Agent 负责，本技能仍检查结构、依赖、资源归属与结果证据。原生 AISBench 配置保存原文，目标/负载不一致时回到交互阶段修正后重新确认。
`launch.args.device-ids`、`ASCEND_RT_VISIBLE_DEVICES` 和 `ASCEND_VISIBLE_DEVICES` 是显式设备选择，必须保留并与 placement.devices 的设备及顺序一致；冲突时报错，不把 CLI 替换成环境变量。未指定任何选择时，保真模式默认生成 ASCEND_RT_VISIBLE_DEVICES，其他模式默认生成 device-ids；交互规划须展示派生值及选择机制。
多 P/D 审阅须核对同一 consumer 所接收的各 producer 的有效 engine_id。MooncakeConnectorV2 按远端 engine_id 缓存元数据，不能靠不同 IP/端口区分重复 ID；程序拒绝同一 consumer 的不同 producer 使用相同声明 ID。DP 的 _dpN 后缀只区分单个部署中的 rank，不保证独立 producer 唯一。采用实例唯一的基础 ID，并结合指定镜像构造逻辑核对有效 ID 集合（含不同基础 ID 经后缀变换后可能重叠的情况）；未核验的 connector 不套用这一实现假设。ID 调整属于需展示依据的参数调整，确认后冲突直接阻塞，不静默重命名。
