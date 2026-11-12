---
name: ascend-llm-plan
description: 为 Deploy and Eval v2 整理部署或独立测试需求、现场绑定、参数来源与审阅证据，生成审阅摘要并冻结结构化输入。
---

# Plan

依赖 deploy-eval-v2 CLI 0.1 与协议 1.0，先执行 `deploy-eval-v2 --version` 核对。共享 CLI 契约随 ascend_llm_cli 包提供；缺失依赖时明确报告，不切换到旧版。不把规划结果标为已经通过运行时冻结校验。

保留原始需求与显式参数；按部署、测试、保留/释放分别确定授权范围。已有授权继续有效，只有影响执行且无法推断的缺口才向用户澄清。未提供服务器及访问授权时只完成本地规划，不连接历史服务器。

结构化交接须覆盖任务身份及版本、用户目标、部署拓扑或外部服务、case、物理绑定、参数来源、资源归属、阶段超时、失败动作、空闲期限及到期动作、审阅证据与 unresolved。使用 draft-review 生成未批准摘要，按实际授权和审阅形成 review.json，再用 freeze-inputs 冻结；不能把草稿自动批准。运行 check-ready 核验支持范围，现场 machine_id 必须有证据，未解决项保持 draft。

部署需求逐项保留参考脚本的 export、unset、bootstrap、CLI、完整 KV JSON 和代理参数；每个绑定差异记录原值、解析值及理由。检查权重、镜像、网络、端口、设备、挂载和依赖可见性；只读取证与真实环境修改分开。

独立阶段先调用 `prepare-inputs --request task.json`：只部署使用 stage=deploy、destination、deployment（完整 deploy-plan）、bindings，并显式设置 keep_service，不嵌入 evaluation/testers；只评测使用 stage=eval、destination、task_id、endpoint+model 或 service 对象、cases、tester。输出待审阅 plan/bindings 路径及派生默认值，随后仍按现有审阅、冻结、发布链执行。

独立测试仅要求端点、served model、测试机/工具/数据绑定及权限；外部部署摘要未知就记录未知，不补造健康或部署来源。

默认值在审阅中展示：普通 formal_requests=concurrency×4，steady_state=true 时×10；显式数值优先。保留预热、输出长度、请求速率及缓存策略，禁止为缩短验证降载。清缓存是独立授权，端点可访问不表示允许清缓存。

端到端可建议 idle_timeout_seconds=900 后释放自有服务；用户要求保留则保留，不把建议当已授权。交接包含计划及绑定位置、摘要、未解决项、下一阶段及授权范围。存在 unresolved 时不进入真实执行。

## 本阶段参考与交付

依赖 CLI >=0.1.0a7、协议 1.0。规划时读取 [计划与绑定](references/plan-contract.md)、[审阅摘要](references/review-summary.md)；参考脚本迁移见 [部署映射](references/deployment-migration.md)，完整字段由 `export-contracts` 从已安装 CLI 导出；不维护技能内 Schema 副本。命令封装见已安装 ascend-llm-router 的 references/cli-contract.md。

按用户授权发现现场，生成 plan、bindings，执行 draft-review；消除 unresolved 并绑定实际授权证据后 freeze-inputs，再 check-ready。交付冻结目录、runtime_sha256、bundle_sha256、原始需求与差异映射，以及下一阶段所需控制端配置。候选 Python/镜像只是发现线索，缺失时允许只读寻找已有兼容候选；实际绑定必须有证据，不擅自安装或修改已有环境。

显式授权创建任务资源且给定工作根时，可创建该目录和新的任务子目录；不能越界、覆盖旧证据或修改 SELinux。tester.environment 冻结真实环境值，预检和负载使用同一环境。日志全部保留在用户工作目录。
