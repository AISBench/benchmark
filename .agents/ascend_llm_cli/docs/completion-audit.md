# 初版本地完成核验 — 2026-11-12

范围：按 `../../docs/deploy-and-eval-v2-design.md` v0.3 完成初版本地代码及必要验证。
终点是可进入已授权环境的上机测试/debug，不是实机验收通过或性能结论。
基线 HEAD：`74a7feffb76c32b23658d37ea681784b3a9b2474`。

## 设计落实与证据

| 范围 | 已实现 | 主要本地验证 |
| --- | --- | --- |
| 契约与编译 | 版本化请求/响应/事件/服务/结果；审阅摘要；参数保真；确定性拓扑、脚本及输入冻结；身份与摘要校验 | plan_model、compiler、review_summary、service_contract、wire_contracts、result_contract 测试 |
| 发布与启动 | runtime/input 包、流式上传、不可变安装、所有节点及 tester 目标绑定、nonce 启动回执、systemd 与 live socket 身份核验 | publication、input_publication、ssh_publication、publication_cli、launcher 测试；真实本地安装子进程 |
| 真实阶段 | Docker/SSH 节点执行；全容器/依赖屏障；健康/冒烟；本地及远端 AISBench；环境预检、缓存操作、原生结果/性能及合成数据审计 | deploy、docker_node、owned_executor、external_runtime、remote_evaluation、tester_worker、cache_reset、native_results、performance_evidence、native_postflight 测试 |
| 生命周期 | 任务/操作幂等；版本互斥；journal 重放；未知结果不重跑；取消及闲置收尾；全局 claims；容器身份、设备/端口释放核验；安全重试 | runtime、recovery、remote_guard、release、owned_executor、deploy 测试；真实子进程超时/取消与环回端口占用测试 |
| 输出预算 | stdout 8 KiB、摘要 1 KiB、每页最多 10 事件；显式日志最多 32 KiB；文件日志、有界管道/HTTP/结果解析、游标及日志定位 | budget、artifact、transport、event_delivery、native_results 等测试；真实输出洪泛子进程 |
| CLI 集成 | 审阅/冻结/打包/检查/发布/启动及全部阶段/查询入口；退出码；预期结果摘要校验；离线 check-ready | cli、publication_cli、readiness 测试；安装版入口和完整测试套件 |
| 五技能 | 五个平级技能及阶段交接；CLI/协议声明；按名称依赖；安装校验与冲突保护；稳定版不变 | 五个 quick_validate 均通过；skill_bundle 测试覆盖安装重放、缺失依赖及冲突无覆盖 |

源码及隔离 wheel 安装环境各发现 **183 项测试：182 通过，1 跳过**。
跳过项为 Linux Unix socket 独立 runtime/client 集成；当前 Windows Python 无相应支持。
安装环境导入位置核验为 `.acceptance-venv/Lib/site-packages/deploy_eval_v2`，不依赖源码 PYTHONPATH。
wheel 构建、依赖一致性检查和 CLI 版本/帮助检查通过。

交付 wheel：`../dist/deploy_eval_v2-0.1.0a1-py3-none-any.whl`。
SHA-256：`0f450b6fad324410ec2f1027414d48a3575ae10958c5890fa07bb6216810654b`。

新增流水线集成测试使用真实环回 HTTP、真实 Python 子进程、原生格式文件和结果验证器；
其中 benchmark/preflight 是明确标记的 fixture，不是 AISBench、模型或 NPU 运行。
Docker、SSH、systemd 的本地验证使用注入响应，不声称证明真实宿主行为。

## 剩余上机测试与 debug

1. 获得明确的机器、账号、目录、设备/端口、服务操作权限及保留策略；核对实际版本和绑定。
2. 在 Linux 运行 Unix socket 测试，要求零跳过；实际核验发布、systemd 启动及 SSH 断线重连。
3. 以批准负载执行外部 eval 和 owned deploy → eval → release；核对 AISBench/驱动/容器/模型兼容性。
4. 按 host-acceptance.md 注入失联、取消、部分失败、runtime 重启、闲置超时及重复请求；随后扩展多节点和远端 tester。
5. 对现场原生结果格式与 prefix/稳态证据进行 debug；缺失证据保持 UNVERIFIED，不改变用户负载以取得通过。

初版采用保守恢复：未知副作用进入 RECOVERY_REQUIRED，保留 claims 和证据后现场核验。
不把自动跨节点修复、主机断电后的自动恢复、聊天关闭后的自动唤醒列为初版缺失代码。
非零预热无法从包含正式请求的累计计数中独立证明覆盖；多 DP 无明确 series 身份时也不推断覆盖。
这些结果有明确证据不足原因，不填写零值、不报告虚假的性能有效。

未连接历史服务器，未操作真实模型/NPU，未修改个人已安装技能。稳定版 tracked diff 为空；
保留现有 docs、exports 及其他修改，无提交、无推送。
