# Ascend LLM CLI 整体设计

## 责任边界

技能处理自然语言、参数来源、用户授权与阶段选择；CLI 校验确定的 JSON、执行有界副作用并返回结果。稳定版完全独立。主 runtime 是独立 Python 进程，使用任务锁和 Unix socket；SSH client 仅观察或提交命令。无 systemd，没有第二套启动后端。

## 阶段数据流

部署 JSON → 现场绑定/审阅 → 冻结部署输入 → 预检/部署/健康/冒烟 → service.json。

endpoint + model 或 service.json，加 case JSON 和 tester 绑定 → 独立评测输入 → 审阅冻结 → 工具依赖与健康预检 → 性能负载 → result JSON。

部署任务可以永不评测；评测任务可以从未部署过的外部服务开始。同一个部署服务可用于多个独立评测任务。评测故障只清理评测客户端，不授予测试任务服务释放权。健康检查或冒烟失败不得产生 deployment.ready。精度测试是后续能力，当前不接入。

## 公开 JSON 与唯一 Schema

Schema 源仅在 src/deploy_eval_v2/schemas。通过 export-contracts 导出供 Agent 使用，不要求阅读 Python 实现。

prepare-inputs：deploy 输入为 stage/destination/deployment/bindings；deployment 是完整 deploy-plan，bindings 是 deploy-bindings。禁止内嵌测试，并要求 keep_service。eval 输入为 stage/destination/task_id/service 或 endpoint+model/cases/tester，可附评测 policy。输出 plan/bindings 路径、derived 默认值及 review_required=true。准备阶段永不自动批准。

service.json：包含 schema_version/task_id/service_id/ownership/endpoint/model、runtime/bundle 摘要、容器归属、保留策略、health_time 与 health_evidence_ref。引用的部署结果同时保存健康与冒烟证据。单份 service.json 是交接源，禁止再在每个操作目录复制服务副本。eval 使用其中的 endpoint/model/service_id；释放权不随对象复制传播。

result JSON：包含 task/operation/service/case 身份、execution_status、evidence_status、performance_validity、actual_load、performance 及原生证据引用。性能条目以明确字段表示单位，如 duration_seconds 和 request_throughput（req/s）；未知、缺失或无效证据不可填成零。只有证据 COMPLETE 才能标记性能 VERIFIED。

release JSON：保存每节点资源结果与 cleanup_status。release.unverified 保留 claims，不能以请求接纳或进程退出替代释放核验。

## 审阅与发布

1. package-runtime 请求为 destination，返回 runtime_sha256 和 archive。
2. draft-review 请求引用 plan、bindings 和 runtime_sha256，生成审阅摘要；根据实际授权形成 review（APPROVED、空 unresolved、非空 evidence_refs，绑定输入摘要）。
3. freeze-inputs 请求含 plan/bindings/review 文件路径、destination、runtime_sha256。冻结后不可手改。
4. control.json 的 tasks[TASK] 包含 host、root、allowed_root、绝对 python；root 必须是 allowed_root 的子目录。附加节点/tester 必须与冻结输入一致。
5. check-ready / publish-task 请求含 task_ref/runtime_archive/input_directory/runtime_sha256/bundle_sha256。前者仅做本地检查，后者发布不可变包。
6. start-task 请求含 task_ref/runtime_sha256/bundle_sha256/nonce。nonce 一次生成并保存，重试不得更换。

部署-only 不绑定 tester；外部 eval 的 runtime 在测试机运行，校验 machine_id 与工具解释器。工具存在不代表依赖可导入。tester.environment 是冻结的显式环境映射，preflight、负载、postflight 使用一致环境，不自动修复环境。

## 阶段命令和异步响应

```text
deploy-eval-v2 --config control.json deploy --request deploy-command.json
deploy-eval-v2 --config control.json eval --request eval-command.json
deploy-eval-v2 --config control.json release --request release-command.json
deploy-eval-v2 --config control.json get-status --task-ref TASK
deploy-eval-v2 --config control.json wait-event --task-ref TASK --after-seq CURSOR --wait-ms 30000
```

副作用命令公共字段为 task_ref/request_id/expected_version/wait_ms。deploy 附 plan_ref=bundle 摘要；eval 附 service_id/case_id；release 附 service_id。cancel 附 scope=task 或 operation，后者要求 operation_id。

阶段返回 EVENTS 或 RUNNING，交付 operation_id/task_version/next_cursor。deployment.ready 才交付服务；evaluation.completed 才交付评测结果；release.completed 才认为资源释放。WAIT_TIMEOUT 不取消任务。后继调用先读当前版本；同 request_id 同内容复用原操作，不同内容拒绝。结果未知先读状态和原游标，不创建新请求绕过。

## 持久状态和恢复

持久事件与诊断共用 task.jsonl；event 记录保留 seq，diagnostic 记录不占事件游标。恢复只重放合法 event，忽略合法 diagnostic；完整损坏行拒绝恢复，末尾半行可在独占任务锁下截断并把原始字节编码存回同一日志。锁保证并发日志记录不交错。

删除逐请求 commands/ 和逐操作 operations/ 状态副本；内存状态及 operation-index 根据事件重建。state.json/operation-index 是可重建查询快照。启动身份、nonce、进程启动时间与 boot_id 放 .runtime；这些是恢复控制数据，不是额外日志。不明副作用保持 RECOVERY_REQUIRED，不能自动重执行。

## 日志及目录

```text
TASK/
  inputs/              # 审阅冻结输入，含编译结果和摘要
  runtime/             # 不可变执行包
  service.json         # 仅成功部署产生
  task.jsonl           # 事件与框架诊断唯一流
  state.json           # 可重建状态快照
  results/             # 各次操作结果、必要的原生证据和索引
  logs/                # 仅服务原生日志
  .runtime/            # 锁、socket、启动身份和进程记录
```

子进程框架输出流式转为有界 JSONL 诊断；AISBench stdout 与其原生文件直接保留。异常堆栈也写 task.jsonl，不产生每组件 error.log。查询日志使用 offset/字节预算；事件查询使用 seq，两者不可混用。results 下保留机器复核必需的配置、退出码、原生汇总和请求明细；这些是结果证据，不再复制为框架日志。远端诊断通过已验证冻结身份的节点/测试器通道分块回收，以来源、摘要和字节偏移去重，集中持久化后校验全量摘要，再确认清空远端暂存。确认前发生断线或并发追加均保留源文件，并记录 remote.log.pending；同来源下一阶段响应后继续回收，绝不为收日志重复副作用。单次回收预算 15 秒；未回收完的暂存仍需保留，不能报告日志归集完整。

## 归属与生命周期

部署服务健康+冒烟成功后保留，不自动 eval 或 release。显式 idle 策略决定是否到期收尾；简化部署入口要求 keep_service。取消由 runtime 监督自有进程组清理；资源释放须核验完整容器 ID、task/token、设备与端口。stop-runtime 通过 nonce 认证的 socket 请求停止无活动资源的 runtime，再核验进程退出。服务尚存时不能直接停止 runtime。

## 验证边界

本地测试覆盖 Schema、独立输入转换、服务权限隔离、事件恢复、并发统一日志、真实子进程输出/取消及模拟执行器。真实模型拓扑及跨主机日志、取消/闲置释放/不明结果故障注入的完整现场验收不在本轮完成范围。无可用现场时不得自动连接历史服务器。

本版只用于新任务；历史冻结包、日志与导出保持原样。不自动把旧 events.jsonl 任务迁移或用新 runtime 接管。
