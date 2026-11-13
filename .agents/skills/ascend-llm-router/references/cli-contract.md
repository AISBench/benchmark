# Public CLI contract (0.1.0a13 / protocol 1.0)

## 准备、审阅与发布请求

以下均为 `deploy-eval-v2 COMMAND --request FILE` 的公开 JSON 请求。表中字段全部必填，不添加阶段命令的 request_id/expected_version 等字段。路径是控制端本地路径；远端路径仅来自 control.json 和冻结绑定。所有摘要使用 CLI 返回的原值。

| COMMAND | FILE 的字段 | 输出及约束 |
| --- | --- | --- |
| package-runtime | destination | 将当前已安装 CLI 打包到新的本地 ZIP 文件；返回 runtime_package，其中 archive、runtime_sha256 用于后续命令 |
| draft-review | plan, bindings, destination, runtime_sha256 | plan/bindings 为文件路径，destination 为新目录；生成 review.json 和 review-summary.json，初始为 DRAFT，必须按实际授权审阅后批准 |
| freeze-inputs | plan, bindings, review, destination, runtime_sha256 | review 为已审阅文件路径；返回 input_bundle.path 与 input_bundle.sha256，后者作为 bundle_sha256 |
| verify-inputs | directory, sha256, runtime_sha256 | directory 为冻结目录，sha256 为 input_bundle.sha256；只校验，不修改冻结输入 |
| check-ready | task_ref, runtime_archive, input_directory, runtime_sha256, bundle_sha256 | 需要 --config CONTROL；仅本地可执行性检查，不代表现场预检成功 |
| publish-task | task_ref, runtime_archive, input_directory, runtime_sha256, bundle_sha256 | 需要 --config CONTROL；发布包到授权任务目录，不能预先用其他文件占据该目录 |
| start-task | task_ref, runtime_sha256, bundle_sha256, nonce | 需要 --config CONTROL；两项摘要和 nonce 都必须是64位小写十六进制字符串（正则[a-f0-9]{64}）；nonce 一次生成并保存，未知结果先查询，不更换身份盲目重试 |
| stop-runtime | task_ref, nonce | 需要 --config CONTROL；使用原启动 nonce，仅停止已允许收尾的框架进程，不替代服务释放 |

例如 package-runtime 的请求文件内容为 `{"destination":"runtime.zip"}`。准备顺序为打包、审阅、冻结、check-ready，再由对应阶段技能发布与启动。destination 使用未存在的新文件或目录（类型按表）；冻结后不手工修改内容。prepare-inputs/export-contracts 见下文及 plan 技能。control.json 的任务键、冻结 plan.task_id 和 task_ref 必须一致。

nonce 是启动身份随机值，不是摘要计算结果，也不是32位GUID；可用控制端 Python `secrets.token_hex(32)` 生成64位值，保存到请求后始终复用。`INVALID_INPUT: startup digest` 表示 runtime_sha256、bundle_sha256 或 nonce 中至少一项格式不符合上述契约；不能因此把已发布包推断为已启动。对未知启动结果仍按身份与持久回执查询，禁止换nonce盲重试。

Production execution is guarded and implemented. Freeze validates schemas, semantic references and compiled inputs. check-ready is local only; it does not attest host readiness. publish-task installs immutable runtime/input packages; start-task verifies detached Python process and live socket identity. Real-host functional acceptance covers the three scenarios and bounded workload documented in the CLI README; it does not attest readiness of another host.

For a model-free Linux socket run use socket-start, documented in socket-validation.md. This explicitly simulates stages; the Agent must submit each subsequent phase itself through the normal CLI.

```text
python -m deploy_eval_v2 --config control.json deploy --request deploy.json
python -m deploy_eval_v2 --config control.json eval --request eval.json
python -m deploy_eval_v2 --config control.json release --request release.json
python -m deploy_eval_v2 --config control.json cancel --request cancel.json
python -m deploy_eval_v2 --config control.json get-status --task-ref task-1
python -m deploy_eval_v2 --config control.json wait-event --task-ref task-1 --after-seq 2 --wait-ms 30000
python -m deploy_eval_v2 --config control.json read-result --task-ref task-1 --artifact-ref task.jsonl --offset 0 --max-bytes 8192
```

control.json 是受信控制端配置：tasks 下以 task_ref 为键，值包含 host、root、allowed_root 和可选 python。root 必须位于 allowed_root 内且不能等于它。配置不含密码。

副作用请求公共字段：task_ref、request_id、expected_version、wait_ms（0—30000）；protocol_version 可省略，CLI 补为 1.0。deploy 加 plan_ref；eval 加 service_id、case_id；release 加 service_id；cancel 加 scope（task/operation）、operation_id。CLI 将 task_ref 映射为协议 task_id，kind 由子命令决定。业务配置放冻结 JSON，不扩展成零散 CLI 参数。

使用 `export-contracts --request export.json`（内容为 `{"destination":"contracts"}`）导出当前安装包的 Schema，command.schema.json 位于导出目录；它是传输内部请求契约，不是 plan Schema。Production plan_ref must match the frozen bundle digest.

## 事件与错误处理

- RUNNING：保存身份与 next_cursor，用 wait-event 继续观察。
- EVENTS：按 task_id+seq 去重；has_more 时继续补页，不能自行跳到最新游标。deployment.ready 还须核对健康/冒烟证据。
- WAIT_TIMEOUT：保持任务运行判断，继续使用安全游标；不重复粘贴服务配置。
- STALE_VERSION：重读事实并重新决策，不盲改版本重发。
- REQUEST_CONFLICT：同幂等键内容不同，停止并核对原请求。
- UNKNOWN_OUTCOME、协议坏 JSON、SSH 断开：先查状态和原游标，不创建新任务或重跑部署。
- RECOVERY_REQUIRED、release.unverified：保留现场，停止复用资源，核验归属与执行证据。

退出码 0 只表示成功提交/查询/等待；操作成功必须看业务事件。设计要求输入/状态错误 2、通信未知 3、确认操作失败 4，当前实现仍需完整集成验证。

普通响应预算 8 KiB、每页最多 10 关键事件；日志默认 8 KiB，显式最多 32 KiB，包含 JSON 转义与元数据。先按 artifact_ref 读取一次所需片段，再按 next_offset 续读；不从 SSH 获取整份服务日志。定位支持互斥的 --tail-bytes N、--line-start N --line-end M、--keyword TEXT；按返回的扫描游标续读。task.jsonl 为统一框架日志，service.json 为部署交接；results/ 与 logs/ 下只读相应证据，禁止目录遍历。
# Task cancellation clarification

`scope=operation` requires the exact active `operation_id`. `scope=task` may omit
that field; this supports cancelling an idle task holding an owned service. If an
operation ID is supplied it must match the active operation. Task cancellation is
durable and prevents later phases; cleanup completes asynchronously and its event
must be checked. Acceptance is not evidence of resource release.

本地业务输入：`prepare-inputs --request task.json`（详见 plan 技能）返回 plan/bindings 路径及待审阅默认值；`export-contracts --request export.json` 导出与 CLI 同版本的权威 JSON Schema。新日志布局仅用于新任务，不用新包接管历史冻结任务。远端日志归集未完成时 task.jsonl 记录 remote.log.pending，不能声称日志完整；原生服务/AISBench 日志保持原位。

CLI >=0.1.0a12：原子JSON产物采用与事件摘要相同的规范序列化，expected-sha256验证文件字节；不改旧包旧产物。read-result在runtime socket不可用时，可核对持久任务身份后只读本任务受限路径的产物；不恢复执行、不改变状态或授权其他任务。停机后仍按offset/next_offset有界取回日志，并独立核验完整性。
