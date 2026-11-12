# Public CLI contract (0.1.0a4 / protocol 1.0)

Production execution is guarded and implemented. Freeze validates schemas, semantic references and compiled inputs. check-ready is local only; it does not attest host readiness. publish-task installs immutable runtime/input packages; start-task verifies detached Python process and live socket identity. Host acceptance is not yet complete.

For a model-free Linux socket run use socket-start, documented in socket-validation.md. This explicitly simulates stages; the Agent must submit each subsequent phase itself through the normal CLI.

```text
python -m deploy_eval_v2 --config control.json deploy --request deploy.json
python -m deploy_eval_v2 --config control.json eval --request eval.json
python -m deploy_eval_v2 --config control.json release --request release.json
python -m deploy_eval_v2 --config control.json cancel --request cancel.json
python -m deploy_eval_v2 --config control.json get-status --task-ref task-1
python -m deploy_eval_v2 --config control.json wait-event --task-ref task-1 --after-seq 2 --wait-ms 30000
python -m deploy_eval_v2 --config control.json read-result --task-ref task-1 --artifact-ref logs/runtime.log --offset 0 --max-bytes 8192
```

control.json 是受信控制端配置：tasks 下以 task_ref 为键，值包含 host、root、allowed_root 和可选 python。root 必须位于 allowed_root 内且不能等于它。配置不含密码。

副作用请求公共字段：task_ref、request_id、expected_version、wait_ms（0—30000）；protocol_version 可省略，CLI 补为 1.0。deploy 加 plan_ref；eval 加 service_id、case_id；release 加 service_id；cancel 加 scope（task/operation）、operation_id。CLI 将 task_ref 映射为协议 task_id，kind 由子命令决定。业务配置放冻结 JSON，不扩展成零散 CLI 参数。

当前 command Schema 位于 ../deploy_eval_v2/schemas/command.schema.json；它是传输内部请求契约，不是 plan Schema。Production plan_ref must match the frozen bundle digest.

## 事件与错误处理

- RUNNING：保存身份与 next_cursor，用 wait-event 继续观察。
- EVENTS：按 task_id+seq 去重；has_more 时继续补页，不能自行跳到最新游标。deployment.ready 还须核对健康/冒烟证据。
- WAIT_TIMEOUT：保持任务运行判断，继续使用安全游标；不重复粘贴服务配置。
- STALE_VERSION：重读事实并重新决策，不盲改版本重发。
- REQUEST_CONFLICT：同幂等键内容不同，停止并核对原请求。
- UNKNOWN_OUTCOME、协议坏 JSON、SSH 断开：先查状态和原游标，不创建新任务或重跑部署。
- RECOVERY_REQUIRED、release.unverified：保留现场，停止复用资源，核验归属与执行证据。

退出码 0 只表示成功提交/查询/等待；操作成功必须看业务事件。设计要求输入/状态错误 2、通信未知 3、确认操作失败 4，当前实现仍需完整集成验证。

普通响应预算 8 KiB、每页最多 10 关键事件；日志默认 8 KiB，显式最多 32 KiB，包含 JSON 转义与元数据。先按 artifact_ref 读取一次所需片段，再按 next_offset 续读；不从 SSH 获取整份服务日志。tail/行号/关键词模式已实现，采用有界扫描和续读游标，参见 artifact-locators.md。
# Task cancellation clarification

`scope=operation` requires the exact active `operation_id`. `scope=task` may omit
that field; this supports cancelling an idle task holding an owned service. If an
operation ID is supplied it must match the active operation. Task cancellation is
durable and prevents later phases; cleanup completes asynchronously and its event
must be checked. Acceptance is not evidence of resource release.
