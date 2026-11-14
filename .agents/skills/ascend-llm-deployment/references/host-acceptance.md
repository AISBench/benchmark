# v2 首轮上机测试

本地初版代码与验证已完成；此文列出经明确环境授权后执行的上机验收，不表示实机或性能已验收。
首轮支持：外部服务独立 eval（runtime 在测试机）、或自有服务
deploy → eval → release（测试进程在主节点或显式绑定的远端测试机）。多节点模型部署
已有执行路径，但首次验证应先完成单节点，再增加节点。

参数必须保留用户原意，不得为上机修改。部署重试已按已确认失败、清理核验及共享期限实现；未知结果不重试。远端独立测试机需配置 tester 发布目标，行为见 remote-tester.md。缓存重置已接入，必须使用已授权的冻结逐实例目标，行为见 cache-reset.md。稳态/性能按实际证据判定；缺失时间戳、未知版本格式或路由/预热覆盖不足时返回 UNVERIFIED。测试同时核验执行、证据和资源生命周期。

## 1. 安装与无模型检查

控制端 Python >=3.11，安装本目录构建出的 wheel（自动安装 jsonschema）。
所有远端 runtime/node Python 同样需要 jsonschema；代码发布只发布本包，
不会自动安装第三方依赖。测试端另需计划指定的 AISBench、数据和 tokenizer。
不把 Windows 依赖 wheel 上传到 Linux；按目标系统/架构准备依赖。

```powershell
py -3.11 -m pip wheel --no-deps --wheel-dir dist .
py -3.11 -m pip install dist/deploy_eval_v2-0.1.0a1-py3-none-any.whl
deploy-eval-v2 --version
```

经授权后，在 Linux 的独立测试目录运行无模型集成测试：

Use the public socket-start workflow in [socket-validation.md](../../ascend-llm-router/references/socket-validation.md). The Agent itself must call deploy, then eval upon deployment.ready, then release upon evaluation.completed. No source or unit test reading is needed. The legacy developer unit test is not a substitute for Agent acceptance.

在每个目标节点核验 Python 版本、jsonschema 导入、/etc/machine-id、
任务根目录权限；在 runtime 主节点核验 detached Python process creation 可用及授权。
模型节点另核验 Docker、NPU、驱动、镜像、挂载与模型路径；远端子节点
须能从 runtime 主节点通过非交互 SSH 访问。控制端 SSH 可达不能替代此检查。
正式启动器还会校验 detached Python process 单元及 live socket 身份，不提供后台 & 降级。

## 2. 形成冻结输入

准备 plan.json、bindings.json，完成现场绑定（含 machine_id）。用
draft-review 生成摘要，审阅通过后形成 review.json，再执行 freeze-inputs。
不要直接把测试 fixture 当作已批准生产输入；其中的主机、路径和参数是示例。

每次源码修改先重新 package-runtime，再重新审阅/冻结；runtime_sha256 与
bundle_sha256 必须匹配。保留原始用户参数与授权证据，不手工改冻结目录。

publish.json 的字段：task_ref、runtime_archive、input_directory、
runtime_sha256、bundle_sha256。control.json 的 tasks[TASK] 包含 host、root、
allowed_root、绝对 python 路径；附加节点放在 nodes[SLOT]，同样包含上述字段。

```sh
deploy-eval-v2 --config control.json check-ready --request publish.json
```

此命令只读本地内容，验证冻结输入、runtime 包和部署目标，并报告不支持的
配置。只有 READY_FOR_HOST_PREFLIGHT 才进入现场检查；remote_checks=NOT_RUN
明确表示没有验证连通性、系统服务、镜像或 NPU。它不会启动任何远端操作。

## 3. 发布、启动与阶段闭环

```sh
deploy-eval-v2 --config control.json publish-task --request publish.json
deploy-eval-v2 --config control.json start-task --request start.json
deploy-eval-v2 --config control.json get-status --task-ref TASK
deploy-eval-v2 --config control.json deploy --request deploy.json
deploy-eval-v2 --config control.json wait-event --task-ref TASK --after-seq CURSOR --wait-ms 30000
```

start.json 含 task_ref、两项摘要和一次生成并保存的 64 位十六进制 nonce。
响应丢失后保留 nonce；不要用新 nonce 尝试绕过原身份。
阶段请求包含 task_ref、request_id、expected_version、wait_ms；deploy 另含
plan_ref（bundle 摘要），eval 含 service_id/case_id，release 含 service_id。
每次后继操作从 get-status 读取当前版本，禁止对旧事件盲发后继操作。

deployment.ready 后核对服务及健康/冒烟证据，再 eval；收到
evaluation.completed 后分别检查 execution_status、evidence_status 和
performance_validity，最后按批准策略 release。外部服务不调用 deploy/release。
wait-event 的 after-seq 只使用实际交付的 next_cursor。

## 4. 必须保留的验收证据

| 场景 | 要求 |
| --- | --- |
| 发布/启动 | 摘要匹配；detached Python process 与 socket 身份匹配；同 nonce 不重复启动 |
| 短阶段/长阶段 | 返回边界，或 RUNNING 后按游标得到边界 |
| 观察中断 | 结束 client/SSH 后 runtime 仍在；重连无重复模型容器 |
| 幂等 | 同 request_id 同内容返回原操作；不同内容拒绝 |
| 等待超时 | WAIT_TIMEOUT 不取消执行、不重复历史 |
| 部分失败/取消 | 客户端停止；逐资源清理证据可核对；不以取消接纳代表清理完成 |
| 释放 | 仅完整容器 ID 与归属一致时释放；资源占用检查通过后才解锁 |
| 崩溃/不明结果 | 进入 RECOVERY_REQUIRED，保留 claims，禁止盲重跑 |
| 输出 | 普通响应 <=8192 字节；日志按引用读取，原日志保留 |

保留完整任务目录、请求文件、响应及现场版本；不要只保存最终 PASS。
恢复不明资源时按初版契约进入 RECOVERY_REQUIRED，保留证据并人工现场核验，不盲目重跑。自动跨节点修复不属于初版要求。

## 五技能安装

共享 CLI 安装后，使用 `install-skills --request skills-install.json`。请求为
`{"source":"D:/autobenchmark/skills","destination":"目标 Agent 的技能目录"}`。
安装器校验五个技能的身份和 CLI/协议声明；已存在且不同的内容返回
REQUEST_CONFLICT，不覆盖个人草案。先核对并选择新目录进行验收。
本次开发没有安装到个人技能目录。

NPU 清理核验需要 fuser 可用且有设备检查权限；缺失工具、设备错误、
端口占用或节点不可达均保留资源声明，不终止其他任务。
