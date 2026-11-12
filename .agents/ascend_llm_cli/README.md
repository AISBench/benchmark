# Ascend LLM CLI

为新版本 Ascend LLM skills 提供 JSON 驱动的部署、评测与资源释放能力。稳定版 `ascend-llm-deploy-eval` 独立保留，不导入、不修改。

## 功能边界

- 只部署：部署环境预检、容器及模型启动、服务健康检查、冒烟测试；成功返回 service.json 并保留服务。
- 只评测：接受 endpoint + model，或已有 service.json；核验测试工具、依赖及服务健康，运行 AISBench 性能测试并输出结构化结果。不需要部署配置，也不取得停止外部服务的权限。
- 部署后评测：把 service.json 交给一个独立评测任务，可使用新的 case 和测试环境，不必重新部署。
- release/cancel：按归属释放本任务资源。停止 runtime 不等于释放服务。
- 精度测试尚未实现，不把健康或冒烟测试视为精度评测。

## 安装

Python >=3.11。控制端可运行在 Windows；执行端需要 Linux、jsonschema，以及对应阶段使用的 Docker/驱动或 AISBench。部署不要求 AISBench；只评测不要求 Docker/NPU 管理权限。

```bash
python -m pip install .
deploy-eval-v2 --version
```

命令仍为 deploy-eval-v2，源码在 src/deploy_eval_v2。技能与 CLI 平级，安装五技能使用 install-skills 的 source/destination JSON；技能负责收集需求和阶段决策，CLI 不猜测用户授权或更改负载。

## JSON 交接

| 阶段 | 输入 | 成功输出 |
| --- | --- | --- |
| 部署 | 部署计划、物理绑定和审阅结果 | service.json，包含端点、归属、健康/冒烟证据引用 |
| 评测 | 服务信息、cases、tester 绑定和审阅结果 | 测试结果 JSON，包含实际负载、性能证据与有效性 |
| 释放 | service_id、task_ref、请求身份及当前版本 | 资源释放结果 JSON |

先使用 prepare-inputs 把业务 JSON 转成待审阅的 plan.json / bindings.json。此步骤仅本地校验，不授权、不连接现场、不执行部署。

只部署输入：

```json
{"stage":"deploy","destination":"prepared-deploy","deployment":{"schema_version":"1.0","task_id":"deploy-01","intent":"deploy","deployment":"按 deploy-plan Schema 填写部署对象","policy":{"idle_timeout_seconds":900,"on_idle":"keep_service","on_failure":"release_owned"},"execution":"按 Schema 填写超时与部署参数","unresolved":[]},"bindings":"按 deploy-bindings Schema 填写节点绑定对象"}
```

上例展示字段组成，字符串占位必须替换为 Schema 指定的对象。部署输入禁止嵌入 evaluation/testers；保留服务策略必须显式为 keep_service。

只评测输入（路径和主机需替换为现场值）：

```json
{"stage":"eval","destination":"prepared-eval","task_id":"eval-01","endpoint":"http://node.example:8000","model":"served-model","tester":{"host":"node.example","work_root":"/work/tasks","python":"/venv/bin/python","aisbench":"/venv/bin/ais_bench","model_path":"/models/tokenizer","datasets":{"synthetic":{"type":"aisbench_synthetic"}}},"cases":[{"case_id":"c1","concurrency":1,"input_tokens":128,"output_tokens":16,"formal_requests":4,"num_warmups":1,"steady_state":false,"reset_cache":false,"dataset_id":"synthetic","timeout_seconds":300}]}
```

也可用 `service` 对象代替 endpoint/model；该对象可直接使用部署输出。评测任务不继承服务生产者的释放权限。独立评测当前不允许清缓存，需使用已审阅的完整计划路径显式授权该操作。

```bash
deploy-eval-v2 prepare-inputs --request task.json
deploy-eval-v2 export-contracts --request export.json
```

export.json 为 `{"destination":"contracts"}`，从当前安装包导出唯一权威 Schema。完整的审阅、打包、冻结、发布、启动及阶段请求字段见 [DESIGN](DESIGN.md)。运行阶段使用 `deploy-eval-v2 --config CONTROL deploy --request FILE`（eval/release 同理）；具体示例见设计文档。

## 输出与日志

stdout 仅返回有界 JSON。长操作先返回 RUNNING、operation_id 和 next_cursor，通过 wait-event 继续等待边界。成功与否以业务结果为准，不只看 CLI 退出码。

每个执行工作根只有一份框架 `task.jsonl`，同时承载持久事件与框架诊断。服务日志、AISBench 控制台及原生产物保留；不新增 preflight/postflight/worker/executor 日志文件。远端节点/测试器的诊断先暂存，阶段响应后有界回收到主任务 task.jsonl，校验摘要并持久化后才确认清空远端暂存。断线或并发追加导致回收不完整时保留暂存，并记录 remote.log.pending；日志回收不重新执行部署或负载。

## 本地验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

本版本不宣称已通过真实模型性能验收。Linux 专用项在不支持 Unix socket 的系统上跳过；本地模拟与原生产物 fixture 不等于模型或 NPU 测试。
