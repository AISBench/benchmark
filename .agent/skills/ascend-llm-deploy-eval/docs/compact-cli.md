# 简洁任务目录与统一 CLI

新入口为 `scripts/run.py`，在 Linux 协调节点运行 FULL。历史冻结任务继续使用各自入口，不迁移、不覆盖。新入口只接受参考保真模式；任务的节点地址、端口及挂载须先完成现场绑定。命令强校验属于上游审阅职责。

```sh
python3 scripts/run.py --job-spec /input/job_spec.json --bindings /input/bindings.json --run-dir /work/20261105-006 --mode scripts
python3 scripts/run.py --job-spec /input/job_spec.json --bindings /input/bindings.json --run-dir /work/20261105-007 --mode FULL
```

两个示例是不同任务，不应对同一目录连续执行生成。输出目录存在时拒绝覆盖，生成失败不留下半成品。CLI 为前台单协调进程，长任务由调用方使用现有进程管理方式托管；本入口不创建第二个后台协调器或监控。

## 远程运行时发布

本地冻结版本可通过本技能支持工具发布最小运行时，无需由调用方手工上传整个 skill。此操作写远程文件，须已有任务授权；不适用于只读预检。示例参数为占位，按已审阅任务填写：

```sh
python scripts/publish_runtime.py --host root@HOST --work-root /work/user --task-id JOB --input /local/proxy.py --input /local/job_spec.json --input /local/bindings.json --receipt /local/JOB-publication.json
```

`--plan-only` 仅生成本地发布计划，不访问远程。`--input` 可重复，只接收明确指定的单个文件；重名拒绝。发布凭据中 `input_paths` 给出每个文件的远程路径，不自动重写 job_spec/bindings。需要引用这些输入的路径应在审阅阶段明确绑定为 `/work/user/JOB.runtime/inputs/文件名`，有其他节点时按已确认共享关系或技能任务分发规则保证可见，不能因路径相同推定共享。代理及原生配置需按该实际路径挂载；本工具不自动准备依赖、数据或模型。

运行时在 `/work/user/JOB.runtime`，有绑定任务身份和全部文件 SHA256 的 `runtime-manifest.json`；只包括执行所需模块、四份 Schema、依赖声明和显式输入，不包括测试、历史兼容模块或开发归档。运行时自身缺失第三方依赖时上报，不自动安装。

在所选 Linux 主节点调用 `python3 -B /work/user/JOB.runtime/scripts/run.py ... --run-dir /work/user/JOB --mode FULL`，这是原有单协调器执行入口，发布工具不引入第二套生命周期。数据准备可使用同一冻结运行时的 `generate_gsm8k.py`，输出必须在已授权任务归属目录中。用 `-B` 避免生成额外字节码缓存。

发布复用已有原子分发与 Linux flock：相同内容可以核验复用，内容不同拒绝覆盖；相同发布不等于允许重跑已执行任务。运行时和任务目录均保留，以凭据关联归属，不能据任务结束删除未知文件。发布失败或中断保留凭据及可能的上传残留，禁止手工绕过；本地凭据已存在也拒绝覆盖。修复后使用新的任务 ID 和冻结版本。

## 两种执行模式

- `scripts`：验证配置并生成脚本和测试配置，状态为 SCRIPTS_READY。不访问服务器、不创建容器、不启动模型。现场绑定必须完整；生成成功不代表现场预检通过。
- `FULL`：完成同样的生成，分发并核验任务包，创建容器、检查设备占用和端口、执行部署和冒烟、原生 AISBench 测试、有限重试、释放并汇集产物。只清理任务标签和本次执行标识匹配的资源，启动和探测同时核验 ID。case 中断会记录 INTERRUPTED。

FULL 与手动运行使用同一份脚本。脚本、输入和生成器摘要记录于 `.runtime/manifest.json`。新入口不复制整套 Skill 快照；只携带运行时需要的数据检查、结果检查辅助程序。协调器本身使用当前固定部署的代码版本，因此运行中不要更新安装代码。

## 目录

```text
JOB_ID/
  job_spec.json
  bindings.json
  status.json
  scripts/DEPLOYMENT/NODE-container.sh
  scripts/DEPLOYMENT/NODE-dependencies.sh
  scripts/DEPLOYMENT/NODE-start.sh
  scripts/DEPLOYMENT/NODE-INSTANCE.sh
  scripts/DEPLOYMENT/NODE-probe.sh
  scripts/DEPLOYMENT/NODE-stop.sh
  scripts/DEPLOYMENT/smoke.sh
  scripts/tests/SUBTASK.sh
  scripts/tests/SUBTASK.config.py
  logs/coordinator.log
  logs/DEPLOYMENT-run01/NODE-INSTANCE.log
  results/SUBTASK-try01/aisbench.log
  results/SUBTASK-try01/outputs/...
  .runtime/
```

`NODE-INSTANCE.sh` 中 INSTANCE 为 job_spec 的实例 id（包括 proxy），与节点级 container/dependencies/start/probe/stop 脚本并存；start.sh 调用这些实例脚本。

`.runtime` 保存分发索引、文件摘要、任务锁、远端执行声明、容器 ID 与执行标识、运行编号和必要检查程序。没有逐阶段快照。status.json 只有协调器写入，包含 case、资源释放与产物汇集状态。共享目录验证可见且摘要一致后复用；独立目录分发同一包，结果最终收集到协调节点。文件和日志的 NODE 指物理机器，不是 P/D 角色。

手动执行时，在各自物理节点先运行 container.sh，再运行 start.sh；跨节点服务全部启动后执行 smoke.sh，然后在测试节点执行对应 tests/SUBTASK.sh，结束后各节点运行 stop.sh。实例脚本在容器内执行，start.sh 已负责 docker exec。run01/try01 是默认运行编号，重新执行须传新编号；测试目录存在则拒绝覆盖。手动操作不会伪造协调器完成状态。

## 现场绑定

模板仅识别无空格的 `{{name}}` 或 `{{paths.key}}` 等标识符路径（字母/下划线开头，后续为字母、数字、下划线或点）。匹配的绑定不存在时仍报 `Unresolved binding`。其他花括号语法按原文保留，例如 `docker inspect -f '{{ index .Config.Labels "bench.task" }}'` 和 `{{.Id}}`。替换只进行一轮，不将绑定值中的花括号再次解释为模板；shell 场景中的绑定值仍通过 shell quoting 处理。

节点绑定包含 host、work_dir、model_paths、paths、mounts、mount_profiles 和 benchmark 等字段。mount_profiles 显式选择驱动/通信模板，mounts 按 target 覆盖模板或追加；不隐式追加旧基线主机目录。生成器另合并任务目录（可写）与模型目录（默认只读）；同 target 的显式 source 必须与绑定路径一致，任务目录不能只读，模型目录保留显式访问模式。容器设备和基础选项由部署的 runtime.container 规范化，完整规则见 [容器基线](../references/container-baseline.md)。

每个物理节点增加确定的部署网络与端口：

```json
{
  "nodes": {
    "node17": {
      "host": "root@192.0.2.17",
      "work_dir": "/work",
      "deployments": {
        "A": {
          "network": {"ip": "192.0.2.17", "interface": "eth0"},
          "ports": {"p1": {"http": 18000, "kv": 18001}}
        }
      }
    }
  }
}
```

上述仅为字段片段，完整绑定仍需模型路径、挂载和测试数据。所有声明端口都必须绑定，生成器不临时改端口。启动前检查占用，失败不抢占其他任务资源。

## AISBench 接口

`execution.enable_cache_reset_api` 默认 `true`，规范化时为所有非 proxy 模型实例补充 `VLLM_SERVER_DEV_MODE=1`（包括保真模式；若显式值为 0 则报冲突，不覆盖）。这是测试控制配置，不改变模型参数。设为 `false` 仅停止自动注入，保留参考脚本显式配置。此开关不调用清理接口，也不改变 `case.reset_cache=false` 默认值；`reset_cache=true` 在测试前逐模型实例检查空闲并直接清理，确认成功后再测试。

case.mode 默认为 `perf`；`accuracy` 是预留意图，可进行结构校验，但当前生成和执行均明确拒绝，不会回退成 perf。协调器 FULL/scripts 与测试 perf/accuracy 独立。AISBench 精度原生阶段为 all/infer/eval/viz，将来由独立适配器映射，不能直接把 accuracy 传给 AISBench。

默认生成既有 VLLMCustomAPI + CustomDataset 的原生配置。可选 case.aisbench_config 指向 bindings.nodes.<测试节点>.benchmark.configs 中的配置 ID，其值为生成机上可读的原生 AISBench Python 配置路径。配置复制到 scripts/tests，保留原生模型/数据集适配器；原文不修改，模型、端点、并发、最大输出和请求速率在运行前与 job_spec/bindings 核验。配置须采用已安装包的绝对导入，不能依赖未打包的相对文件。配置中的地址、模型和数据路径须由调用者与当前部署对齐。

测试直接调用 `ais_bench CONFIG --mode perf --work-dir ... --num-prompts ... --num-warmups ...`，不加 debug，不修改 AISBench 源码。保留原生输出目录与 worker 日志；退出码、每组请求详情和成功状态均核验。自带配置支持多组性能结果，不要求全部结果压成单张自制表。不同 AISBench 版本的结果布局仍需现场验证，不能声称所有版本与全部插件均已验证。

当前简洁入口支持 after_case=keep_service/stop_container、task_end=stop_container；stop_service 尚未接入，此类输入在生成前明确拒绝。旧公开入口已停用。不能用新入口静默替代不支持的策略。主进程 SIGKILL/主机断电无法执行 finally；任务锁和容器身份保留，需核验后收尾，不能盲目重新提交。

任务目录不自动删除。确认资源释放及归档完整后，方可由外层清理；异常产物保留。

官方接口依据：[AISBench 运行模式](https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/mode.md)、[CLI 参数](https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/cli_args.md)。

## 契约

job_spec.schema 校验任务；bindings.schema 在生成前校验绑定；shared_storage.schema 校验共享声明并接入 FULL 探针；status.schema 在每次状态写入前校验。旧 script_result 协议已删除。原生配置 models/datasets 须有唯一 abbr，结果按预期组合核验。

交互预检使用 review_job.py 采集权重与环境证据供 Agent 判断；确认后发生冲突阻塞，不修改已确认参数。

完整占位示例：[job_spec.compact.json](../examples/job_spec.compact.json) 与 [bindings.compact.json](../examples/bindings.compact.json)。示例为 draft，须现场审阅后解决 unresolved 并确认 ready。


## Explicit stages and parallel branches

Existing 3.0 inputs without `stages` retain serial `sub_tasks` semantics. Keep all sub_task definitions and add:

```json
{"stages":[
  {"id":"single","branches":[{"id":"node17","sub_tasks":["a-t1"]},{"id":"node20","sub_tasks":["b-t1"]}]},
  {"id":"multi","branches":[{"id":"cross-node","sub_tasks":["c-t1","c-t2","d-t1"]}]}
]}
```

Every sub_task must occur exactly once. Stages run in array order; branches run concurrently, with serial cases inside each branch. Concurrent branches must own disjoint physical node sets. Generation checks both node IDs and identical bound hosts, ignoring SSH usernames. Normalize DNS/IP aliases for the same machine to one host before confirmation; different aliases do not prove physical isolation.

One Linux coordinator process uses worker threads for remote scripts; only its main thread writes status.json. A branch may retain the same deployment between cases, but always releases at its end. Each stage waits for all branches, verified resource release, and collection of required artifacts before its successor starts. Any failed case, unverified release or missing artifact blocks later stages.

Failure or interruption signals cooperative cancellation: pending cases, container creation and readiness loops stop progressing. Already running remote calls finish within their configured timeouts before cleanup; unrelated workloads are never stopped to implement cancellation. status.timeline records stage, branch and case events with Unix timestamps. Hard process termination still cannot guarantee finally cleanup.

Migration: phase_id alone does not introduce parallelism. Add explicit stages and retain globally unique sub_task IDs; preserve configured deployment semantics and ports. Re-deployments receive distinct run numbers, and each case keeps separate result paths. Do not replay an existing task directory to recover a failed stage.

AISBench 默认生成配置使用 `retry=1`：已核验的 AISBench 3.1 实现将该字段作为总尝试次数（`range(self.retry)`），因此这表示初次请求一次、工具内部零重试。`execution.max_retries=0` 仍表示协调器不重试 case；两层独立，不把协调器重试数直接写入工具字段。`VLLMCustomAPI` 接收服务根 URL 后追加 `v1/completions`。用户提供的原生配置保留原文，需现场核验其版本重试语义；本次修复不改 AISBench 包。
FULL 增加创建全部容器→全部依赖预检通过→模型启动屏障。scripts生成NODE-dependencies.sh但不运行。模型import torch_npu使用镜像ENV+launch.env；不执行bootstrap。实际测试容器记录AISBench入口所属发行包/版本，检查数据可读与字段；原生未知loader延后原生加载验证。execution.dependency_timeout_seconds默认120秒，容器内监督进程超时终止检查进程组；失败按现有归属收尾。logs/DEPLOYMENT-runNN/NODE-dependencies.log为必需产物，重建容器重新检查。

同一任务的不同部署需要不同代理程序时，在对应 proxy 实例设置 `launch.proxy_program_key`（默认 `proxy_program`，只允许字母/下划线开头的标识符），选择 `bindings.nodes[节点].paths[该键]` 的物理路径。例如 A/B 使用 `reference_proxy`，C/D 使用 `image_proxy`；同一节点同时绑定两个路径，无须切换文件或 wrapper。该字段仅供 proxy 使用，模型实例声明会被拒绝；缺少所选路径时生成阻断。它不改变代理参数、路由或健康门控。

任务锁冲突时给出锁路径和 PID 存活线索。PID 存在不证明仍是原协调器（可能复用）；PID 不存在也不证明远端资源已释放。请核验协调器身份、status、两节点资源归属与日志，保留旧目录，使用新任务编号重新规划；不要直接删除锁重跑旧任务。

自动重试只用于已识别的传输失败（SSH 退出码255）及测试阶段以外的连接/超时；配置错误、远端脚本拒绝、未分类错误和测试阶段失败/超时不重跑。status.attempt_failures 保留阶段、原因、可重试判断与实际是否重试，max_retries 仍是上限，释放失败继续阻止推进。


## 当前协议与兼容边界

- SSH 主机和部署网络目前支持 IPv4/ASCII 主机名，不支持 IPv6 字面量。当前部署生成 HTTP 服务；AISBench 配置生成器可表达 HTTPS，不等于本技能已经实现 TLS 部署、证书和 HTTPS 健康检查。
- 当前阶段依次为本地 Schema/绑定校验、远端容器依赖检查、服务健康检查、测试及证据验证。preflight.py/task_preflight.py 属于退役协议的内部兼容代码，不作为新任务入口；旧端口自动分配及本地 HTTP URL 限制不定义当前接口。
- 端口检查只证明检查时可用，不占有后续监听端口。检查与服务绑定之间存在竞争窗口；冲突仍按启动/健康失败处理，不承诺自动换端口或保证重试成功。
- 分发在同一父目录下的独立临时目录写入、校验后再发布；同名发布通过 Linux flock 串行化。已存在且摘要匹配的目录可复用，内容不同则拒绝覆盖；独立 execution.claim 继续阻止重复协调执行。正常失败清理本次临时目录，强杀可能留下隐藏的 `.JOB.upload-*`，但不会占据最终目录。核验没有活动发布进程后可人工清理对应临时目录；不要删除共享 `.JOB.publish.lock` 文件或他人的任务目录。既有不完整最终目录、不同内容的同名任务，保留现场并改用新 job_id；本地目录保留为诊断证据。
- 只读进度观察：`python scripts/watch_events.py --run-dir /work/JOB`，读取当前 status.timeline，按终端本地时区显示；历史任务兼容 logs/events.jsonl。仅保存读取游标，不累计无界去重集合。关闭观察器不控制任务。
- launch.args 按最终 CLI 传递：True/False 为 `--name`/`--no-name`，dict/list 为单个 JSON 参数。只可用于目标版本支持此语法的参数；保真传参不是版本兼容性认证。示例中的 `attention_config.indexer_kv_dtype` 等是现场版本参数，须按所选镜像的 `vllm serve --help` 核验，不能当作通用默认参数照抄。


## 现行代码导航与历史兼容范围

新任务执行以 **run.py** 为准；这是唯一 FULL/scripts 生命周期入口。现行文件如下（均在 scripts/）：

| 用途 | 文件 |
|---|---|
| 执行与分发 | run.py、bundle_distribution.py、remote_node.py、shared_storage.py |
| 配置与脚本编译 | task_model.py、topology.py、container_baseline.py、mount_profiles.py |
| 现场审阅与授权数据生成 | review_job.py、generate_gsm8k.py |
| 容器依赖及测试证据 | container_dependencies.py、aisbench_cli.py、test_evidence.py、validate_case.py、cache_reset.py、metric_evidence.py、ownership_evidence.py |
| 只读观察 | watch_logs.py、watch_events.py、task_events.py（历史事件流兼容） |

以下文件保留用于历史协议回归或其内部依赖，不是新任务的执行入口：case_runner.py、cluster_runtime.py、deploy.py、discover_tool.py、disk_guard.py、evaluate.py、finalize.py、group_run.py、group_worker.py、node_identity.py、node_prepare.py、node_rpc.py、plan_run.py、preflight.py、run_task.py、service_process.py、stage_runtime.py、submission.py、submit_job.py、task_preflight.py、task_runtime.py。不移动历史模块以免破坏现有回归导入路径；新入口不依赖这些模块。停用的公开命令及 stage_runtime.entry 的拒绝执行由 test_review_closure.py/test_stage_entry.py 验证，stage_runtime 的模块执行共用 entry 的一处停用提示。

采集通过 nonce 文件证明是否共享同一目录，并通过 is_local 显式选择本机复制或 SSH/scp，不以路径字符串相同或命令列表恰好相等作判定。

最终收尾逐项尝试释放、采集、状态写入和解锁，记录 `.runtime/finalization.json` 的原始异常与各步骤结果。状态写入失败也会尝试解锁；解锁/写入错误不得覆盖原任务异常或用户中断。无原始异常但收尾失败时整体失败；若诊断文件本身无法写入，协调器日志保留错误。INTERRUPTED 不表示释放成功，必须另查 release 和 finalization 证据。

代理依赖检查读取生成器写入的 `# @proxy-command` 标记后的命令，允许尾部空行/注释；不把 bootstrap 内的其他 exec 当成代理入口。参数缺失、非法超时或标记/命令异常会给出明确错误。
