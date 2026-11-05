# 简洁任务目录与统一 CLI

新入口为 `scripts/run.py`，在 Linux 协调节点运行 FULL。历史冻结任务继续使用各自入口，不迁移、不覆盖。新入口只接受参考保真模式；任务的节点地址、端口及挂载须先完成现场绑定。命令强校验属于上游审阅职责。

```sh
python3 scripts/run.py --job-spec /input/job_spec.json --bindings /input/bindings.json --run-dir /work/20261105-006 --mode scripts
python3 scripts/run.py --job-spec /input/job_spec.json --bindings /input/bindings.json --run-dir /work/20261105-007 --mode FULL
```

两个示例是不同任务，不应对同一目录连续执行生成。输出目录非空时拒绝覆盖。CLI 为前台单协调进程，长任务由调用方使用现有进程管理方式托管；本入口不创建第二个后台协调器或监控。

## 两种模式

- `scripts`：验证配置并生成脚本和测试配置，状态为 SCRIPTS_READY。不访问服务器、不创建容器、不启动模型。现场绑定必须完整；生成成功不代表现场预检通过。
- `FULL`：完成同样的生成，分发并核验任务包，创建容器、检查设备占用和端口、执行部署和冒烟、原生 AISBench 测试、有限重试、释放并汇集产物。仅对归属标签及实际容器 ID 匹配的资源清理。case 中断会记录 INTERRUPTED。

FULL 与手动运行使用同一份脚本。脚本、输入和生成器摘要记录于 `.runtime/manifest.json`。新入口不复制整套 Skill 快照；只携带运行时需要的数据检查、结果检查辅助程序。协调器本身使用当前固定部署的代码版本，因此运行中不要更新安装代码。

## 目录

```text
JOB_ID/
  job_spec.json
  bindings.json
  status.json
  scripts/DEPLOYMENT/NODE-container.sh
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

`.runtime` 保存分发索引、文件摘要、任务锁、容器 ID、当前部署运行编号和三个必要检查程序。没有逐阶段快照。status.json 只有协调器写入，包含 case、资源释放与产物汇集状态。共享目录验证可见且摘要一致后复用；独立目录分发同一包，结果最终收集到协调节点。文件和日志的 NODE 指物理机器，不是 P/D 角色。

手动执行时，在各自物理节点先运行 container.sh，再运行 start.sh；跨节点服务全部启动后执行 smoke.sh，然后在测试节点执行对应 tests/SUBTASK.sh，结束后各节点运行 stop.sh。实例脚本在容器内执行，start.sh 已负责 docker exec。run01/try01 是默认运行编号，重新执行须传新编号；测试目录存在则拒绝覆盖。手动操作不会伪造协调器完成状态。

## 现场绑定

沿用 nodes、host、work_dir、model_paths、paths、mounts 和 benchmark 字段。新入口显式使用 bindings.mounts，不再隐式追加主机目录。容器设备和基础选项由 job_spec.runtime.container 规范化。额外挂载任务目录和模型目录。

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

`execution.enable_cache_reset_api` 默认 `true`，规范化时为所有非 proxy 模型实例补充 `VLLM_SERVER_DEV_MODE=1`（包括保真模式；若显式值为 0 则替换为 1）。这是测试控制配置，不改变模型参数。设为 `false` 仅停止自动注入，保留参考脚本显式配置。此开关不调用清理接口，也不改变 `case.reset_cache=false` 默认值；简洁入口目前仍拒绝 `reset_cache=true`。

case.mode 默认为 `perf`；`accuracy` 是预留意图，可进行结构校验，但当前生成和执行均明确拒绝，不会回退成 perf。协调器 FULL/scripts 与测试 perf/accuracy 独立。AISBench 精度原生阶段为 all/infer/eval/viz，将来由独立适配器映射，不能直接把 accuracy 传给 AISBench。

默认生成既有 VLLMCustomAPI + CustomDataset 的原生配置。可选 case.aisbench_config 指向 bindings.nodes.<测试节点>.benchmark.configs 中的配置 ID，其值为生成机上可读的原生 AISBench Python 配置路径。配置复制到 scripts/tests，保留原生模型/数据集适配器；并发、最大输出和请求速率由 job_spec 覆盖。配置须采用已安装包的绝对导入，不能依赖未打包的相对文件。配置中的地址、模型和数据路径须由调用者与当前部署对齐。

测试直接调用 `ais_bench CONFIG --mode perf --work-dir ... --num-prompts ... --num-warmups ...`，不加 debug，不修改 AISBench 源码。保留原生输出目录与 worker 日志；退出码、每组请求详情和成功状态均核验。自带配置支持多组性能结果，不要求全部结果压成单张自制表。不同 AISBench 版本的结果布局仍需现场验证，不能声称所有版本与全部插件均已验证。

当前简洁入口支持 after_case=keep_service/stop_container、task_end=stop_container；显式清缓存与 stop_service 尚未接入，此类输入在生成前明确拒绝。原入口的这些能力仍保留。不能用新入口静默替代不支持的策略。主进程 SIGKILL/主机断电无法执行 finally；任务锁和容器身份保留，需核验后收尾，不能盲目重新提交。

任务目录不自动删除。确认资源释放及归档完整后，方可由外层清理；异常产物保留。

官方接口依据：[AISBench 运行模式](https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/mode.md)、[CLI 参数](https://gitee.com/aisbench/benchmark/blob/master/doc/users_guide/cli_args.md)。
