# 原生 CLI 适配

[返回快速开始](../README.md)

Skill 只编排任务、生成配置和验证结果。数据构造、预热、命中率测量和正式性能测试复用本仓库的 `ais-bench-prefix-cache` 与 `ais_bench`，不分发工具源码、语料或 PD 代理。

## 镜像与现场准备

在目标执行镜像内，按本仓库[安装文档](../../../../docs/source_zh_cn/get_started/install.md)安装 AISBench 及服务化依赖，并安装原生插件。以下命令在 AISBench 仓库根目录执行：

```sh
python -m pip install -e . --use-pep517
python -m pip install -r requirements/api.txt
python -m pip install -r requirements/extra.txt
python -m pip install -e plugins/prefix_cache
ais_bench --help
ais-bench-prefix-cache --help
```

预检从镜像发现上述命令。不会自动安装依赖或下载语料。绑定示例中的每个执行节点需要填写：

```json
{
  "benchmark": {
    "corpus_path": "/data/corpus.jsonl",
    "corpus_field": "question",
    "block_size": 16
  },
  "paths": {
    "proxy_program": "/opt/deployment/reviewed-pd-proxy.py"
  }
}
```

`block_size` 必须与服务实际配置一致，示例值不是默认推断。语料使用原生插件支持的 JSONL 格式，字段由 `corpus_field` 指定。语料和代理必须是宿主机上存在的绝对路径，编排器为未被同路径挂载覆盖的文件增加只读挂载。

`proxy_program` 仅 PD 部署需要；使用与生成的 prefiller/decoder hosts/ports 参数兼容、已审阅的现场代理。Skill 不提供或下载代理实现。代理的额外运行依赖由执行环境提供。

## 配置与命令

每次 attempt 生成独立的 `scenario.json` 和 AISBench 配置薄模板，然后依次执行：

```sh
ais-bench-prefix-cache prepare --mode text --scenario /work/attempt/scenario.json
ais-bench-prefix-cache run --scenario /work/attempt/scenario.json
```

原生插件调用 AISBench 的 `perf` CLI。输入输出长度、正式请求数、并发、请求速率和 seed 来自 case；配置薄模板仅调用仓库内的配置构建函数。`steady_state` 选择 AISBench 原生 `stable_stage` 统计器，不自动启用会改变请求数量的 `--pressure` 模式。

`prefix_ratio` 映射为原生插件的全局理论 `target_hit_rate`，`prefix_num` 映射为 Prefix Group 数量；这与旧工具“公共前缀长度比例”的算法不完全等价。请求目标、理论可达值、实测值及告警分别保留，不承诺实测值精确等于目标。`warmup=false` 使用原生 cold 模式，`reset_cache=false` 当前明确拒绝。

正式请求发送到部署入口；metrics 与 reset 使用 prefill 服务。多 DP 前缀预热要求入口支持 `X-data-parallel-rank` 路由，并提供带 engine 标签的 DP 指标。PD 代理必须保留此能力；现场不满足时不能宣称支持该评测。详见仓库[Prefix Cache 文档](../../../../docs/source_zh_cn/advanced_tutorials/prefix_cache.md)。

## 证据与边界

每次 attempt 保留原生命令、prepare/benchmark 日志、Scenario、Manifest、Analysis、性能明细与清理记录。原生插件日志负责预热和正式压测细节，Skill 事件记录 prepare/benchmark 等外层阶段。

验证要求原生 analysis 为 complete、AISBench 退出码为 0、请求数和输入输出 token 长度符合配置、DP 指标完整且预热覆盖符合计划。原生告警随 validated.json 保留。超时或失败仍走客户端和服务释放流程。

不再维护自有数据生成器、工具日志解析、仓库文件软链接或跨 attempt 数据集缓存。所有测试产物保存在任务目录，避免写入 AISBench 源码树。离线测试通过不代表真实服务器、PD 路由及模型性能已验证。
