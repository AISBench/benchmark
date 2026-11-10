# AISBench 原生前缀测试数据

新前缀测试统一使用目标镜像内 AISBench 的 `SyntheticDataset`（`Type=tokenid`）。skill 只生成原生配置、核对能力和收集结果，不调用外部 `aisbench_auto_tools_prefix`，不要求 GSM8K 路径、生成器目录或 generation_manifest。用户只需提供输入/输出长度、并发、请求数和前缀目标比例等测试需求。

## 配置

在测试节点的 bindings 中选择原生数据源：

```json
{"benchmark":{"datasets":{"default":{"type":"aisbench_synthetic"}}}}
```

case 保留 `dataset_id=default`、`input_tokens`、`output_tokens`、`formal_requests`、`concurrency`，并用 `shared_prefix_ratio` 表达0到1的前缀目标。不传该比例时原生前缀长度为0。skill 映射：

- `RequestCount = formal_requests`；预热仍由原生 CLI 的 `num_warmups` 控制。
- `RequestSize = input_tokens`。
- `PrefixLen = floor(input_tokens × shared_prefix_ratio)`。
- `TrustRemoteCode = false`；tokenizer路径使用本部署已绑定的模型目录。

例如输入65536、前缀目标90%映射为PrefixLen=58982。该取整值应在审阅中展示，不引入额外前缀预热或修改请求数量。原生绑定禁止同时携带外部path或generation_manifest，不静默替换用户指定的真实数据。

## 能力与证据

scripts模式只生成配置，不访问服务器或生成样本。容器依赖门控检查实际AISBench发行包、SyntheticDataset及其原生配置校验和可检查的PrefixLen实现，保存模块路径和源码摘要；这不代表样本生成或服务执行已通过。不支持时上报，不安装插件、不回退外部生成器。样本由原生AISBench加载器在正式CLI流程中生成，skill不再生成第二套样本。

本次核对的原生tokenid实现会解码token为文本；重新分词可能改变长度和公共前缀边界。`input_tokens`仍是实际请求长度的严格验收值，长度不符应失败并保留证据，不能偷偷截断、补齐或降低目标。`synthetic-evidence.json`使用原生正式请求明细的input文本进行只读重新分词，保存实际长度、共同前缀、目标长度及明细摘要；缺少文本或无法核验时记录UNVERIFIED并失败。单请求无法测量跨请求公共前缀。

目标前缀比例、实际公共前缀与实际缓存命中率分开报告。缓存metrics窗口仍含原生预热，不证明所有P/DP路由被覆盖。原生随机生成遵循实际AISBench版本行为，当前不承诺跨运行生成完全相同的数据。输出仍是最大长度，允许EOS。

## 外部数据与历史兼容

普通外部JSONL/CSV性能测试仍可绑定path；明确指定真实数据时不擅自改用synthetic。旧的派生数据与generation_manifest校验保留以读取既有证据，但不是新前缀测试的生成路径。

下文为旧GSM8K适配流程记录，仅用于解释历史结果。`scripts/generate_gsm8k.py`保留作历史回归，不再列入新任务最小运行时，也不应向新任务用户索取上游生成工具目录。

### 历史：授权的 GSM8K 派生输入

旧版本在用户明确要求并授权生成时使用 `scripts/generate_gsm8k.py`；以下描述仅供历史结果追溯，新任务不采用此流程。原始JSONL只读，逐行检查question和answer。上游工具由现场提供，固定generate_dataset.py和data_picker.py的SHA256，不随技能携带数据。所有派生文件保存在仓库外任务目录。

脚本必填参数：`--source --upstream --output --tokenizer --generator-sha256 --picker-sha256 --tokens --count --seed --prefix-ratio`。默认AutoTokenizer；经服务实现核验后可选`--tokenizer-backend tokenizers`。此选项只处理原始文本，服务聊天模板开销仍须现场验证。

适配保留上游GSM8K重复/截断与前缀拼接逻辑，将源文件和挑选记录定位到指定路径，先设置随机种子，关闭进度条监控线程。拼接后重新编码，有限次修正长度；无法精确满足则失败。保存修正记录、实际公共前缀长度及逐行长度。0%表示未人为添加公共前缀，不承诺任意两行完全无重复。20%为生成目标，不等于缓存命中率。

generation.json记录源、上游代码、适配器、tokenizer文件摘要、依赖版本、参数、种子、挑选记录和派生摘要。同输出目录仅在身份与文件摘要一致且长度已验证时复用；不完整目录保留证据，另选新目录。不在每个attempt重新生成。

将派生路径放入`bindings.nodes.<node>.benchmark.datasets.<id>.path`，同项增加绝对路径`generation_manifest`。case指定`dataset_id`、`input_tokens`及`shared_prefix_ratio`。运行前核验内容摘要、长度、请求数量与前缀目标。原生预热仍为1次，不执行上游`prefix_test`；其额外请求及dp重复次数不能证明多P/DP覆盖。

标准配置生成阶段即检查：显式声明 `shared_prefix_ratio`（包括 0）必须绑定绝对 `generation_manifest` 路径，缺失时生成失败且不保留半成品、不创建容器。scripts 模式仍不读取远端数据文件；路径存在性、摘要、实际 token 长度及目标比例由实际执行环境继续核验，不能以路径已填写代替数据证据。

每个attempt保留各实例metrics前后原文和时间，prefix-evidence.json按命中增量之和除以查询增量之和计算；计数重置、缺失或标签变化标记无法验证。采样窗口包含原生预热，不能称为纯正式请求命中率；预热路由覆盖另需证据。

`on_failure=skip_higher_concurrency`在失败且完成资源协调后暂停同部署、同输入长度的更高并发，其他长度可继续。未释放资源仍阻塞后续。所有结果与释放结论分别报告。

前缀计数覆盖 prefill 和 mixed 实例，先逐实例/标签求增量再加权汇总；decode 不参与该口径。缺失、重置或标签变化继续标记 UNVERIFIED。
