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

原生tokenid实现会解码token为文本；重新分词可能改变长度和公共前缀边界。原生合成模式下，`input_tokens` 和 `shared_prefix_ratio` 是 RequestSize/PrefixLen 的生成目标，不是重新分词后必须精确相等的验收值。90%表示约90%的生成目标，不额外设定未经用户要求的比例容差或缓存命中率阈值。优先核验原生命令和配置是否正确传递、AISBench成功退出、正式请求成功与数量一致、原始结果及worker日志完整。

`synthetic-evidence.json` 对原始input文本做只读测量，记录实际长度、公共前缀、目标值和明细摘要；偏差仅报告，不据此失败、停止后续case或重建服务。缺少文本、tokenizer无法加载等诊断问题记录UNVERIFIED，不阻断成功请求的验收。不能据UNVERIFIED宣称实测前缀已验证。`validated.json.input_length.policy=native_target` 明确此口径，同时保留target和exact_match。请求失败、错误退出、缺失正式明细/worker日志、请求数或输出上限异常仍失败。单请求无法测量跨请求公共前缀。不自行截断、补齐或改造AISBench数据。


目标前缀比例、实际公共前缀与实际缓存命中率分开报告。缓存metrics窗口仍含原生预热，不证明所有P/DP路由被覆盖。原生随机生成遵循实际AISBench版本行为，当前不承诺跨运行生成完全相同的数据。输出仍是最大长度，允许EOS。

## 外部数据

普通 JSONL/CSV 性能测试可绑定 path；明确指定真实数据时不擅自改用 synthetic。数据路径和字段映射见[原生 CLI 适配](native-cli.md)。
