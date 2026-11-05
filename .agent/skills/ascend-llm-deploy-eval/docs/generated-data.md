# 授权的 GSM8K 派生输入

仅在用户明确要求并授权生成时使用 `scripts/generate_gsm8k.py`。原始JSONL只读，逐行检查question和answer。上游工具由现场提供，固定generate_dataset.py和data_picker.py的SHA256，不随技能携带数据。所有派生文件保存在仓库外任务目录。

脚本必填参数：`--source --upstream --output --tokenizer --generator-sha256 --picker-sha256 --tokens --count --seed --prefix-ratio`。默认AutoTokenizer；经服务实现核验后可选`--tokenizer-backend tokenizers`。此选项只处理原始文本，服务聊天模板开销仍须现场验证。

适配保留上游GSM8K重复/截断与前缀拼接逻辑，将源文件和挑选记录定位到指定路径，先设置随机种子，关闭进度条监控线程。拼接后重新编码，有限次修正长度；无法精确满足则失败。保存修正记录、实际公共前缀长度及逐行长度。0%表示未人为添加公共前缀，不承诺任意两行完全无重复。20%为生成目标，不等于缓存命中率。

generation.json记录源、上游代码、适配器、tokenizer文件摘要、依赖版本、参数、种子、挑选记录和派生摘要。同输出目录仅在身份与文件摘要一致且长度已验证时复用；不完整目录保留证据，另选新目录。不在每个attempt重新生成。

将派生路径放入`bindings.nodes.<node>.benchmark.datasets.<id>.path`，同项增加绝对路径`generation_manifest`。case指定`dataset_id`、`input_tokens`及`shared_prefix_ratio`。运行前核验内容摘要、长度、请求数量与前缀目标。原生预热仍为1次，不执行上游`prefix_test`；其额外请求及dp重复次数不能证明多P/DP覆盖。

每个attempt保留各实例metrics前后原文和时间，prefix-evidence.json按命中增量之和除以查询增量之和计算；计数重置、缺失或标签变化标记无法验证。采样窗口包含原生预热，不能称为纯正式请求命中率；预热路由覆盖另需证据。

`on_failure=skip_higher_concurrency`在失败且完成资源协调后暂停同部署、同输入长度的更高并发，其他长度可继续。未释放资源仍阻塞后续。所有结果与释放结论分别报告。
