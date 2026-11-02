# CMATH dev

中文小学数学文本精度评测。官方来源：https://github.com/XiaoMi/cmath ，固定 revision `9fa13ef0b2d03f5a18c1ace7001248d3981d65ea`。

下载该版本 `datasets/cmath_dev.jsonl` 到 `ais_bench/datasets/cmath/cmath_dev.jsonl`，预期 600 条，字段 `input/golden`。数据 CC BY 4.0，请保留官方来源及作者署名（Tianwen Wei 等，CMATH，2023）。不随代码提交全量数据。此 dev 在框架映射为 test split，不是官方 test 全集。

```bash
ais_bench --models vllm_api_general_chat --datasets cmath_gen_0_shot_cot_chat_prompt --num-prompts 10
```

运行前设置模型配置的服务地址、模型名及凭据；凭据不要提交。去掉 `--num-prompts 10` 评估全部 600 条。str 任务为 `cmath_gen_0_shot_cot_str`。

0-shot 中文 CoT，无 few-shot。保留官方最后两个候选、分数/百分数转换及数值比较，修复官方 has_exception 调用参数错误。官方整数严格相等，其余采用相对差 < 1%；未额外增加 boxed 优先等规则。官方抽取对负号等格式存在限制，保留以避免静默改变协议。失败或无答案算错且保留分母，与官方排除 ERROR 的行为不同。结果 accuracy 为百分制，details 保存原输出、候选、参考和正确性。

English: Evaluates the official 600-example CMATH dev set, mapped to the framework test split. Uses zero-shot Chinese CoT and upstream candidate extraction/numeric matching; failed responses remain in the denominator. This is not an official test-set or published-score reproduction. Download data from the pinned upstream revision and retain CC BY 4.0 attribution.

## Validation / 验证

- Unit and dataset-reader integration tests: 18 CMATH + 14 GSM8K regression tests, 32 passed.
- Official dev validation: 600 records; 2,400 synthetic extraction/scoring cases matched the pinned upstream helpers after repairing their has_exception call signature. Synthetic cases are not model scores.
- Ubuntu ARM64 full CLI smoke: deepseek-flash, first 10 dev examples, 0-shot CoT, temperature 0, max_tokens 1024, thinking disabled, one attempt per example. 10 successful requests, no failures; accuracy 100% (10/10), independently recomputed from saved predictions and references. This small smoke does not establish full-set accuracy or reproduce a published score.
- Loader supplies an empty train split for DatasetReader compatibility; no evaluation examples are used as few-shot demonstrations.
