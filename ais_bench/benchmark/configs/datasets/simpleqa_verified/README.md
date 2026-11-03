# SimpleQA Verified

Google DeepMind/Google Research 的 1000 条英文事实问答。数据官方来源 https://huggingface.co/datasets/google/simpleqa-verified ，固定 revision `0dc97e0d28d8233463e005cdc4475cc2a13ba2dc`，数据卡标记 MIT。部署官方 `simpleqa_verified.csv` 到 `ais_bench/datasets/simpleqa_verified/`，或在配置中指定 ModelScope 下载后的本地 CSV/JSONL 路径。保留 original_index、problem、answer 字段及来源说明。

官方 eval split 映射到框架 test，train 为空；不使用工具、搜索、检索或 few-shot。待测模型收到原始 problem。不得将 SimpleQA 的原始 4326 题替代 Verified 子集。

## 运行

配置待测模型，并编辑数据集配置中的 `judge_infer_cfg.judge_model`（独立模型名、服务地址和参数）。默认名称 gpt-4.1，localhost:8005 仅是需要替换的服务占位，不会自动提供官方裁判。

```bash
ais_bench --models vllm_api_general_chat --datasets simpleqa_verified_gen_0_shot_chat --num-prompts 10 --num-warmups 0
```

默认 all 流程会执行推理、裁判、计分与汇总。去掉 num-prompts 执行全部 1000 条。不要向仓库提交密钥；使用组织已有的安全服务配置。替换裁判后必须报告口径差异，不声称复现官方分数。

## 裁判与指标

提示词来自 Google 官方数据卡链接的 [Nan Liao starter notebook](https://www.kaggle.com/code/nanliao7/simpleqa-verified-benchmark-starter-code)，版本 9；资源 `simpleqa_verified_prompt.txt` 保留完整模板，运行时仅映射字段名。研究报告：https://arxiv.org/abs/2509.07968 。

严格解析 A/CORRECT、B/INCORRECT、C/NOT_ATTEMPTED。拒绝多标签、歧义和无效响应，单独报告 invalid_judge_count 并停止生成指标；与官方示例无效时默认 C 的行为不同。也修正其宽松正则及 CORRECT 子串可误匹配 INCORRECT 的问题。不把裁判失败等同于模型未作答。传输失败需检查 CLI 子任务及原始预测，不能仅信任主进程退出码。

指标均为 **0–1**：accuracy/correct=A/N，incorrect=B/N，not_attempted=C/N，accuracy_given_attempted=A/(A+B)，f1=2A/(N+A+B)，无尝试时为0。details 保留裁判输出、grade 和 model_pred_uuid。需要用 UUID 连接原始模型预测。

English: Loads the official 1,000-example CSV (or local JSONL), preserves metadata, and reuses AISBench's model → LLM judge → evaluation workflow. Configure the target and judge services separately. The official grader is GPT-4.1; alternative judges are integration tests, not comparable official scores. All reported metrics use a 0–1 scale. Invalid judge outputs withhold metrics instead of silently counting as abstentions.

## Integration validation

Ubuntu ARM64/Python 3.10 full CLI smoke (2026-11-03): first 10 eval records, deepseek-flash as both target and alternative judge, temperature 0, thinking disabled, target max_tokens 1024 / judge 32, one attempt each. 10 target + 10 judge requests succeeded; A/B/C=7/3/0, invalid=0. Accuracy and F1 were 0.70 (0–1 scale). UUID joins and independent metric recomputation matched the CLI summary. This is not GPT-4.1 grading or full-set score reproduction.
