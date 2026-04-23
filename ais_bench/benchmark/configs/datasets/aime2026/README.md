# AIME2026

## 数据集简介
AIME2026 数据集来源于 2026 年的 American Invitational Mathematics Examination (AIME)，包含了一系列中学阶段的高难度数学竞赛题。每道题都具有单一整数解，题目设计注重逻辑推理和数学建模能力。

## 数据集部署
- 数据来源: HuggingFace `math-ai/aime26`
- 下载后转换为 JSONL 格式，确保包含 `problem` 和 `answer` 字段
- 建议部署在 `{工具根路径}/ais_bench/datasets/aime2026/` 目录下

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|aime2026_gen|AIME2026|数据集生成式任务|准确率(accuracy)|0-shot|对话格式|aime2026_gen_0_shot_chat_prompt.py|
|aime2026_gen_0_shot_str|AIME2026|数据集生成式任务|准确率(accuracy)|0-shot|字符串格式|aime2026_gen_0_shot_str.py|