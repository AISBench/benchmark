# AIME2026
中文 | [English](README_en.md)
## 数据集简介
AIME2026 数据集来源于 2026 年的 American Invitational Mathematics Examination（AIME），包含 AIME I 和 AIME II 的正式竞赛题。AIME 是美国高难度高中数学竞赛之一，旨在考察参赛者在代数、几何、数论、组合数学、概率等方向的创造性数学推理与问题解决能力。AIME2026 数据集共收录 30 道题，每道题的答案均为 0 到 999 之间的整数，适合用于评估模型在复杂数学推理、逐步解题和符号计算方面的能力。

> 🔗 数据集主页链接：[https://modelscope.cn/datasets/evalscope/aime26/summary](https://modelscope.cn/datasets/evalscope/aime26/summary)


## 数据集部署
- 可以从 ModelScope 数据集主页 🔗 [https://modelscope.cn/datasets/evalscope/aime26/summary](https://modelscope.cn/datasets/evalscope/aime26/summary) 获取数据集，并整理为 `aime2026.jsonl` 文件。
- 建议部署在`ais_bench/datasets`目录下（数据集任务中设置的默认路径），以 Linux 上部署为例，目标目录结构如下：
```bash
# linux服务器内，处于ais_bench工具目录下
mkdir -p datasets/aime2026
# 将整理后的 aime2026.jsonl 放置到该目录
```
- 在`ais_bench/datasets`目录下执行`tree aime2026/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    aime2026/
    └── aime2026.jsonl
    ```

## 数据格式
数据文件为 JSON Lines 格式，每行对应一道题目，主要字段如下：

|字段|简介|
| --- | --- |
|problem|AIME 数学题题干|
|answer|题目的整数答案|

## 提示模板
默认 0-shot 任务使用如下提示模板，要求模型逐步求解并将最终答案放在 `\boxed{}` 中，便于答案抽取和评估。

```text
Solve the following math problem step by step. Put your answer inside \boxed{}.

{problem}

Remember to put your answer inside \boxed{}.
```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|aime2026_gen|AIME2026 数据集生成式任务|准确率(accuracy)|0-shot|对话格式|aime2026_gen_0_shot_chat_prompt.py|
|aime2026_gen_0_shot_str|AIME2026 数据集生成式任务|准确率(accuracy)|0-shot|字符串格式|aime2026_gen_0_shot_str.py|
