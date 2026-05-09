# AIME2025
中文 | [English](README_en.md)
## 数据集简介
AIME2025 数据集来源于 2025 年的 American Invitational Mathematics Examination（AIME），包含了一系列中学阶段的高难度数学竞赛题。该考试专为美国高中生设计，旨在筛选进入美国数学奥林匹克（USAMO）的候选人。AIME2025 数据集共收录 30 道正式比赛题，涵盖代数、数论、组合数学、几何等多个方向。每道题都具有单一整数解，题目设计注重逻辑推理和数学建模能力，难度远高于常规中学数学题。适合用于评估模型在复杂数学推理和符号计算方面的能力。

> 🔗 数据集主页链接[https://huggingface.co/datasets/opencompass/AIME2025](https://huggingface.co/datasets/opencompass/AIME2025)


## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip
unzip aime2025.zip
rm aime2025.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree aime2025/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    aime2025/
    └── aime2025.jsonl
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|aime2025_gen|AIME2025|数据集生成式任务|准确率(accuracy)|0-shot|对话格式|aime2025_gen_0_shot_chat_prompt.py|
|aime2025_gen_0_shot_llmjudge|AIME2025|数据集生成式任务|准确率(accuracy)， 裁判模型评价的结果|0-shot|对话格式|aime2025_gen_0_shot_llmjudge.py|