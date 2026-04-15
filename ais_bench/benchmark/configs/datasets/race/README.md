# race
中文 | [English](README_en.md)
## 数据集简介
RACE（Reading Comprehension from Examinations）数据集是一个大规模的机器阅读理解数据集。该数据集由中国12-18岁学生的英语考试题目构成，包含27933篇文章和97867个问题。RACE数据集分为两个子集：RACE-M和RACE-H，分别对应初中和高中的题目难度。RACE-M包含28293个问题，适合初中生水平；RACE-H包含69574个问题，适合高中生水平。每个问题都有四个备选答案，其中一个是正确答案。

> 🔗 数据集主页链接[https://huggingface.co/datasets/allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)

## 数据集部署
- 可以从opencompass提供的汇总数据集链接🔗 [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)将压缩包中`data/race/`下的文件复制到`race/`中
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir race/
cp -r OpenCompassData-core-20240207/data/race/* race/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree race/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    race/
    ├── test/
    ├───── high.jsonl
    ├───── middle.jsonl
    ├── validation/
    ├───── high.jsonl
    ├───── middle.jsonl
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|race_middle_gen_5_shot_chat|race数据集生成式任务|accuracy|5-shot|对话格式|[race_middle_gen_5_shot_chat.py](race_middle_gen_5_shot_chat.py)|
|race_middle_gen_5_shot_cot_chat|race数据集生成式任务|accuracy|5-shot|对话格式|[race_middle_gen_5_shot_cot_chat.py](race_middle_gen_5_shot_cot_chat.py)|
|race_high_gen_5_shot_chat|race数据集生成式任务|accuracy|5-shot|对话格式|[race_high_gen_5_shot_chat.py](race_high_gen_5_shot_chat.py)|
|race_high_gen_5_shot_cot_chat|race数据集生成式任务|accuracy|5-shot|对话格式|[race_high_gen_5_shot_cot_chat.py](race_high_gen_5_shot_cot_chat.py)|
|race_ppl_0_shot_chat|race数据集PPL任务|accuracy|0-shot|对话格式|[race_ppl_0_shot_chat.py](race_ppl_0_shot_chat.py)|