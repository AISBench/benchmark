# SIQA
中文 | [English](README_en.md)
## 数据集简介
SIQA（Social Interaction QA） 是一个用于测试社会常识智能的问答基准。与许多关注物理或分类知识的先前基准不同，SIQA专注于推理人们的行为及其社会影响。例如，给定一个动作如“杰西看了一场音乐会”和一个问题如“杰西为什么这么做？”，人类可以轻松推断出杰西想“看他最喜欢的表演者”或“享受音乐”，而不是“看看里面发生了什么”或“看看是否有效”。

> 🔗 数据集主页链接[https://huggingface.co/datasets/allenai/social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa)

## 数据集部署
- 可以从OpenCompass提供的汇总数据集链接🔗 [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)将压缩包中`data/siqa/`下的文件复制到`siqa/`中
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir siqa/
cp -r OpenCompassData-core-20240207/data/siqa/* siqa/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree siqa/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    siqa/
    ├── dev.jsonl
    ├── dev-labels.lst
    ├── train.jsonl
    ├── train-labels.lst
    ```
## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|siqa_gen_0_shot_chat|siqa数据集生成式任务；`EDAccEvaluator`精度评估方式会通过`Levenshtein距离算法`选取最接近的答案，可能会造成误判，导致精度得分结果偏高。|accuracy|0-shot|对话格式|[siqa_gen_0_shot_chat.py](siqa_gen_0_shot_chat.py)|
|siqa_ppl_0_shot_chat|siqa数据集PPL任务|accuracy|0-shot|对话格式|[siqa_ppl_0_shot_chat.py](siqa_ppl_0_shot_chat.py)|