# Xsum
中文 | [English](README_en.md)
## 数据集简介
XSum（Extreme Summarization）数据集是用于评估抽象单文档摘要系统的数据集。其目标是创建一个简短的、一句话的新摘要，回答“这篇文章是关于什么的？”这个问题。该数据集包含226711篇新闻文章，每篇文章都附有一句话摘要。这些文章来自BBC（2010年至2017年），涵盖了广泛的领域，如新闻、政治、体育、天气、商业、技术、科学、健康、家庭、教育、娱乐和艺术。

> 🔗 数据集主页链接[https://huggingface.co/datasets/EdinburghNLP/xsum](https://huggingface.co/datasets/EdinburghNLP/xsum)

## 数据集部署
- 可以从opencompass提供的汇总数据集链接🔗 [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)将压缩包中`data/Xsum/`下的文件复制到`Xsum/`中
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir Xsum/
cp -r OpenCompassData-core-20240207/data/Xsum/* Xsum/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree Xsum/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    Xsum/
    ├── dev.csv
    ├── dev.json
    ├── dev.jsonl
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|Xsum_gen_0_shot_chat|Xsum数据集生成式任务|accuracy|0-shot|对话格式|[Xsum_gen_0_shot_chat.py](Xsum_gen_0_shot_chat.py)|
|Xsum_gen_0_shot_str|Xsum数据集生成式任务|accuracy|0-shot|字符串格式|[Xsum_gen_0_shot_str.py](Xsum_gen_0_shot_str.py)|