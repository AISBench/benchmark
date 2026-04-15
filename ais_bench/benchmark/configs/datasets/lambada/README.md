# LAMBADA
中文 | [English](README_en.md)
## 数据集简介
LAMBADA（Language Modeling Broadened to Account for Discourse Aspects）数据集是一种开放式填空任务，旨在评估计算模型对文本理解的能力。该数据集包含约10000个从BooksCorpus中提取的段落，每个段落的最后一句话缺少一个目标词，要求模型预测这个缺失的词。
> 🔗 数据集主页[https://huggingface.co/datasets/cimec/lambada](https://huggingface.co/datasets/cimec/lambada)

## 数据集部署
- 可以从opencompass提供的汇总数据集链接🔗 [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)将压缩包中`data/lambada/`下的文件复制到`lambada/`中
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir lambada/
cp -r OpenCompassData-core-20240207/data/lambada/* lambada/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree lambada/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    lambada/
    ├── test.jsonl
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|lambada_gen_0_shot_chat|lambada数据集生成式任务|accuracy|0-shot|对话格式|[lambada_gen_0_shot_chat.py](lambada_gen_0_shot_chat.py)|
|lambada_gen_0_shot_str|lambada数据集生成式任务|accuracy|0-shot|字符串格式|[lambada_gen_0_shot_str.py](lambada_gen_0_shot_str.py)|
