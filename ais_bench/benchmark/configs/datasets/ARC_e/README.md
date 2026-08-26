# ARC Easy Set
中文 | [English](README_en.md)
## 数据集简介
ARC是一个包含7,787道真实小学阶段科学选择题的新数据集，旨在推动高级问答技术的研究。该数据集分为挑战集（Challenge Set）和简单集（Easy Set），其中挑战集仅包含基于检索算法和词语共现算法均回答错误的难题。本文涉及的是Easy Set。

> 🔗 数据集主页链接[https://huggingface.co/datasets/allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)

## 数据集部署
- 可以从opencompass提供的汇总数据集链接🔗 [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)将压缩包中`data/ARC/`下的文件复制到`ARC/`中
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir ARC/
cp -r OpenCompassData-core-20240207/data/ARC/* ARC/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree ARC/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    ARC/
    └── ARC-e
        ├── ARC-Easy-Dev.jsonl
        └── ARC-Easy-Test.jsonl
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|ARC_e_gen_0_shot_chat_prompt|ARC Easy Set数据集生成式任务|accuracy|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.ARC_e.ARC_e_gen_0_shot_chat_prompt import ARC_e_datasets as datasets`|[ARC_e_gen_0_shot_chat_prompt.py](ARC_e_gen_0_shot_chat_prompt.py)|
|ARC_e_gen_25_shot_chat_prompt|ARC Easy Set数据集生成式任务|accuracy|25-shot|对话格式|`from ais_bench.benchmark.configs.datasets.ARC_e.ARC_e_gen_25_shot_chat_prompt import ARC_e_datasets as datasets`|[ARC_e_gen_25_shot_chat_prompt.py](ARC_e_gen_25_shot_chat_prompt.py)|
|ARC_e_ppl_0_shot_str|ARC Easy Set数据集PPL任务|accuracy|0-shot|字符串模式|`from ais_bench.benchmark.configs.datasets.ARC_e.ARC_e_ppl_0_shot_str import ARC_e_datasets as datasets`|[ARC_e_ppl_0_shot_str.py](ARC_e_ppl_0_shot_str.py)|