# WinoGrande
中文 | [English](README_en.md)
## 数据集简介
WinoGrande是一个包含44,000道题目的新型数据集，其设计灵感源自Winograd Schema Challenge（Levesque、Davis和Morgenstern，2011年），但通过调整规模并增强对数据集特定偏见的鲁棒性进行了改进。该任务采用二选一的填空形式，目标是为给定句子选择符合常识推理的正确选项。

> 🔗 数据集主页[https://huggingface.co/datasets/allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande)

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/winogrande.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/winogrande.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/winogrande.zip
unzip winogrande.zip
rm winogrande.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree winogrande/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    winogrande
    ├── dev.jsonl
    ├── dev-labels.lst
    ├── eval.py
    ├── README.md
    ├── sample-submission-labels.lst
    ├── test.jsonl
    ├── train_debiased.jsonl
    ├── train_debiased-labels.lst
    ├── train_l.jsonl
    ├── train_l-labels.lst
    ├── train_m.jsonl
    ├── train_m-labels.lst
    ├── train_s.jsonl
    ├── train_s-labels.lst
    ├── train_xl.jsonl
    ├── train_xl-labels.lst
    ├── train_xs.jsonl
    └── train_xs-labels.lst
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|winogrande_gen_0_shot_chat_prompt|winogrande数据集生成式任务|accuracy|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.winogrande.winogrande_gen_0_shot_chat_prompt import winogrande_datasets as datasets`|[winogrande_gen_0_shot_chat_prompt.py](winogrande_gen_0_shot_chat_prompt.py)|
|winogrande_gen_5_shot_chat_prompt|piqa数据集生成式任务|accuracy|5-shot|对话格式|`from ais_bench.benchmark.configs.datasets.winogrande.winogrande_gen_5_shot_chat_prompt import winogrande_datasets as datasets`|[winogrande_gen_5_shot_chat_prompt.py](winogrande_gen_5_shot_chat_prompt.py)|
