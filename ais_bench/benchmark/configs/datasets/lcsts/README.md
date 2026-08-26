# LCSTS
中文 | [English](README_en.md)
## 数据集简介
LCSTS数据集是一个大规模、高质量中文短文本摘要数据集，由哈尔滨工业大学收集，用于新闻摘要生成任务。该数据集主要来源于中国的微博平台，包含了超过200万条真实的中文短文本及其作者给出的简短摘要。此外，研究者还手动标注了其中10666条摘要与对应短文本的相关性。
> 🔗 数据集主页链接[https://huggingface.co/datasets/aligeniewcp22/LCSTS](https://huggingface.co/datasets/aligeniewcp22/LCSTS)

## 数据集部署
- 可以从opencompass提供的汇总数据集链接🔗 [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)将压缩包中`data/LCSTS/`下的文件复制到`LCSTS/`中
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip -d OpenCompassData-core-20240207
mkdir LCSTS/
cp -r OpenCompassData-core-20240207/data/LCSTS/* LCSTS/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree LCSTS/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    LCSTS/
    ├── test.src.txt
    ├── test.tgt.txt
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|lcsts_gen_0_shot_chat|lcsts数据集生成式任务|accuracy|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.lcsts.lcsts_gen_0_shot_chat import lcsts_datasets as datasets`|[lcsts_gen_0_shot_chat.py](lcsts_gen_0_shot_chat.py)|
|lcsts_gen_0_shot_str|lcsts数据集生成式任务|accuracy|0-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.lcsts.lcsts_gen_0_shot_str import lcsts_datasets as datasets`|[lcsts_gen_0_shot_str.py](lcsts_gen_0_shot_str.py)|
