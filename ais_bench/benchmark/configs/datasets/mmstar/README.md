# MMStar
中文 | [English](README_en.md)
## 数据集简介
MMStar 是一个“精英级”视觉-语言评测集，共 1500 道英文选择题，全部经过人工复审，确保每题都必须看图才能回答、训练数据泄露风险极低，并均衡覆盖 6 大核心能力（粗粒度感知、细粒度感知、实例推理、逻辑推理、数学、科技）与 18 个细粒度维度，用于严格检验大模型真正的多模态理解力。

> 🔗 数据集主页[https://huggingface.co/datasets/Lin-Chen/MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar)

## 数据集部署
- 数据集下载：modelscope提供的链接🔗 [https://www.modelscope.cn/datasets/evalscope/MMStar/resolve/master/MMStar.tsv](https://www.modelscope.cn/datasets/evalscope/MMStar/resolve/master/MMStar.tsv)。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
mkdir mmstar
cd mmstar
wget https://www.modelscope.cn/datasets/evalscope/MMStar/resolve/master/MMStar.tsv
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree mmstar/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    mmstar
    └── MMStar.tsv
    ```

## 可用数据集任务
### mmstar_gen
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|mmstar_gen|mmstar数据集生成式任务|acc|0-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.mmstar.mmstar_gen import mmstar_datasets as datasets`|[mmstar_gen.py](mmstar_gen.py)|
|mmstar_gen_cot|mmstar数据集思维链生成式任务|acc|0-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.mmstar.mmstar_gen_cot import mmstar_datasets as datasets`|[mmstar_gen_cot.py](mmstar_gen_cot.py)|
