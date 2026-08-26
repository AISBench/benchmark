# DocVQA
中文 | [English](README_en.md)
## 数据集简介
DocVQA是一个面向文档图像视觉问答的经典数据集，于 2021 年推出，旨在推动文档分析与识别领域朝着 “目标驱动” 的方向发展。该数据集包含 1.2 万余张文档图像，对应 5 万个问题，文档图像涵盖信件等多种类型，其中融合了打印体、手写体等不同样式的文本，还包含表格、勾选框、分隔符等视觉元素。与普通视觉问答任务不同，它要求模型不仅能提取文档中的文本内容，还需解读文档布局、字体样式等视觉线索来解答问题。例如回答信件中提及的公司地址这类问题，就需要结合文本与文档结构综合判断。

> 🔗 数据集主页[https://huggingface.co/datasets/lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)

## 数据集部署
- 对该数据集的精度测评对齐OpenCompass的多模态测评工具VLMEvalkit，数据集格式为OpenCompass提供的tsv文件
- 数据集下载：opencompass提供的链接🔗 [https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv](https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv)。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
mkdir DocVQA
cd DocVQA
wget https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree DocVQA/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    DocVQA
    └── DocVQA_VAL.tsv
    ```

## 可用数据集任务
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|docvqa_gen|docvqa数据集生成式任务|anls|0-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.docvqa.docvqa_gen import docvqa_datasets as datasets`|[docvqa_gen.py](docvqa_gen.py)|