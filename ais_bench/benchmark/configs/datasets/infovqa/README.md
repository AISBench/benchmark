# InfoVQA
中文 | [English](README_en.md)
## 数据集简介
InfoVQA数据集于 2021 年作为 DocVQA 挑战赛第三项任务的专属数据集推出，核心用于测试模型对信息图表类图像的视觉问答能力。它包含 5485 张从互联网收集的多样化信息图表图像，搭配 30035 条人工标注的问答对。其问题设计极具针对性，不仅要求模型结合文档布局、文本内容、图形元素和数据可视化内容综合推理，还侧重包含需要基础推理和简单算术技能的题目，比如依据图表中的数据进行计数、比较等操作。

> 🔗 数据集主页[https://huggingface.co/datasets/WinKawaks/InfoVQA](https://huggingface.co/datasets/WinKawaks/InfoVQA)

## 数据集部署
- 对该数据集的精度测评对齐OpenCompass的多模态测评工具VLMEvalkit，数据集格式为OpenCompass提供的tsv文件
- 数据集下载：opencompass提供的链接🔗 [https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv](https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv)。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
mkdir InfoVQA
cd InfoVQA
wget https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree InfoVQA/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    InfoVQA
    └── InfoVQA_VAL.tsv
    ```

## 可用数据集任务
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|infovqa_gen|infovqa数据集生成式任务|anls|0-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.infovqa.infovqa_gen import infovqa_datasets as datasets`|[infovqa_gen.py](infovqa_gen.py)|