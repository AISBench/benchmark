# LongBench v2
中文 | [English](README_en.md)
## 数据集简介
LongBench v2旨在评估LLM进行深度理解和推理的长上下文问题的能力。LongBench v2具有以下特点：
（1）长度：上下文长度从8k到2M不等，大多数长度在128k以下。
（2）难度：难度极高，即使是人类专家使用文档内的搜索工具也无法在短时间内做出正确回答。
（3）覆盖范围：涵盖各种现实场景。
（4）可靠性：所有问题均采用多项选择题形式，以确保评估的可靠性。
LongBench v2包含503道富有挑战性的多项选择题，涵盖六大任务类别：单文档问答、多文档问答、长上下文学习、长对话历史理解、代码仓库理解和长结构化数据理解。
> 🔗 数据集主页链接[https://huggingface.co/datasets/zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2)
## 数据集部署
建议从HuggingFace下载数据集：[https://huggingface.co/datasets/zai-org/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2)
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径）
- 部署完成后，在`{工具根路径}/ais_bench/datasets`目录下执行`tree LongBench-v2/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    LongBench-v2/
    └── data.json
    ```
## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|longbenchv2_gen|longbenchv2|准确率(accuracy)|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.longbenchv2.longbenchv2_gen import LongBenchv2_datasets as datasets`|[longbenchv2_gen.py](longbenchv2_gen.py)|