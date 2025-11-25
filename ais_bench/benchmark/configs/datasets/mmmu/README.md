# MMMU
中文 | [English](README_en.md)
## 数据集简介
MMMU是一个面向大学水平的跨学科图文推理评测集，从教学用图（图表、乐谱、化学结构等）获取得到，覆盖艺术、商业、理工、医学、人文、工程等 6 大领域，用于衡量多模态模型在复杂语义与视觉符号上的综合理解与推理能力。

> 🔗 数据集主页[https://huggingface.co/datasets/MMMU/MMMU](https://huggingface.co/datasets/MMMU/MMMU)

## 数据集部署
- 对该数据集的精度测评对齐OpenCompass的多模态测评工具VLMEvalkit，数据集格式为OpenCompass提供的tsv文件
- 数据集下载：opencompass提供的链接🔗验证集 [https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv](https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv)🔗 测试集[https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv](https://opencompass.openxlab.space/utils/VLMEval/MMMU_TEST.tsv)。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
mkdir mmmu
cd mmmu
wget https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree mmmu/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    mmmu
    └── MMMU_DEV_VAL.tsv
    ```

## 可用数据集任务
### mmmu_gen
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|mmmu_gen|mmmu数据集生成式任务|acc|0-shot|字符串格式|[mmmu_gen.py](mmmu_gen.py)|
