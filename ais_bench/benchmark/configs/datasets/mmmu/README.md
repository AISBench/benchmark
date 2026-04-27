# MMMU
中文 | [English](README_en.md)
## 数据集简介
MMMU是一个面向大学水平的跨学科图文推理评测集，从教学用图（图表、乐谱、化学结构等）获取得到，覆盖艺术、商业、理工、医学、人文、工程等 6 大领域，用于衡量多模态模型在复杂语义与视觉符号上的综合理解与推理能力。

> 🔗 数据集主页[https://modelscope.cn/datasets/AI-ModelScope/MMMU/summary](https://modelscope.cn/datasets/AI-ModelScope/MMMU/summary)

## 数据集部署
- 该数据集实现对齐 evalscope，数据来源为 ModelScope 数据集 `AI-ModelScope/MMMU`，默认评估 `validation` 划分。
- 配置文件 [mmmu_gen.py](mmmu_gen.py) 中默认从 `{工具根路径}/ais_bench/datasets/mmmu` 读取本地 parquet 数据文件。
- 默认 prompt 已在 [mmmu_gen.py](mmmu_gen.py) 中以 `MULT_CHOICE_PROMPT` 和 `OPEN_PROMPT` 暴露，用户可直接修改配置文件来自定义选择题和开放题提示词。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径）。以 linux 上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
git lfs install
git clone https://www.modelscope.cn/datasets/AI-ModelScope/MMMU.git mmmu
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree mmmu/`查看目录结构，若能看到各学科目录或 parquet 数据文件，则说明数据集部署成功。
    ```
    mmmu
    ├── Accounting
    │   └── validation-*.parquet
    ├── Agriculture
    │   └── validation-*.parquet
    └── ...
    ```

## 可用数据集任务
### mmmu_gen
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|mmmu_gen|MMMU 数据集生成式任务：选择题使用 CoT 单选模板，开放题使用 `ANSWER: [ANSWER]` 模板|acc|0-shot|多模态对话格式|[mmmu_gen.py](mmmu_gen.py)|
