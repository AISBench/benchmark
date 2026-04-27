# MathVision
中文 | [English](README_en.md)

## 数据集简介
MathVision（MATH-V）是一个面向多模态数学推理的评测数据集，包含 3,040 道来自真实数学竞赛的高质量视觉数学题。题目通常包含几何图形、函数图像、统计图表等视觉上下文，用于评估模型在图文联合理解、数学推理和答案抽取上的能力。

> 🔗 数据集主页：[https://modelscope.cn/datasets/evalscope/MathVision/summary](https://modelscope.cn/datasets/evalscope/MathVision/summary)

## 数据集部署
- 本任务参考 evalscope 的 MathVision 实现，默认数据集来源为 evalscope 提供的 ModelScope 数据集：`evalscope/MathVision`。
- `MathVisionDataset` 支持通过 `datasets.load_dataset('evalscope/MathVision', split='test')` 直接加载，也支持将数据集下载到本地后在配置文件中指定本地路径。
- 当前配置文件 [mathvision_gen.py](mathvision_gen.py) 中的 `path` 可按实际部署环境调整：
```python
# 使用 evalscope 提供的数据集连接
path = 'evalscope/MathVision'

# 或使用本地数据目录
path = '/path/to/mathvision'
```
- 如果采用本地部署，建议将数据集放置在 `{工具根路径}/ais_bench/datasets/mathvision` 目录下。数据文件可为 HuggingFace datasets 可识别的目录，也可包含 `test.parquet`、`test-*.parquet`、`test.jsonl`、`test.json` 等文件。
- 数据加载过程中会将样本中的图像字段解析为本地图片文件，并保存到数据目录下的 `MathVision_images` 子目录，推理时以 `file://{image}` 的形式传入多模态模型服务。

示例目录结构如下：
```text
mathvision
├── data
│   └── test-00000-of-00001.parquet
└── MathVision_images
    ├── 0.png
    ├── 1.png
    └── ...
```

## 可用数据集任务
| 任务名称 | 简介 | 评估指标 | few-shot | prompt 格式 | 对应源码配置文件路径 |
| --- | --- | --- | --- | --- | --- |
| mathvision_gen | MathVision 数据集生成式多模态数学推理任务，支持选择题和自由作答题；选择题要求最后一行输出 `ANSWER: [LETTER]`，自由作答题要求最终答案放在 `\boxed{}` 中 | Accuracy | 0-shot | 多模态对话格式（文本 + 图片） | [mathvision_gen.py](mathvision_gen.py) |
