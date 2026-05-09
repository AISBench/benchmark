# RefCOCO

中文 | [English](README_en.md)

## 数据集简介

RefCOCO 是基于 COCO 图像构建的 referring expression grounding 数据集，任务目标是根据自然语言描述在图像中定位对应目标，并输出目标框坐标。AISBench 当前提供 RefCOCO 的 `val`、`test`、`testA`、`testB` 四个 split 测评任务，分别对应约 8.81k、5k、1.98k、1.81k 条原始样本。

在 AISBench 的实现中，RefCOCO 数据以 parquet shard 形式加载，图像内容直接保存在 parquet 中。加载器会将每条样本中的 `answer` 列表展开为多条 benchmark 样本，并支持两种图像传输模式：默认文件路径模式会在运行时把图像缓存到 `data/temp_save_images/<split>/` 下，base64 模式则直接把图像内容嵌入多模态请求。评估阶段使用 `BBoxIoUEvaluator` 按交并比（IoU）$\ge 0.5$ 计算 `Accuracy@0.5`。当前配置中的 `coord_scale=1000.0` 对应一类常见的 `0-1000` 归一化目标框表达方式，这属于模型相关约定，而不是所有模型都必须遵循的统一格式。

- 任务类型：MMU 2D grounding（多模态 visual grounding / referring expression comprehension）
- 数据格式：`data/` 目录下按 split 命名的 parquet 分片文件，文件名模式为 `<split>-*.parquet`
- 标注字段：`question_id`、`image`、`question`、`answer`、`bbox`、`file_name` 等

> 🔗 数据集主页 [https://huggingface.co/datasets/lmms-lab/RefCOCO](https://huggingface.co/datasets/lmms-lab/RefCOCO)

## 数据集部署

- 数据源为 HuggingFace 数据集仓库 `lmms-lab/RefCOCO`。
- 配置文件默认读取路径为 `{tool_root_path}/ais_bench/datasets/RefCOCO/data/`，建议将整个仓库下载到 `{tool_root_path}/ais_bench/datasets/RefCOCO/` 下。
- 以 Linux 环境为例，可执行以下命令完成下载：

```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
mkdir -p RefCOCO
huggingface-cli download lmms-lab/RefCOCO --repo-type dataset --local-dir RefCOCO
```

- 在 `{tool_root_path}/ais_bench/datasets` 目录下执行 `tree RefCOCO/` 查看目录结构。若目录结构如下所示，则说明数据集部署成功：

```text
RefCOCO/
├── .gitattributes
├── README.md
└── data/
    ├── test-00000-of-00002.parquet
    ├── test-00001-of-00002.parquet
    ├── testA-00000-of-00001.parquet
    ├── testB-00000-of-00001.parquet
    ├── val-00000-of-00004.parquet
    ├── val-00001-of-00004.parquet
    ├── val-00002-of-00004.parquet
    └── val-00003-of-00004.parquet
```

- 使用默认文件路径配置时，首次运行测评会自动在 `data/temp_save_images/<split>/` 下生成 JPEG 缓存图片，无需手动创建该目录。
- 使用文件路径图像模式时，缓存图片存储于 `{tool_root_path}/ais_bench/datasets/RefCOCO/data/temp_save_images/`。需确保推理服务能够访问该路径下的本地文件。以 vLLM 为例，启动服务时需添加参数 `--allowed-local-media-path {tool_root_path}/ais_bench/datasets/RefCOCO/data`。

## 可用数据集任务

| Task Name          | Introduction                                                                                                                                                                                                    | Evaluation Metric | Few-Shot | Prompt Format                      | Config File                                    |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | -------- | ---------------------------------- | ---------------------------------------------- |
| refcoco_gen        | RefCOCO 生成式定位任务配置，使用文件路径图像输入（`file://{image}`），导出 `RefCOCO_val`、`RefCOCO_test`、`RefCOCO_testA`、`RefCOCO_testB` 四个 split 任务                                                      | Accuracy@0.5      | 0-shot   | 多模态对话格式（MMPromptTemplate） | [refcoco_gen.py](refcoco_gen.py)               |
| refcoco_gen_base64 | RefCOCO 生成式定位任务配置，使用 base64 data URL 图像输入（`data:image/jpeg;base64,{image}`），导出 `RefCOCO_base64_val`、`RefCOCO_base64_test`、`RefCOCO_base64_testA`、`RefCOCO_base64_testB` 四个 split 任务 | Accuracy@0.5      | 0-shot   | 多模态对话格式（MMPromptTemplate） | [refcoco_gen_base64.py](refcoco_gen_base64.py) |

## 数据集分类

通过 `--datasets` 参数可直接选择以下数据集配置文件名：

- `refcoco_gen`：文件路径图像输入配置，对应配置文件 [refcoco_gen.py](refcoco_gen.py)。
- `refcoco_gen_base64`：base64 图像输入配置，对应配置文件 [refcoco_gen_base64.py](refcoco_gen_base64.py)。

其中，`refcoco_gen` 会在运行时导出 `RefCOCO_val`、`RefCOCO_test`、`RefCOCO_testA`、`RefCOCO_testB` 四个 split 数据集简称；`refcoco_gen_base64` 会导出 `RefCOCO_base64_val`、`RefCOCO_base64_test`、`RefCOCO_base64_testA`、`RefCOCO_base64_testB` 四个 split 数据集简称。上述简称主要用于结果目录、日志和 summarizer 展示，共享同一套读取和评估逻辑。

## 使用示例

下面给出一个精简的配置片段，用于说明 RefCOCO 数据集任务中最关键的参数含义。

文件路径模式示例：

```python
from ais_bench.benchmark.openicl.icl_prompt_template import MMPromptTemplate

refcoco_reader_cfg = dict(
    input_columns=['question', 'image'],
    output_column='answer',
)

refcoco_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt_mm={
                    'text': {
                        'type': 'text',
                        'text': 'Locate every object that matches the description "{question}" in the image. Report bbox coordinates in JSON format.'
                    },
                    'image': {'type': 'image_url', 'image_url': {'url': 'file://{image}'}},
                })
            ]
        )
    ),
)

dataset = dict(
    abbr='RefCOCO_val',
    path='ais_bench/datasets/RefCOCO/data',
    split='val',
    reader_cfg=refcoco_reader_cfg,
    infer_cfg=refcoco_infer_cfg,
)

refcoco_eval_cfg = dict(
    evaluator=dict(
        type=BBoxIoUEvaluator,
        iou_threshold=0.5,
        coord_scale=1000.0,
        smart_resize_cfg=None,
    ),
    pred_postprocessor=dict(type=refcoco_bbox_postprocess),
)
```

- `input_columns=['question', 'image']`：表示 `format_mm` 会从样本的多模态内容中提取文本问题和图像输入，并映射到 prompt 模板中的 `{question}` 和 `{image}`。
- `output_column='answer'`：表示评测时使用数据集中 `answer` 列保存的标准目标框标注。
- `image_url.url='file://{image}'`：表示推理时向模型传递本地图片路径，适合能够访问文件系统路径的多模态模型或服务。
- `abbr`：运行后在输出目录、日志和汇总结果中使用的数据集简称。
- `split='val'`：表示当前数据集任务实际读取 `val-*.parquet` 分片。
- `iou_threshold=0.5`：表示只有当预测框与标准框的交并比（IoU）不低于 `0.5` 时才计为正确。
- `coord_scale=1000.0`：表示当前示例配置按 `0-1000` 归一化坐标解释模型输出。这是一个模型相关参数，当前设置主要用于适配 Qwen3-VL 技术报告中的 2D grounding 坐标约定；如果模型使用其他目标框表达方式，应相应调整该值。
- `smart_resize_cfg`：默认值为 `None`，表示不启用 smart resize 相关坐标缩放。仅当模型（如 Qwen3-VL / Qwen3.5 系列）在接收图像前执行了 smart resize 预处理，且输出坐标基于缩放后图像的坐标尺度时，才需要将此参数设置为包含 `factor`、`min_pixels`、`max_pixels` 的字典。评估器会利用该配置将模型输出的目标框坐标映射到 smart resize 后的像素坐标尺度，再参与后续交并比（IoU）计算；它并不表示将坐标逆变换回原始图像坐标系。
- `pred_postprocessor=refcoco_bbox_postprocess`：表示先从模型输出文本中提取第一组目标框坐标，再交给评估器计算指标。

base64 模式与上述配置的主要区别有两点：

```python
dataset = dict(
    abbr='RefCOCO_base64_val',
    image_type='base64',
)

image_field = {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,{image}'}}
```

- `image_type='base64'`：表示加载器会把 parquet 中的图片编码为 base64 字符串，而不是落盘为临时 JPEG 文件。
- `data:image/jpeg;base64,{image}`：表示 prompt 中直接内嵌图片内容，适合要求 data URL 输入的 API 类多模态模型。

## 使用建议

- 请选择支持图像输入的多模态生成模型，并保证模型输出中包含 4 个目标框坐标；AISBench 会使用 `refcoco_bbox_postprocess` 从预测字符串中提取坐标。
- `coord_scale=1000.0` 是当前 RefCOCO 配置采用的模型对齐参数，适用于使用 `0-1000` 归一化目标框表达方式的模型；如果模型直接输出原图像素坐标或其他尺度，应同步修改该参数。
- [refcoco_gen.py](refcoco_gen.py) 使用文件路径图像输入，prompt 中的图像字段格式为 `file://{image}`；默认建议优先使用这一模式。
- [refcoco_gen_base64.py](refcoco_gen_base64.py) 使用 base64 data URL 图像输入，prompt 中的图像字段格式为 `data:image/jpeg;base64,{image}`；如果是要求 data URL 的 API 类多模态模型，可改用该配置。
- 文件路径模式会将 parquet 中的图像落盘到 `temp_save_images` 目录，建议预留额外磁盘空间。
- base64 模式不会生成运行时图片缓存，但会增大请求体和内存占用。
