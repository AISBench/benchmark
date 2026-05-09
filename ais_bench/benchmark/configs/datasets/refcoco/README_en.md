# RefCOCO

[中文](README.md) | English

## Dataset Introduction

RefCOCO is a referring expression grounding dataset built on top of COCO images. The task is to localize the target object described by a natural-language expression and return its bounding box coordinates. AISBench currently provides four RefCOCO split tasks: `val`, `test`, `testA`, and `testB`, which correspond to about 8.81k, 5k, 1.98k, and 1.81k raw samples respectively.

In AISBench, RefCOCO is loaded from parquet shards, and the image bytes are embedded directly in the parquet files. The loader expands each row's `answer` list into multiple benchmark rows and supports two image transport modes: file-path mode caches images under `data/temp_save_images/<split>/` at runtime, while base64 mode embeds the image content directly into the multimodal request. Predictions are evaluated with `BBoxIoUEvaluator` using `Accuracy@0.5` for Intersection over Union (IoU) $\ge 0.5$. In the current configs, `coord_scale=1000.0` matches a common `0-1000` normalized bounding box convention, but this is model-specific rather than a universal requirement.

- Task type: MMU 2D grounding (multimodal visual grounding / referring expression comprehension)
- Data format: parquet shards under `data/` with filenames matching `<split>-*.parquet`
- Annotated fields: `question_id`, `image`, `question`, `answer`, `bbox`, `file_name`, and related metadata

> 🔗 Dataset Homepage

- HuggingFace: [https://huggingface.co/datasets/lmms-lab/RefCOCO](https://huggingface.co/datasets/lmms-lab/RefCOCO)

## Dataset Deployment

- The dataset source is the HuggingFace dataset repository `lmms-lab/RefCOCO`.
- The config reads from `{tool_root_path}/ais_bench/datasets/RefCOCO/data/` by default, so the full repository should be downloaded into `{tool_root_path}/ais_bench/datasets/RefCOCO/`.
- On Linux, you can deploy it with the following commands:

```bash
# On a Linux server, under the tool root path
cd ais_bench/datasets
mkdir -p RefCOCO
huggingface-cli download lmms-lab/RefCOCO --repo-type dataset --local-dir RefCOCO
```

- Run `tree RefCOCO/` under `{tool_root_path}/ais_bench/datasets`. If the directory structure looks like the following, the dataset has been deployed correctly:

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

- When using the default file-path config, the loader will automatically create JPEG caches under `data/temp_save_images/<split>/` on the first evaluation run. You do not need to create that directory in advance.
- When using file-path image mode, the cached images are stored under `{tool_root_path}/ais_bench/datasets/RefCOCO/data/temp_save_images/`. Ensure your inference server can read local files from that path. For vLLM, pass `--allowed-local-media-path {tool_root_path}/ais_bench/datasets/RefCOCO/data` when starting the server.

## Available Dataset Tasks

| Task Name          | Introduction                                                                                                                                                                                                                         | Evaluation Metric | Few-Shot | Prompt Format                             | Config File                                    |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | -------- | ----------------------------------------- | ---------------------------------------------- |
| refcoco_gen        | RefCOCO generative grounding config that uses file-path image input (`file://{image}`) and exports `RefCOCO_val`, `RefCOCO_test`, `RefCOCO_testA`, and `RefCOCO_testB` split tasks                                                   | Accuracy@0.5      | 0-shot   | Multimodal chat format (MMPromptTemplate) | [refcoco_gen.py](refcoco_gen.py)               |
| refcoco_gen_base64 | RefCOCO generative grounding config that uses base64 data-URL image input (`data:image/jpeg;base64,{image}`) and exports `RefCOCO_base64_val`, `RefCOCO_base64_test`, `RefCOCO_base64_testA`, and `RefCOCO_base64_testB` split tasks | Accuracy@0.5      | 0-shot   | Multimodal chat format (MMPromptTemplate) | [refcoco_gen_base64.py](refcoco_gen_base64.py) |

## Dataset Classification

Use `--datasets` with the following dataset config filenames:

- `refcoco_gen`: file-path image input config, defined in [refcoco_gen.py](refcoco_gen.py).
- `refcoco_gen_base64`: base64 image input config, defined in [refcoco_gen_base64.py](refcoco_gen_base64.py).

At runtime, `refcoco_gen` exports four split dataset abbreviations: `RefCOCO_val`, `RefCOCO_test`, `RefCOCO_testA`, and `RefCOCO_testB`. `refcoco_gen_base64` exports `RefCOCO_base64_val`, `RefCOCO_base64_test`, `RefCOCO_base64_testA`, and `RefCOCO_base64_testB`. These abbreviations are mainly used in outputs, logs, and summarizer displays, while both configs share the same reader and evaluator logic.

## Usage Examples

The following minimal config snippet highlights the parameters that matter most in RefCOCO dataset tasks.

File-path mode example:

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

- `input_columns=['question', 'image']`: `format_mm` fills `{question}` and `{image}` in the multimodal prompt template from the dataset sample.
- `output_column='answer'`: the evaluator reads the reference bounding box payload from the dataset's `answer` column.
- `image_url.url='file://{image}'`: the prompt passes a local image path to the model, which is suitable for multimodal models or services that can access filesystem paths.
- `abbr`: the dataset abbreviation used in outputs, logs, and summary tables.
- `split='val'`: this dataset entry loads parquet shards matching `val-*.parquet`.
- `iou_threshold=0.5`: a prediction is counted as correct only when the Intersection over Union (IoU) between the predicted box and the reference box is at least `0.5`.
- `coord_scale=1000.0`: in the current example config, model outputs are interpreted on a `0-1000` normalized bounding box scale. This is a model-specific parameter; the current value is mainly used to align with the 2D grounding coordinate convention described in the Qwen3-VL technical report, and should be changed if your model uses a different bounding box scale.
- `smart_resize_cfg`: defaults to `None`, meaning no smart resize-specific scaling is applied. Set this to a dict with `factor`, `min_pixels`, and `max_pixels` only when the model (such as the Qwen3-VL / Qwen3.5 series) applies smart resize preprocessing before receiving an image and its bounding box outputs are expressed relative to the resized-image dimensions. With this config, the evaluator scales the predicted box according to the smart-resized width/height ratios (that is, it interprets the normalized coordinates on the resized image size used by the model) before computing Intersection over Union (IoU) against the ground-truth box.
- `pred_postprocessor=refcoco_bbox_postprocess`: AISBench first extracts the first bounding box coordinate sequence from the model output text before evaluation.

Base64 mode differs from the file-path config in two key places:

```python
dataset = dict(
    abbr='RefCOCO_base64_val',
    image_type='base64',
)

image_field = {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,{image}'}}
```

- `image_type='base64'`: the loader encodes images from parquet into base64 strings instead of materializing temporary JPEG files.
- `data:image/jpeg;base64,{image}`: the prompt embeds the image bytes directly as a data URL, which is suitable for API-based multimodal models that require inline image payloads.

## Usage Recommendations

- Use a multimodal generative model that accepts image input and returns four bounding box coordinates. AISBench extracts the coordinates from the prediction string with `refcoco_bbox_postprocess`.
- `coord_scale=1000.0` is the model-alignment setting used by the current RefCOCO configs. It is suitable for models that emit `0-1000` normalized bounding box coordinates; if your model emits original-image pixels or another scale, adjust this parameter accordingly.
- [refcoco_gen.py](refcoco_gen.py) uses file-path image input, and its prompt image field is formatted as `file://{image}`. This is the recommended default mode.
- [refcoco_gen_base64.py](refcoco_gen_base64.py) uses base64 data-URL image input, and its prompt image field is formatted as `data:image/jpeg;base64,{image}`. Use it for API-based multimodal models that expect data URLs rather than filesystem paths.
- The runtime image cache under `temp_save_images` consumes additional disk space in file-path mode because images are materialized from parquet during loading.
- Base64 mode avoids runtime image caches but increases prompt payload size and request memory usage.
