# RefCOCO+

[中文](README.md) | English

## Dataset Introduction

RefCOCO+ is a variant of RefCOCO for evaluating whether a model can localize the target object in an image from a referring expression. AISBench currently provides three split tasks: `val`, `testA`, and `testB`, with about 3.81k, 1.98k, and 1.8k raw samples respectively.

RefCOCO+ shares the same data schema and evaluation flow as RefCOCO. The dataset is consumed from parquet shards, and the image bytes are embedded directly in the parquet files. During loading, AISBench expands each row's `answer` list and supports both the default file-path mode with runtime image caches under `data/temp_save_images/<split>/` and an opt-in base64 mode for API-based multimodal requests. Predictions are scored with `BBoxIoUEvaluator` using `Accuracy@0.5` for Intersection over Union (IoU) $\ge 0.5$. In the current configs, `coord_scale=1000.0` matches a common `0-1000` normalized bounding box convention, but this is model-specific rather than a universal requirement.

- Task type: MMU 2D grounding (multimodal visual grounding / referring expression comprehension)
- Dataset characteristic: compared with RefCOCO, RefCOCO+ avoids explicit spatial-location words in the referring expressions
- Data format: `<split>-*.parquet` files under the `data/` directory

> 🔗 Dataset Homepage [https://huggingface.co/datasets/lmms-lab/RefCOCOplus](https://huggingface.co/datasets/lmms-lab/RefCOCOplus)

## Dataset Deployment

- The dataset source is the HuggingFace dataset repository `lmms-lab/RefCOCOplus`.
- The config reads from `{tool_root_path}/ais_bench/datasets/RefCOCOplus/data/` by default, so the full repository should be downloaded into `{tool_root_path}/ais_bench/datasets/RefCOCOplus/`.
- On Linux, you can deploy it with the following commands:

```bash
# On a Linux server, under the tool root path
cd ais_bench/datasets
mkdir -p RefCOCOplus
huggingface-cli download lmms-lab/RefCOCOplus --repo-type dataset --local-dir RefCOCOplus
```

- Run `tree RefCOCOplus/` under `{tool_root_path}/ais_bench/datasets`. If the directory structure looks like the following, the dataset has been deployed correctly:

```text
RefCOCOplus/
├── .gitattributes
├── README.md
└── data/
    ├── testA-*.parquet
    ├── testB-*.parquet
    └── val-*.parquet
```

- When using the default file-path config, the loader will automatically create JPEG caches under `data/temp_save_images/<split>/` on the first evaluation run. You do not need to create that directory in advance.
- When using file-path image mode, the cached images are stored under `{tool_root_path}/ais_bench/datasets/RefCOCOplus/data/temp_save_images/`. Ensure your inference server can read local files from that path. For vLLM, pass `--allowed-local-media-path {tool_root_path}/ais_bench/datasets/RefCOCOplus/data` when starting the server.

## Available Dataset Tasks

| Task Name               | Introduction                                                                                                                                                                                                               | Evaluation Metric | Few-Shot | Prompt Format                             | Config File                                              |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | -------- | ----------------------------------------- | -------------------------------------------------------- |
| refcoco_plus_gen        | RefCOCO+ generative grounding config that uses file-path image input (`file://{image}`) and exports `RefCOCOPlus_val`, `RefCOCOPlus_testA`, and `RefCOCOPlus_testB` split tasks                                            | Accuracy@0.5      | 0-shot   | Multimodal chat format (MMPromptTemplate) | [refcoco_plus_gen.py](refcoco_plus_gen.py)               |
| refcoco_plus_gen_base64 | RefCOCO+ generative grounding config that uses base64 data-URL image input (`data:image/jpeg;base64,{image}`) and exports `RefCOCOPlus_base64_val`, `RefCOCOPlus_base64_testA`, and `RefCOCOPlus_base64_testB` split tasks | Accuracy@0.5      | 0-shot   | Multimodal chat format (MMPromptTemplate) | [refcoco_plus_gen_base64.py](refcoco_plus_gen_base64.py) |

## Dataset Classification

Use `--datasets` with the following dataset config filenames:

- `refcoco_plus_gen`: file-path image input config, defined in [refcoco_plus_gen.py](refcoco_plus_gen.py).
- `refcoco_plus_gen_base64`: base64 image input config, defined in [refcoco_plus_gen_base64.py](refcoco_plus_gen_base64.py).

At runtime, `refcoco_plus_gen` exports three split dataset abbreviations: `RefCOCOPlus_val`, `RefCOCOPlus_testA`, and `RefCOCOPlus_testB`. `refcoco_plus_gen_base64` exports `RefCOCOPlus_base64_val`, `RefCOCOPlus_base64_testA`, and `RefCOCOPlus_base64_testB`. These abbreviations are mainly used in outputs, logs, and summarizer displays, while both configs share the same reader and evaluator logic.

## Usage Examples

The following minimal config snippet highlights the parameters that matter most in RefCOCO+ dataset tasks.

File-path mode example:

```python
from ais_bench.benchmark.openicl.icl_prompt_template import MMPromptTemplate

refcoco_plus_reader_cfg = dict(
    input_columns=['question', 'image'],
    output_column='answer',
)

refcoco_plus_infer_cfg = dict(
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
    abbr='RefCOCOPlus_val',
    path='ais_bench/datasets/RefCOCOplus/data',
    split='val',
    reader_cfg=refcoco_plus_reader_cfg,
    infer_cfg=refcoco_plus_infer_cfg,
)

refcoco_plus_eval_cfg = dict(
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
- `image_url.url='file://{image}'`: the prompt passes a local image path to the model.
- `abbr`: the dataset abbreviation used in outputs, logs, and summary tables.
- `split='val'`: this dataset entry loads parquet shards matching `val-*.parquet`.
- `iou_threshold=0.5`: a prediction is counted as correct only when the Intersection over Union (IoU) between the predicted box and the reference box is at least `0.5`.
- `coord_scale=1000.0`: in the current example config, model outputs are interpreted on a `0-1000` normalized bounding box scale. This is a model-specific parameter; the current value is mainly used to align with the 2D grounding coordinate convention described in the Qwen3-VL technical report, and should be changed if your model uses a different bounding box scale.
- `smart_resize_cfg`: defaults to `None`, meaning no smart-resize-related coordinate scaling is applied. Set this to a dict with `factor`, `min_pixels`, and `max_pixels` only when the model (such as the Qwen3-VL / Qwen3.5 series) applies smart resize preprocessing before receiving an image and its bounding box outputs are expressed relative to the resized image. The evaluator uses this config to scale the model's predicted box into the smart-resized image coordinate space used by the current evaluation flow; it does not perform an inverse transform back to the original image coordinates.
- `pred_postprocessor=refcoco_bbox_postprocess`: AISBench first extracts the first bounding box coordinate sequence from the model output text before evaluation.

Base64 mode differs from the file-path config in two key places:

```python
dataset = dict(
    abbr='RefCOCOPlus_base64_val',
    image_type='base64',
)

image_field = {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,{image}'}}
```

- `image_type='base64'`: the loader encodes images from parquet into base64 strings instead of materializing temporary JPEG files.
- `data:image/jpeg;base64,{image}`: the prompt embeds the image bytes directly as a data URL, which is suitable for API-based multimodal models.

## Usage Recommendations

- The RefCOCO+ evaluation flow is identical to RefCOCO and still requires the model to return a parsable four-value bbox.
- `refcoco_bbox_postprocess` extracts the first coordinate sequence shaped like `[x1, y1, x2, y2]` from the prediction text, so stable JSON or array-style output is recommended.
- `coord_scale=1000.0` is the model-alignment setting used by the current RefCOCO+ configs. It is suitable for models that emit `0-1000` normalized bounding box coordinates; if your model emits original-image pixels or another scale, adjust this parameter accordingly.
- [refcoco_plus_gen.py](refcoco_plus_gen.py) uses file-path image input, and its prompt image field is formatted as `file://{image}`. This is the recommended default mode.
- [refcoco_plus_gen_base64.py](refcoco_plus_gen_base64.py) uses base64 data-URL image input, and its prompt image field is formatted as `data:image/jpeg;base64,{image}`. Use it for API-based multimodal models that expect base64 data URLs.
