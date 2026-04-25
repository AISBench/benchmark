from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import MMPromptTemplate
from ais_bench.benchmark.datasets import RefCOCOgDataset, refcoco_bbox_postprocess
from ais_bench.benchmark.openicl.icl_evaluator import BBoxIoUEvaluator


refcocog_reader_cfg = dict(input_columns=["question", "image"], output_column="answer")

refcocog_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt_mm={
                        "text": {
                            "type": "text",
                            "text": 'Locate every object that matches the description "{question}" in the image. Report bbox coordinates in JSON format.',
                        },
                        "image": {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,{image}"},
                        },
                    },
                )
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

refcocog_eval_cfg = dict(
    evaluator=dict(
        type=BBoxIoUEvaluator,
        iou_threshold=0.5,
        coord_scale=1000.0,
        smart_resize_cfg=dict(
            factor=32,
            min_pixels=65536,
            max_pixels=16 * 16 * 4 * 16384,
        ),
    ),
    pred_postprocessor=dict(type=refcoco_bbox_postprocess),
)

_splits = [
    "val",
    "test",
]

refcocog_datasets = [
    dict(
        abbr="RefCOCOg_base64_" + split,
        type=RefCOCOgDataset,
        path="ais_bench/datasets/RefCOCOg/data",
        split=split,
        image_type="base64",
        reader_cfg=refcocog_reader_cfg,
        infer_cfg=refcocog_infer_cfg,
        eval_cfg=refcocog_eval_cfg,
    )
    for split in _splits
]
