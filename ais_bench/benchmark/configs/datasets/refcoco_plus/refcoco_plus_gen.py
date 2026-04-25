from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import MMPromptTemplate
from ais_bench.benchmark.datasets import RefCOCOPlusDataset, refcoco_bbox_postprocess
from ais_bench.benchmark.openicl.icl_evaluator import BBoxIoUEvaluator


refcoco_plus_reader_cfg = dict(
    input_columns=["question", "image"], output_column="answer"
)

refcoco_plus_infer_cfg = dict(
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
                            "image_url": {"url": "file://{image}"},
                        },
                    },
                )
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

refcoco_plus_eval_cfg = dict(
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
    "testA",
    "testB",
]

refcoco_plus_datasets = [
    dict(
        abbr="RefCOCOPlus_" + split,
        type=RefCOCOPlusDataset,
        path="ais_bench/datasets/RefCOCOplus/data",
        split=split,
        reader_cfg=refcoco_plus_reader_cfg,
        infer_cfg=refcoco_plus_infer_cfg,
        eval_cfg=refcoco_plus_eval_cfg,
    )
    for split in _splits
]
