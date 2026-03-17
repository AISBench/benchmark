from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import MMPromptTemplate
from ais_bench.benchmark.datasets import RefCOCOgDataset
from ais_bench.benchmark.datasets.refcoco import refcoco_bbox_postprocess
from ais_bench.benchmark.openicl.icl_evaluator import BBoxIoUEvaluator


refcocog_reader_cfg = dict(
    input_columns=['content'],
    output_column='answer'
)

refcocog_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt_mm={
                    'text': {'type': 'text', 'text': '{question}'},
                    'image': {'type': 'image_url', 'image_url': {'url': 'file://{image}'}},
                })
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

refcocog_eval_cfg = dict(
    evaluator=dict(type=BBoxIoUEvaluator, iou_threshold=0.5, coord_scale=1000.0),
    pred_postprocessor=dict(type=refcoco_bbox_postprocess),
)

_splits = [
    ('RefCOCOg_val', 'val'),
    ('RefCOCOg_test', 'test'),
]

refcocog_datasets = [
    dict(
        abbr=abbr,
        type=RefCOCOgDataset,
        path='ais_bench/datasets/RefCOCOg/data',
        split=split,
        reader_cfg=refcocog_reader_cfg,
        infer_cfg=refcocog_infer_cfg,
        eval_cfg=refcocog_eval_cfg,
    )
    for abbr, split in _splits
]