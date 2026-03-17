from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import MMPromptTemplate
from ais_bench.benchmark.datasets import RefCOCOPlusDataset
from ais_bench.benchmark.datasets.refcoco import refcoco_bbox_postprocess
from ais_bench.benchmark.openicl.icl_evaluator import BBoxIoUEvaluator


refcoco_plus_reader_cfg = dict(
    input_columns=['content'],
    output_column='answer'
)

refcoco_plus_infer_cfg = dict(
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

refcoco_plus_eval_cfg = dict(
    evaluator=dict(type=BBoxIoUEvaluator, iou_threshold=0.5, coord_scale=1000.0),
    pred_postprocessor=dict(type=refcoco_bbox_postprocess),
)

_splits = [
    ('RefCOCOPlus_val', 'val'),
    ('RefCOCOPlus_testA', 'testA'),
    ('RefCOCOPlus_testB', 'testB'),
]

refcoco_plus_datasets = [
    dict(
        abbr=abbr,
        type=RefCOCOPlusDataset,
        path='ais_bench/datasets/RefCOCOplus/data',
        split=split,
        reader_cfg=refcoco_plus_reader_cfg,
        infer_cfg=refcoco_plus_infer_cfg,
        eval_cfg=refcoco_plus_eval_cfg,
    )
    for abbr, split in _splits
]