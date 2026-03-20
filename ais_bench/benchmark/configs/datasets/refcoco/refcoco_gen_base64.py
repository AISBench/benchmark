from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import MMPromptTemplate
from ais_bench.benchmark.datasets import RefCOCODataset
from ais_bench.benchmark.datasets.refcoco import IMAGE_BASE64_TYPE, refcoco_bbox_postprocess
from ais_bench.benchmark.openicl.icl_evaluator import BBoxIoUEvaluator


refcoco_reader_cfg = dict(
    input_columns=['question', 'image'],
    output_column='answer'
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
                    'image': {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,{image}'}},
                })
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

refcoco_eval_cfg = dict(
    evaluator=dict(type=BBoxIoUEvaluator, iou_threshold=0.5, coord_scale=1000.0),
    pred_postprocessor=dict(type=refcoco_bbox_postprocess),
)

_splits = [
    'val',
    'test',
    'testA',
    'testB',
]

refcoco_datasets = [
    dict(
        abbr='RefCOCO_base64_' + split,
        type=RefCOCODataset,
        path='ais_bench/datasets/RefCOCO/data',
        split=split,
        image_type=IMAGE_BASE64_TYPE,
        reader_cfg=refcoco_reader_cfg,
        infer_cfg=refcoco_infer_cfg,
        eval_cfg=refcoco_eval_cfg,
    )
    for split in _splits
]