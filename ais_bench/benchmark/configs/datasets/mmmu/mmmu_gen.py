from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MMMUDataset, MMMUEvaluator


mmmu_reader_cfg = dict(
    input_columns=['content'],
    output_column='answer'
)

mmmu_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question}"},
                    "image": {"type": "image_url", "image_url": {"url": "file://{image}"}},
                    "video": {"type": "video_url", "video_url": {"url": "file://{video}"}},
                    "audio": {"type": "audio_url", "audio_url": {"url": "file://{audio}"}},
                })
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

mmmu_eval_cfg = dict(
    evaluator=dict(type=MMMUEvaluator)
)

mmmu_datasets = [
    dict(
        abbr='mmmu',
        type=MMMUDataset,
        path='ais_bench/datasets/mmmu/MMMU_DEV_VAL.tsv', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=mmmu_reader_cfg,
        infer_cfg=mmmu_infer_cfg,
        eval_cfg=mmmu_eval_cfg
    )
]