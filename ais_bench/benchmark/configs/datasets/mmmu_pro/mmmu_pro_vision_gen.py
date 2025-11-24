from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MMMUProVisionDataset, MMMUProEvaluator


mmmu_pro_reader_cfg = dict(
    input_columns=['content'],
    output_column='answer'
)

mmmu_pro_infer_cfg = dict(
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

mmmu_pro_eval_cfg = dict(
    evaluator=dict(type=MMMUProEvaluator)
)

mmmu_pro_datasets = [
    dict(
        abbr='mmmu_pro',
        type=MMMUProVisionDataset,
        path='ais_bench/datasets/mmmu/MMMU_Pro_V.tsv', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        is_cot=False,
        reader_cfg=mmmu_pro_reader_cfg,
        infer_cfg=mmmu_pro_infer_cfg,
        eval_cfg=mmmu_pro_eval_cfg
    )
]