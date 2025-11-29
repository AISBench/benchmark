from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import VideoMMEDataset, VideoMMEEvaluator


videomme_reader_cfg = dict(
    input_columns=['question', 'video'],
    output_column='answer'
)

videomme_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question}"},
                    "video": {"type": "video_url", "video_url": {"url": "file://{video}"}},
                })
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

videomme_eval_cfg = dict(
    evaluator=dict(type=VideoMMEEvaluator)
)

videomme_datasets = [
    dict(
        abbr='videomme',
        type=VideoMMEDataset,
        path='ais_bench/datasets/Video-MME/videomme/test-00000-of-00000.parquet', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        video_path='ais_bench/datasets/Video-MME/video',
        reader_cfg=videomme_reader_cfg,
        infer_cfg=videomme_infer_cfg,
        eval_cfg=videomme_eval_cfg
    )
]