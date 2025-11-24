from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import VocalSoundDataset, VocalSoundEvaluator


vocalsound_reader_cfg = dict(
    input_columns=['question', 'audio_url'],
    output_column='answer'
)


vocalsound_infer_cfg = dict(
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

vocalsound_eval_cfg = dict(
    evaluator=dict(type=VocalSoundEvaluator)
)

vocalsound_datasets = [
    dict(
        abbr='vocalsound',
        type=VocalSoundDataset,
        path='ais_bench/datasets/vocalsound', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        audio_type="audio_path",
        reader_cfg=vocalsound_reader_cfg,
        infer_cfg=vocalsound_infer_cfg,
        eval_cfg=vocalsound_eval_cfg
    )
]