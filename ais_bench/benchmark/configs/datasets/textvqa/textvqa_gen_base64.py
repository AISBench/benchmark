from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import TEXTVQADataset, TEXTEvaluator


textvqa_reader_cfg = dict(
    input_columns=['question', 'image_url'],
    output_column='answer'
)


textvqa_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question} Answer the question using a single word or phrase."},
                    "image": {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
                    "video": {"type": "video_url", "video_url": {"url": "data:video/jpeg;base64,{video}"}},
                    "audio": {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,{audio}"}},
                })
            ]
            )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

textvqa_eval_cfg = dict(
    evaluator=dict(type=TEXTEvaluator)
)

textvqa_datasets = [
    dict(
        abbr='textvqa',
        type=TEXTVQADataset,
        path='ais_bench/datasets/textvqa/textvqa_json/textvqa_val.jsonl', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        image_type="image_base64",
        reader_cfg=textvqa_reader_cfg,
        infer_cfg=textvqa_infer_cfg,
        eval_cfg=textvqa_eval_cfg
    )
]