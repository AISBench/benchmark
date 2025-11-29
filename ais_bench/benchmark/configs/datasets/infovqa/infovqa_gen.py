from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import InfoVQADataset, InfoVQAEvaluator


infovqa_reader_cfg = dict(
    input_columns=['question', 'image'],
    output_column='answer'
)

infovqa_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question}"},
                    "image": {"type": "image_url", "image_url": {"url": "file://{image}"}},
                })
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

infovqa_eval_cfg = dict(
    evaluator=dict(type=InfoVQAEvaluator)
)

infovqa_datasets = [
    dict(
        abbr='InfoVQA',
        type=InfoVQADataset,
        path='ais_bench/datasets/InfoVQA/InfoVQA_VAL.tsv', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=infovqa_reader_cfg,
        infer_cfg=infovqa_infer_cfg,
        eval_cfg=infovqa_eval_cfg
    )
]