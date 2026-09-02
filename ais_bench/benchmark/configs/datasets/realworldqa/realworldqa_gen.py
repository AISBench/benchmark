from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import RealworldQADataset, RealworldQAEvaluator


realworldqa_reader_cfg = dict(
    input_columns=['question', 'image'],
    output_column='answer'
)


realworldqa_infer_cfg = dict(
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


realworldqa_eval_cfg = dict(
    evaluator=dict(type=RealworldQAEvaluator)
)


realworldqa_datasets = [
    dict(
        abbr='realworldqa',
        type=RealworldQADataset,
        path='ais_bench/datasets/RealworldQA/data/test-00000-of-00002.parquet',
        image_type="image_path",
        reader_cfg=realworldqa_reader_cfg,
        infer_cfg=realworldqa_infer_cfg,
        eval_cfg=realworldqa_eval_cfg
    )
]
