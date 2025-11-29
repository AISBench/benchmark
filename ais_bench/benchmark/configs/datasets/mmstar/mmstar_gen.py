from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MMStarDataset, MMStarEvaluator


mmstar_reader_cfg = dict(
    input_columns=['question', 'image'],
    output_column='answer'
)

mmstar_infer_cfg = dict(
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

mmstar_eval_cfg = dict(
    evaluator=dict(type=MMStarEvaluator)
)

mmstar_datasets = [
    dict(
        abbr='mmstar',
        type=MMStarDataset,
        path='ais_bench/datasets/mmstar/MMStar.tsv', # Dataset path. Relative to the root of the source code. Absolute paths are also supported.
        reader_cfg=mmstar_reader_cfg,
        infer_cfg=mmstar_infer_cfg,
        eval_cfg=mmstar_eval_cfg
    )
]
