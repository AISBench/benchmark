from ais_bench.benchmark.datasets import MMEDataset, MMEEvaluator
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

mme_reader_cfg = dict(
    input_columns=["question", "image"],
    output_column="reference",
)

mme_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt_mm={
                        "text": {"type": "text", "text": "{question}"},
                        "image": {
                            "type": "image_url",
                            "image_url": {"url": "{image}"},
                        },
                    },
                )
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

mme_eval_cfg = dict(
    evaluator=dict(
        type=MMEEvaluator,
        results_subdir="mme_results",
    )
)

mme_datasets = [
    dict(
        abbr="mme",
        type=MMEDataset,
        path=r"C:\需求\MME\MME\data",
        reader_cfg=mme_reader_cfg,
        infer_cfg=mme_infer_cfg,
        eval_cfg=mme_eval_cfg,
    )
]
