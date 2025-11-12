from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import (
    Aime2025Dataset,
    MATHEvaluator,
    math_postprocess_v2,
)


aime2025_reader_cfg = dict(input_columns=["question"], output_column="answer")


aime2025_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="{question}\nRemember to put your final answer within \\boxed{}.",
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

aime2025_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version="v2"),
    pred_postprocessor=dict(type=math_postprocess_v2),
)

aime2025_datasets = [
    dict(
        abbr="aime2025",
        type=Aime2025Dataset,
        path="ais_bench/datasets/aime2025/aime2025.jsonl",
        reader_cfg=aime2025_reader_cfg,
        infer_cfg=aime2025_infer_cfg,
        eval_cfg=aime2025_eval_cfg,
    )
]
