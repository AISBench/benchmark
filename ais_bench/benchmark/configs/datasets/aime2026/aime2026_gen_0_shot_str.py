from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import Aime2026Dataset, MATHEvaluator, math_postprocess_v2


aime2026_reader_cfg = dict(
    input_columns=['problem'],
    output_column='answer'
)


aime2026_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='\nSolve the following math problem step by step. Put your answer inside \\boxed{}.\n\n{problem}\n\nRemember to put your answer inside \\boxed{}.'
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

aime2026_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2)
)

aime2026_datasets = [
    dict(
        abbr='aime2026',
        type=Aime2026Dataset,
        path='ais_bench/datasets/aime2026/aime2026.jsonl',
        reader_cfg=aime2026_reader_cfg,
        infer_cfg=aime2026_infer_cfg,
        eval_cfg=aime2026_eval_cfg
    )
]