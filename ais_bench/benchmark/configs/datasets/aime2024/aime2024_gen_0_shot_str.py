from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import Aime2024Dataset, MATHEvaluator, math_postprocess_v2


aime2024_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer'
)


aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.'
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

aime2024_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2)
)

aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        path='ais_bench/datasets/aime/aime.jsonl', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=aime2024_reader_cfg,
        infer_cfg=aime2024_infer_cfg,
        eval_cfg=aime2024_eval_cfg
    )
]