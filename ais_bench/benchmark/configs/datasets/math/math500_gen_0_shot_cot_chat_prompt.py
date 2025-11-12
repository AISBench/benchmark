from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.openicl.icl_evaluator import MATHEvaluator
math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.'
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

math_eval_cfg = dict(evaluator=dict(type=MATHEvaluator))

math_datasets = [
    dict(
        type=CustomDataset,
        abbr='math_prm800k_500',
        path='ais_bench/datasets/math',  # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        file_name='test_prm800k_500.jsonl',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
