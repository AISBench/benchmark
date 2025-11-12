from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import FixKRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import (
    MATHDataset,
    MATHEvaluator,
    math_postprocess_v2,
    normalize_final_answer,
)

math_reader_cfg = dict(input_columns=['problem'], output_column='solution', train_split='train')

math_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=
        '{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.\nAnswer: {solution}\n',
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template='</E>\n{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.',
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=GenInferencer)
)

# postprocess v2
math_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'),
    pred_postprocessor=dict(type=math_postprocess_v2),
)

math_datasets = [
    dict(
        type=MATHDataset,
        abbr='math_prm800k_500',
        path='ais_bench/datasets/math',  # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        file_name='test_prm800k_500.json',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
