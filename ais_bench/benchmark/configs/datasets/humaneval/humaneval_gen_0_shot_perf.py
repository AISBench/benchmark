from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import HumanevalDataset, HumanEvalEvaluator, humaneval_postprocess_v2

humaneval_reader_cfg = dict(input_columns=['prompt'], output_column='task_id', train_split='test')

HUMANEVAL_TEMPLATE = '{prompt}\n'

humaneval_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template=HUMANEVAL_TEMPLATE),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

humaneval_eval_cfg = dict(
    evaluator=dict(type=HumanEvalEvaluator),
    k=[1, 10, 100],
    pred_postprocessor=dict(type=humaneval_postprocess_v2),
)

humaneval_datasets = [
    dict(
        abbr='openai_humaneval',
        type=HumanevalDataset,
        path='ais_bench/datasets/humaneval/human-eval-v2-20210705.jsonl',  # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=humaneval_reader_cfg,
        infer_cfg=humaneval_infer_cfg,
        eval_cfg=humaneval_eval_cfg,
    )
]