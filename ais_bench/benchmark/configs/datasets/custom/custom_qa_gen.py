from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator


custom_reader_cfg = dict(
    input_columns=['question', 'max_out_len'],
    output_column='answer',
)


custom_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='Question: {question}\nAnswer: {answer}'
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

custom_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator), pred_role='BOT',
)

custom_datasets = [
    dict(
        abbr='qa',
        type=CustomDataset,
        path='ais_bench/datasets/custom/qa.jsonl', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        meta_path = '',   # 可选，传入数据集补充信息
        reader_cfg=custom_reader_cfg,
        infer_cfg=custom_infer_cfg,
        eval_cfg=custom_eval_cfg
    )
]