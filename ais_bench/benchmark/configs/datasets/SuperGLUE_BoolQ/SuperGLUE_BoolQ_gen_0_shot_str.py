from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import BoolQDatasetV2
from ais_bench.benchmark.utils.postprocess.text_postprocessors import first_capital_postprocess

BoolQ_reader_cfg = dict(
    input_columns=['question', 'passage'],
    output_column='label',
)

BoolQ_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{passage}\nQuestion: {question}\nA. Yes\nB. No\nAnswer:'
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

BoolQ_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

BoolQ_datasets = [
    dict(
        abbr='BoolQ',
        type=BoolQDatasetV2,
        path='ais_bench/datasets/SuperGLUE/BoolQ/val.jsonl',  # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=BoolQ_reader_cfg,
        infer_cfg=BoolQ_infer_cfg,
        eval_cfg=BoolQ_eval_cfg,
    )
]
