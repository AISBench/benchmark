from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import BFCLDataset, BFCLMultiTurnEvaluator

bfcl_category = "multi_turn_base"

bfcl_reader_cfg = dict(
    input_columns=["id", "question", "function", "missed_function", "involved_classes", "initial_config"],
    output_column="ground_truth",
)

bfcl_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template="{question}"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

bfcl_eval_cfg = dict(
    evaluator=dict(
        type=BFCLMultiTurnEvaluator, category=bfcl_category
    ),
)

bfcl_datasets = [
    dict(
        abbr=f"BFCL-v3-{bfcl_category}",
        type=BFCLDataset,
        category=bfcl_category,
        path="", # Use the dataset in the bfcl_eval site package by default, no need to specify.  # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=bfcl_reader_cfg,
        infer_cfg=bfcl_infer_cfg,
        eval_cfg=bfcl_eval_cfg,
    )
]
