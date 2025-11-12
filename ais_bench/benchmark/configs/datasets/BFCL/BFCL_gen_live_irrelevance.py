from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import BFCLDataset, BFCLRelevanceEvaluator

bfcl_category = "live_irrelevance"

bfcl_reader_cfg = dict(
    input_columns=["id", "question", "function"],
    output_column="ground_truth",
)

bfcl_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template="{question}"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

bfcl_eval_cfg = dict(
    evaluator=dict(
        type=BFCLRelevanceEvaluator, category=bfcl_category
    ),
)

bfcl_datasets = [
    dict(
        abbr=f"BFCL-v3-{bfcl_category}",
        type=BFCLDataset,
        category=bfcl_category,
        path="", # Use the dataset in the bfcl_eval site package by default, no need to specify.
        reader_cfg=bfcl_reader_cfg,
        infer_cfg=bfcl_infer_cfg,
        eval_cfg=bfcl_eval_cfg,
    )
] 