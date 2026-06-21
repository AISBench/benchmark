from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets.aa_lcr import AALCRDataset, AALCREvaluator

aa_lcr_reader_cfg = dict(
    input_columns=['input'],
    output_column='answers',
)

aa_lcr_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{input}',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

aa_lcr_eval_cfg = dict(
    evaluator=dict(type=AALCREvaluator),
)

aa_lcr_datasets = [
    dict(
        abbr='aa_lcr',
        type=AALCRDataset,
        path='ais_bench/datasets/aa_lcr/input_data.jsonl',
        reader_cfg=aa_lcr_reader_cfg,
        infer_cfg=aa_lcr_infer_cfg,
        eval_cfg=aa_lcr_eval_cfg,
    )
]
