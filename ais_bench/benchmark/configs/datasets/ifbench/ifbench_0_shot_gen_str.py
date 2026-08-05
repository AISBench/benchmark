from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets.ifbench import IFBenchDataset, IFBenchEvaluator

ifbench_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='reference',
)

ifbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}',
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

ifbench_eval_cfg = dict(
    evaluator=dict(type=IFBenchEvaluator),
)

ifbench_datasets = [
    dict(
        abbr='ifbench',
        type=IFBenchDataset,
        path='ais_bench/datasets/ifbench/data/train-00000-of-00001.parquet',
        nltk_path='/path/to/nltk_data',
        reader_cfg=ifbench_reader_cfg,
        infer_cfg=ifbench_infer_cfg,
        eval_cfg=ifbench_eval_cfg,
    )
]
