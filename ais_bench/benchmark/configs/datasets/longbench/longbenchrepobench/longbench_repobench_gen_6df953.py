from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import LongBenchCodeSimEvaluator, LongBenchrepobenchDataset

LongBench_repobench_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

LongBench_repobench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Please complete the code given below. \n{context}{input}Next line of code:\n',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

LongBench_repobench_eval_cfg = dict(
    evaluator=dict(type=LongBenchCodeSimEvaluator), pred_role='BOT'
)

LongBench_repobench_datasets = [
    dict(
        type=LongBenchrepobenchDataset,
        abbr='LongBench_repobench-p',
        path='ais_bench/datasets/LongBench',
        name='repobench-p',
        reader_cfg=LongBench_repobench_reader_cfg,
        infer_cfg=LongBench_repobench_infer_cfg,
        eval_cfg=LongBench_repobench_eval_cfg,
    )
]
