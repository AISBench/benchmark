from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import LongBenchF1Evaluator, LongBenchmusiqueDataset

LongBench_musique_reader_cfg = dict(
    input_columns=['context', 'input'],
    output_column='answers',
    train_split='test',
    test_split='test',
)

LongBench_musique_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:',
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

LongBench_musique_eval_cfg = dict(
    evaluator=dict(type=LongBenchF1Evaluator), pred_role='BOT'
)

LongBench_musique_datasets = [
    dict(
        type=LongBenchmusiqueDataset,
        abbr='LongBench_musique',
        path='ais_bench/datasets/LongBench',
        name='musique',
        reader_cfg=LongBench_musique_reader_cfg,
        infer_cfg=LongBench_musique_infer_cfg,
        eval_cfg=LongBench_musique_eval_cfg,
    )
]
