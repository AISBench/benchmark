from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import FixKRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import RaceDataset
from ais_bench.benchmark.utils.postprocess.text_postprocessors import (
    first_option_postprocess,
)

QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

Article: {article}

Q: {question}

A. {A}
B. {B}
C. {C}
D. {D}
""".strip()

race_reader_cfg = dict(
    input_columns=['article', 'question', 'A', 'B', 'C', 'D'],
    output_column='answer',
    train_split='validation',
    test_split='test',
)

race_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt=QUERY_TEMPLATE),
                dict(role='BOT', prompt='{answer}'),
            ],
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=GenInferencer),
)

race_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    pred_role='BOT',
)

race_datasets = [
    dict(
        abbr='race-middle',
        type=RaceDataset,
        path='ais_bench/datasets/race',
        name='middle',
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg,
    )
]
