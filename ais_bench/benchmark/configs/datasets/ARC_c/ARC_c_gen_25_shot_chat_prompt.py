from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import ARCDataset
from ais_bench.benchmark.utils.postprocess.text_postprocessors import last_capital_postprocess

ARC_c_reader_cfg = dict(
    input_columns=['question', 'textA', 'textB', 'textC', 'textD'],
    output_column='answerKey',
    train_split='Dev',
    test_split='Test',
)

_hint = f'There is a single choice question. Answer the question by replying A, B, C or D. ' + \
        'The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the question.'

ARC_c_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(
                    role='HUMAN',
                    prompt='Question: {question}\nA. {textA}\nB. {textB}\nC. {textC}\nD. {textD}\nAnswer:',
                ),
                dict(role='BOT', prompt='{answerKey}'),
            ],
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[i for i in range(25)]),
    inferencer=dict(type=GenInferencer),
)

ARC_c_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=last_capital_postprocess),
)

ARC_c_datasets = [
    dict(
        abbr='ARC-c',
        type=ARCDataset,
        path='ais_bench/datasets/ARC/ARC-c',
        name='ARC-Challenge',
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg,
        eval_cfg=ARC_c_eval_cfg,
    )
]
