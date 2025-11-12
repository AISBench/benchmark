from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import ARCDataset
from ais_bench.benchmark.utils.postprocess.text_postprocessors import last_option_postprocess

ARC_e_reader_cfg = dict(
    input_columns=['question', 'textA', 'textB', 'textC', 'textD'],
    output_column='answerKey',
    train_split='Dev',
    test_split='Test',
)

_hint = f'There is a single choice question. Answer the question by replying A, B, C or D. ' + \
        'The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the question.'

ARC_e_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=
                    'Question: {question}\nA. {textA}\nB. {textB}\nC. {textC}\nD. {textD}\nAnswer:'
                )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

ARC_e_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=last_option_postprocess, options='ABCD'),
)

ARC_e_datasets = [
    dict(
        abbr='ARC-e',
        type=ARCDataset,
        path='ais_bench/datasets/ARC/ARC-e',
        name='ARC-Easy',
        reader_cfg=ARC_e_reader_cfg,
        infer_cfg=ARC_e_infer_cfg,
        eval_cfg=ARC_e_eval_cfg,
    )
]
