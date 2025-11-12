from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import FixKRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccwithDetailsEvaluator
from ais_bench.benchmark.datasets import WinograndeDatasetV3
from ais_bench.benchmark.utils.postprocess.text_postprocessors import last_option_postprocess

winogrande_reader_cfg = dict(
    input_columns=['prompt', 'only_option1', 'only_option2'],
    output_column='answer',
    train_split='train_xs',
    test_split='dev',
)

_hint = 'There is a single choice question about physical knowledge. Answer the question by replying A or B. ' + \
        'The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the question.\n\n'

winogrande_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt=f'{_hint}Question: {{prompt}}\nA. {{only_option1}}\nB. {{only_option2}}'),
                dict(role='BOT', prompt='Answer: {answer}'),
            ]
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 2, 4, 6, 8]),
    inferencer=dict(type=GenInferencer),
)

winogrande_eval_cfg = dict(
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=last_option_postprocess, options='AB'),
)

winogrande_datasets = [
    dict(
        abbr='winogrande',
        type=WinograndeDatasetV3,
        path='ais_bench/datasets/winogrande',
        reader_cfg=winogrande_reader_cfg,
        infer_cfg=winogrande_infer_cfg,
        eval_cfg=winogrande_eval_cfg,
    )
]
