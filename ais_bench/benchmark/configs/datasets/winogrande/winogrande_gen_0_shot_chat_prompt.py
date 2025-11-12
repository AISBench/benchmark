from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import WinograndeDatasetV2
from ais_bench.benchmark.utils.postprocess.text_postprocessors import last_option_postprocess

winogrande_reader_cfg = dict(
    input_columns=['prompt', 'only_option1', 'only_option2'],
    output_column='answer',
)

_hint = 'There is a single choice question about physical knowledge. Answer the question by replying A or B. ' + \
        'The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the question.\n\n'

winogrande_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=f'{_hint}Question: {{prompt}}\nA. {{only_option1}}\nB. {{only_option2}}'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

winogrande_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=last_option_postprocess, options='AB'),
)

winogrande_datasets = [
    dict(
        abbr='winogrande',
        type=WinograndeDatasetV2,
        path='ais_bench/datasets/winogrande',
        reader_cfg=winogrande_reader_cfg,
        infer_cfg=winogrande_infer_cfg,
        eval_cfg=winogrande_eval_cfg,
    )
]
