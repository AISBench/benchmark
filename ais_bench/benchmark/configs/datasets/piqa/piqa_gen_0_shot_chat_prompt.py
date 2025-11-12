from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import PIQADatasetV2
from ais_bench.benchmark.utils.postprocess.text_postprocessors import last_option_postprocess

piqa_reader_cfg = dict(
    input_columns=['goal', 'sol1', 'sol2'],
    output_column='answer',
    test_split='validation')

_hint = 'There is a single choice question about physical knowledge. Answer the question by replying A or B. ' + \
        'The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the question.\n\n'

piqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=f'{_hint}{{goal}}\nA. {{sol1}}\nB. {{sol2}}')
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

piqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=last_option_postprocess, options='AB'),
)

piqa_datasets = [
    dict(
        abbr='piqa',
        type=PIQADatasetV2,
        path='ais_bench/datasets/physicaliqa-train-dev',
        reader_cfg=piqa_reader_cfg,
        infer_cfg=piqa_infer_cfg,
        eval_cfg=piqa_eval_cfg)
]
