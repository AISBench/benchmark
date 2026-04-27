from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MathVisionDataset, MathVisionEvaluator


OPEN_PROMPT_TEMPLATE = (
    '{question}\nPlease reason step by step, and put your final answer within \\boxed{{}} without units.'
)

SINGLE_ANSWER_COT_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the following "
    "format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before "
    'answering.\n\n{question}\n\n{choices}'
)

mathvision_reader_cfg = dict(
    input_columns=['content'],
    output_column='answer'
)

mathvision_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt_mm={
                    'text': {'type': 'text', 'text': '{question}'},
                    'image': {'type': 'image_url', 'image_url': {'url': 'file://{image}'}},
                })
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

mathvision_eval_cfg = dict(
    evaluator=dict(type=MathVisionEvaluator)
)

mathvision_datasets = [
    dict(
        abbr='MathVision',
        type=MathVisionDataset,
        path='ais_bench/datasets/mathvision',
        split='test',
        subset_list=None,    # None: run all 5 levels; or a list like ['level 1'] or ['level 2', 'level 4'] to filter
        open_prompt_template=OPEN_PROMPT_TEMPLATE,
        single_answer_cot_template=SINGLE_ANSWER_COT_TEMPLATE,
        reader_cfg=mathvision_reader_cfg,
        infer_cfg=mathvision_infer_cfg,
        eval_cfg=mathvision_eval_cfg
    )
]
