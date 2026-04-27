from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MMMUDataset, MMMUEvaluator


MULT_CHOICE_PROMPT = (
    "Answer the following multiple choice question. The last line of your response should be of the following "
    "format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before "
    "answering.\n\n{question}\n\n{choices}"
)

OPEN_PROMPT = """
Solve the following problem step by step. The last line of your response should be of the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem.

{question}

Remember to put your answer on its own line at the end in the form "ANSWER: [ANSWER]" (without quotes) where [ANSWER] is the answer to the problem, and you do not need to use a \\boxed command.

"""

mmmu_reader_cfg = dict(
    input_columns=['content'],
    output_column='answer'
)

mmmu_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question}"},
                    "image": {"type": "image_url", "image_url": {"url": "file://{image}"}},
                })
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

mmmu_eval_cfg = dict(
    evaluator=dict(type=MMMUEvaluator)
)

mmmu_datasets = [
    dict(
        abbr='mmmu',
        type=MMMUDataset,
        path='ais_bench/datasets/mmmu',
        split='validation',
        mult_choice_prompt=MULT_CHOICE_PROMPT,
        open_prompt=OPEN_PROMPT,
        reader_cfg=mmmu_reader_cfg,
        infer_cfg=mmmu_infer_cfg,
        eval_cfg=mmmu_eval_cfg
    )
]
