from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import (
    LCBCodeGenerationDataset,
    LCBCodeExecutionDataset,
    LCBTestOutputPredictionDataset,
    LCBCodeGenerationEvaluator,
    LCBCodeExecutionEvaluator,
    LCBTestOutputEvaluator
)
from ais_bench.benchmark.datasets.livecodebench import TestOutputPromptConstants

DATASET_RELEASE_VERSION = 'v4_v5' ## same with DeepSeek-R1 Evaluation: LiveCodeBench (Jain et al., 2024) (2024-08 – 2025-01)

lcb_code_generation_reader_cfg = dict(
    input_columns=[
        'question_content',
        'format_prompt',
    ],
    output_column='question_id',
)

SYSTEM_MESSAGE_GENERIC = f'You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.'

prompt_template = '### Question:\n{question_content}\n\n{format_prompt}' + \
                    '### Answer: (use the provided format with backticks)\n\n'


# Code Generation Tasks
lcb_code_generation_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt=prompt_template
                )
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

lcb_code_generation_eval_cfg = dict(
    evaluator=dict(
        type=LCBCodeGenerationEvaluator,
        num_process_evaluate=4,
        timeout=500,
        release_version=DATASET_RELEASE_VERSION,
    ),
    pred_role='BOT',
)

LCBCodeGeneration_dataset = dict(
    type=LCBCodeGenerationDataset,
    abbr='lcb_code_generation_v4_v5',
    path='ais_bench/datasets/code_generation_lite', # https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/main
    release_version=DATASET_RELEASE_VERSION,
    reader_cfg=lcb_code_generation_reader_cfg,
    infer_cfg=lcb_code_generation_infer_cfg,
    eval_cfg=lcb_code_generation_eval_cfg
)

LCB_datasets = [
    LCBCodeGeneration_dataset,
]
