from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MMMUDataset, MMMUEvaluator


START_TEXT_PROMPT = "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Think step by step before answering.\n\n"
END_TEXT_PROMPT = ""
OPTIONS_PROMPT = ""

mmmu_reader_cfg = dict(
    input_columns=['question', 'image'],
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
        path='ais_bench/datasets/mmmu/MMMU_DEV_VAL.tsv', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        start_text_prompt=START_TEXT_PROMPT,
        end_text_prompt=END_TEXT_PROMPT,
        option_prompt=OPTIONS_PROMPT,
        reader_cfg=mmmu_reader_cfg,
        infer_cfg=mmmu_infer_cfg,
        eval_cfg=mmmu_eval_cfg
    )
]
