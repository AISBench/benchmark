from mmengine.config import read_base
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import FixKRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MMLUProDataset, MMLUProBaseEvaluator

with read_base():
    from .mmlu_pro_categories import categories

mmlu_pro_datasets = []

for category in categories:
    hint = f'Answer the following multiple choice question about {category}, and give your answer option directly.'
    question_and_options = 'Question:\n{question}\nOptions:\n{options_str}'
    mmlu_pro_reader_cfg = dict(
        input_columns=['question', 'cot_content', 'options_str'],
        output_column='answer_string',
        train_split='validation',
        test_split='test',
    )
    mmlu_pro_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=f'{question_and_options}\nAnswer: {{answer}}'),
        prompt_template=dict(
            type=PromptTemplate,
            template=f'{hint}\n</E>{question_and_options}\nAnswer: ',
            ice_token='</E>'
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
            inferencer=dict(type=GenInferencer)
    )

    mmlu_pro_eval_cfg = dict(
        evaluator=dict(type=MMLUProBaseEvaluator)
    )

    mmlu_pro_datasets.append(
        dict(
            abbr=f'mmlu_pro_{category.replace(" ", "_")}',
            type=MMLUProDataset,
            path='ais_bench/datasets/mmlu_pro',  # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
            category=category,
            reader_cfg=mmlu_pro_reader_cfg,
            infer_cfg=mmlu_pro_infer_cfg,
            eval_cfg=mmlu_pro_eval_cfg,
        ))
