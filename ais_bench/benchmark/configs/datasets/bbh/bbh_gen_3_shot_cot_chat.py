from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import BBHDataset, BBHEvaluator, bbh_mcq_postprocess, BBHEvaluator_mcq
from mmengine.config import read_base

with read_base():
    from .bbh_multiple_choice_sets_lib_prompt import bbh_multiple_choice_sets_dict
    from .bbh_free_form_sets_lib_prompt import bbh_free_form_sets_dict

bbh_reader_cfg = dict(input_columns=['input'], output_column='target')

bbh_multiple_choice_sets = [
    'temporal_sequences',
    'disambiguation_qa',
    'date_understanding',
    'tracking_shuffled_objects_three_objects',
    'penguins_in_a_table',
    'geometric_shapes',
    'snarks',
    'ruin_names',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_five_objects',
    'logical_deduction_three_objects',
    'hyperbaton',
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects',
    'movie_recommendation',
    'salient_translation_error_detection',
    'reasoning_about_colored_objects',
]
bbh_free_form_sets = [
    'multistep_arithmetic_two',
    'navigate',
    'dyck_languages',
    'word_sorting',
    'sports_understanding',
    'boolean_expressions',
    'object_counting',
    'formal_fallacies',
    'causal_judgement',
    'web_of_lies',
]
dataset_path = "ais_bench/datasets/BBH/data" # 数据集路径

bbh_datasets = []
for _name in bbh_multiple_choice_sets:
    _hint = bbh_multiple_choice_sets_dict[_name]
    bbh_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f"Follow the given examples and answer the question.\n{_hint}\n\nQ: {{input}}\nA: Let's think step by step."
                )
            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))
    bbh_eval_cfg = dict(
        evaluator=dict(type=BBHEvaluator_mcq),
        pred_role='BOT',
        pred_postprocessor=dict(type=bbh_mcq_postprocess),
        dataset_postprocessor=dict(type=bbh_mcq_postprocess))

    bbh_datasets.append(
        dict(
            type=BBHDataset,
            path='ais_bench/datasets/BBH/data',
            name=_name,
            abbr='bbh-' + _name,
            reader_cfg=bbh_reader_cfg,
            infer_cfg=bbh_infer_cfg.copy(),
            eval_cfg=bbh_eval_cfg.copy()))

for _name in bbh_free_form_sets:
    _hint = bbh_free_form_sets_dict[_name]
    bbh_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                dict(
                    role='HUMAN',
                    prompt=
                    f"Follow the given examples and answer the question.\n{_hint}\n\nQ: {{input}}\nA: Let's think step by step."
                )
            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer))
    bbh_eval_cfg = dict(evaluator=dict(type=BBHEvaluator), pred_role='BOT')

    bbh_datasets.append(
        dict(
            type=BBHDataset,
            path=dataset_path,
            name=_name,
            abbr='bbh-' + _name,
            reader_cfg=bbh_reader_cfg,
            infer_cfg=bbh_infer_cfg.copy(),
            eval_cfg=bbh_eval_cfg.copy()))