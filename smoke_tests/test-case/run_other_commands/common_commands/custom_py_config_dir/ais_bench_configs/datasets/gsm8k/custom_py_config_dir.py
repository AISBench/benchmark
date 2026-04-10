from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="Question: {question}\nLet's think step by step\nAnswer:"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

gsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_role='BOT',
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='ais_bench/datasets/gsm8k',  # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]

gsm8k_datasets[0]['reader_cfg']['test_range'] = '[0:10]'