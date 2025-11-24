from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import PPLInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import CslDataset

csl_reader_cfg = dict(
    input_columns=['abst', 'keywords'], output_column='label')

csl_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            0: '摘要：{abst}',
            1: '摘要：{abst}\n关键词：{keywords}'
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

csl_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

# 841b62
csl_datasets = [
    dict(
        type=CslDataset,
        path='json',
        abbr='csl_dev',
        data_files='ais_bench/datasets/FewCLUE/csl/dev_few_all.json',
        split='train',
        reader_cfg=csl_reader_cfg,
        infer_cfg=csl_infer_cfg,
        eval_cfg=csl_eval_cfg),
    dict(
        type=CslDataset,
        path='json',
        abbr='csl_test',
        data_files='ais_bench/datasets/FewCLUE/csl/test_public.json',
        split='train',
        reader_cfg=csl_reader_cfg,
        infer_cfg=csl_infer_cfg,
        eval_cfg=csl_eval_cfg)
]
