from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import PPLInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import HFDataset

eprstmt_reader_cfg = dict(
    input_columns=['sentence'], output_column='label', test_split='train')

eprstmt_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            'Negative':
            dict(round=[
                dict(role='HUMAN', prompt='内容： "{sentence}"。情绪分类：'),
                dict(role='BOT', prompt='消极。')
            ]),
            'Positive':
            dict(round=[
                dict(role='HUMAN', prompt='内容： "{sentence}"。情绪分类：'),
                dict(role='BOT', prompt='积极。')
            ]),
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

eprstmt_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

eprstmt_datasets = [
    dict(
        type=HFDataset,
        abbr='eprstmt-dev',
        path='json',
        data_files='ais_bench/datasets/FewCLUE/eprstmt/dev_few_all.json',
        split='train',
        reader_cfg=eprstmt_reader_cfg,
        infer_cfg=eprstmt_infer_cfg,
        eval_cfg=eprstmt_eval_cfg),
    dict(
        type=HFDataset,
        abbr='eprstmt-test',
        path='json',
        data_files='ais_bench/datasets/FewCLUE/eprstmt/test_public.json',
        split='train',
        reader_cfg=eprstmt_reader_cfg,
        infer_cfg=eprstmt_infer_cfg,
        eval_cfg=eprstmt_eval_cfg)
]
