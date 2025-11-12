from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import lambadaDataset, LambadaEvaluator

lambada_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='label',
    train_split='test',
    test_split='test')

lambada_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role='HUMAN',
                prompt='Please complete the following sentence:\n{prompt}')
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

lambada_eval_cfg = dict(evaluator=dict(type=LambadaEvaluator))

lambada_datasets = [
    dict(
        abbr='lambada',
        type=lambadaDataset,
        path='ais_bench/datasets/lambada',
        reader_cfg=lambada_reader_cfg,
        infer_cfg=lambada_infer_cfg,
        eval_cfg=lambada_eval_cfg)
]