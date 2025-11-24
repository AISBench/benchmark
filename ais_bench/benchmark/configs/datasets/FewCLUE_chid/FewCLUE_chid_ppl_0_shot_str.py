from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import PPLInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import CHIDDataset

chid_reader_cfg = dict(
    input_columns=[f'content{i}' for i in range(7)], output_column='answer')

chid_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={i: f'以下句子是否通顺？\n{{content{i}}}\n这个句子是通顺的。'
                  for i in range(7)}),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

chid_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

chid_datasets = [
    dict(
        type=CHIDDataset,
        path='json',
        abbr='chid-dev',
        data_files='ais_bench/datasets/FewCLUE/chid/dev_few_all.json',
        split='train',
        reader_cfg=chid_reader_cfg,
        infer_cfg=chid_infer_cfg,
        eval_cfg=chid_eval_cfg),
    dict(
        type=CHIDDataset,
        path='json',
        abbr='chid-test',
        data_files='ais_bench/datasets/FewCLUE/chid/test_public.json',
        split='train',
        reader_cfg=chid_reader_cfg,
        infer_cfg=chid_infer_cfg,
        eval_cfg=chid_eval_cfg),
]
