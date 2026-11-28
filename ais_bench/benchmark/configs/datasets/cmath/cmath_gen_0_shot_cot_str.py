from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import CMATHDataset, CmathEvaluator, cmath_postprocess

cmath_datasets = [dict(
    abbr='cmath-dev-str', type=CMATHDataset, path='ais_bench/datasets/cmath',
    reader_cfg=dict(input_columns=['question'], output_column='answer'),
    infer_cfg=dict(prompt_template=dict(type=PromptTemplate, template='请逐步解答下面的数学题，最后一行写“答案：数值”，不要重复题目。\n题目：{question}'),
                   retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer)),
    eval_cfg=dict(evaluator=dict(type=CmathEvaluator), pred_postprocessor=dict(type=cmath_postprocess)))
]
