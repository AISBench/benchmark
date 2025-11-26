from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import SyntheticDataset, MATHEvaluator, math_postprocess_v2
from mmengine.config import read_base

with read_base():
    from ais_bench.datasets.synthetic.synthetic_config import synthetic_config

synthetic_reader_cfg = dict(
    input_columns=['question', 'max_out_len'],
    output_column='answer'
)


synthetic_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{question}"
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

synthetic_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2)
)


synthetic_datasets = [
    dict(
        abbr='synthetic',
        type=SyntheticDataset,
        config=synthetic_config,
        reader_cfg=synthetic_reader_cfg,
        infer_cfg=synthetic_infer_cfg,
        eval_cfg=synthetic_eval_cfg
    )
]
