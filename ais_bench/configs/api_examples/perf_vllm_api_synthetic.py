from mmengine.config import read_base
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import SyntheticDataset, MATHEvaluator, math_postprocess_v2

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import (
        models as vllm_api_general_stream,
    )

synthetic_config = {
    "Type": "string",
    "RequestCount": 100,
    "TrustRemoteCode": False,
    "StringConfig": {
        "Input": {
            "Method": "uniform",
            "Params": {"MinValue": 1, "MaxValue": 500}
        },
        "Output": {
            "Method": "gaussian",
            "Params": {"Mean": 200, "Var": 100, "MinValue": 1, "MaxValue": 500}
        }
    },
}

datasets = [
    dict(
        abbr='synthetic_custom',
        type=SyntheticDataset,
        config=synthetic_config,
        reader_cfg=dict(input_columns=['question', 'max_out_len'], output_column='answer'),
        infer_cfg=dict(
            prompt_template=dict(type=PromptTemplate, template="{question}"),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        eval_cfg=dict(
            evaluator=dict(type=MATHEvaluator, version='v2'),
            pred_postprocessor=dict(type=math_postprocess_v2),
        ),
    )
]

models = vllm_api_general_stream
work_dir = 'outputs/synthetic_perf_custom/'
