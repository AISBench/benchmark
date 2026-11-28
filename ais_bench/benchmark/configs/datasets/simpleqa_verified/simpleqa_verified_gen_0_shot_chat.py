from ais_bench.benchmark.datasets.simpleqa_verified import (
    SimpleQAVerifiedDataset,
    SimpleQAVerifiedEvaluator,
    SimpleQAVerifiedJudgeDataset,
)
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

simpleqa_verified_datasets = [
    {
        "abbr": "simpleqa-verified",
        "type": SimpleQAVerifiedDataset,
        "path": "ais_bench/datasets/simpleqa_verified",
        "reader_cfg": {"input_columns": ["problem"], "output_column": "answer"},
        "infer_cfg": {
            "prompt_template": {
                "type": PromptTemplate,
                "template": {"round": [{"role": "HUMAN", "prompt": "{problem}"}]},
            },
            "retriever": {"type": ZeroRetriever},
            "inferencer": {"type": GenInferencer},
        },
        "judge_infer_cfg": {
            "judge_reader_cfg": {
                "input_columns": ["judge_prompt"],
                "output_column": "model_pred_uuid",
            },
            "judge_model": {
                "attr": "service",
                "type": VLLMCustomAPIChat,
                "abbr": "gpt-4.1-judge",
                "path": "",
                "model": "gpt-4.1",
                "stream": False,
                "request_rate": 1,
                "use_timestamp": False,
                "retry": 1,
                "api_key": "",
                "host_ip": "localhost",
                "host_port": 8005,
                "url": "",
                "max_out_len": 16,
                "batch_size": 1,
                "trust_remote_code": False,
                "generation_kwargs": {"temperature": 0},
            },
            "judge_dataset_type": SimpleQAVerifiedJudgeDataset,
            "prompt_template": {
                "type": PromptTemplate,
                "template": {"round": [{"role": "HUMAN", "prompt": "{judge_prompt}"}]},
            },
            "retriever": {"type": ZeroRetriever},
            "inferencer": {"type": GenInferencer},
        },
        "eval_cfg": {"evaluator": {"type": SimpleQAVerifiedEvaluator}},
    }
]
