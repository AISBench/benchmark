from ais_bench.benchmark.datasets import SWEBenchDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import SWEBenchInferTask, SWEBenchEvalTask
from ais_bench.benchmark.summarizers import DefaultSummarizer

STEP_LIMIT = 2

# For local vLLM: set model (e.g. hosted_vllm/qwen3), url (vLLM API base), api_key (e.g. "EMPTY").
# Example matching: mini-extra swebench -m hosted_vllm/qwen3 -c model.model_kwargs.api_base='"http://127.0.0.1:2998/v1"' ...
models = [
    dict(
        attr="local",
        abbr="swebench",
        type="LiteLLMChat",
        model="qwen3",  # e.g. hosted_vllm/qwen3 for local vLLM
        api_key="EMPTY",
        url="http://127.0.0.1:2998/v1",  # vLLM API base
        batch_size=2,
        generation_kwargs=dict(),
    )
]

datasets = [
    dict(
        type=SWEBenchDataset,
        abbr="swebench_lite",
        path="/data/zhanggaohua/datasets/SWE-bench_Lite",
        name="lite",
        split="test",
        filter_spec="",
        shuffle=False,
        step_limit=STEP_LIMIT,
    ),
]

summarizer = dict(
    attr="accuracy",
    type=DefaultSummarizer,
)


infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=SWEBenchInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=SWEBenchEvalTask),
    ),
)
