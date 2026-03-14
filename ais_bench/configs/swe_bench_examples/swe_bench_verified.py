from ais_bench.benchmark.datasets import SWEBenchDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import SWEBenchInferTask, SWEBenchEvalTask
from ais_bench.benchmark.summarizers import DefaultSummarizer

models = [
    dict(
        attr="local",
        abbr="swebench",
        type="LiteLLMChat",
        model="",
        api_key="",
        url="",
        batch_size=1,
        generation_kwargs=dict(),
    )
]

datasets = [
    dict(
        type=SWEBenchDataset,
        abbr="swebench_verified",
        path="ais_bench/datasets/SWE-bench_Verified",
        name="verified",
        split="test",
        filter_spec="",
        shuffle=False,
        prediction_file_extension="jsonl",
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
