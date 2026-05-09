from ais_bench.benchmark.datasets import SWEBenchDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import SWEBenchInferTask, SWEBenchEvalTask
from ais_bench.benchmark.summarizers import SWEBenchSummarizer

STEP_LIMIT = 200

models = [
    dict(
        attr="local",
        abbr="swebench",
        type="LiteLLMChat",
        model="",
        api_key="EMPTY",
        url="http://127.0.0.1:8000/v1",  #API base, e.g. http://127.0.0.1:8000/v1
        batch_size=1,
        generation_kwargs=dict(),
    )
]

datasets = [
    dict(
        type=SWEBenchDataset,
        abbr="swebench_full",
        # Relative to AIS_BENCH_DATASETS_CACHE (default: project root); missing dir -> HF snapshot_download
        path="",
        name="full",
        split="test",
        step_limit=STEP_LIMIT,
        filter_spec="",
        shuffle=False,
    ),
]

summarizer = dict(
    attr="accuracy",
    type=SWEBenchSummarizer,
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
