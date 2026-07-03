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
        url="http://127.0.0.1:8000/v1",  # API base, e.g. http://127.0.0.1:8000/v1
        batch_size=1,
        generation_kwargs=dict(
            # Supports arbitrary generation parameters, consistent with regular model tasks.
            # Common parameters include temperature, top_p, top_k, timeout, etc.
            # temperature=0.0,   # Set 0 for deterministic output; omit or set >0 for diversity
            # top_p=1.0,
            # top_k=-1,
            # timeout=200,       # Inference timeout in seconds
        ),
    )
]

datasets = [
    dict(
        type=SWEBenchDataset,
        abbr="swebench_lite",
        # Relative to AIS_BENCH_DATASETS_CACHE (default: project root); missing -> HF download
        path="",
        name="lite",
        split="test",
        filter_spec="",
        shuffle=False,
        step_limit=STEP_LIMIT,
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
