from ais_bench.benchmark.datasets import SWEBenchProDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import SWEBenchProInferTask, SWEBenchProEvalTask
from ais_bench.benchmark.summarizers import SWEBenchProSummarizer

STEP_LIMIT = 250

models = [
    dict(
        attr="local",
        abbr="swebench_pro_full_module",
        type="LiteLLMChat",
        model="",
        api_key="EMPTY",
        url="http://127.0.0.1:8000/v1",
        batch_size=1,
        generation_kwargs=dict(),
    )
]

SWEBP_SCRIPT_PATH_ABS = ""
SWEBP_DOCKER_PATH_ABS = ""

datasets = [
    dict(
        type=SWEBenchProDataset,
        abbr="swebench_pro_full_data",
        path="",
        name="full",
        split="test",
        step_limit=STEP_LIMIT,
        filter_spec="",
        shuffle=False,
        swebp_scripts_dir= SWEBP_SCRIPT_PATH_ABS,
        swebp_docker_dir= SWEBP_DOCKER_PATH_ABS,
    ),
]

summarizer = dict(
    attr="accuracy",
    type=SWEBenchProSummarizer,
)

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=SWEBenchProInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=SWEBenchProEvalTask),
    ),
)