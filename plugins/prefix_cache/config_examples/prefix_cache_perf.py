import os

from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

from ais_bench_prefix_cache.config import build_dataset_config, build_model_config


scenario = os.environ["AISBENCH_PREFIX_CACHE_SCENARIO"]
datasets = [build_dataset_config(scenario)]
models = [build_model_config(scenario)]
infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(type=LocalRunner, max_num_workers=1, task=dict(type=OpenICLApiInferTask)),
)
summarizer = dict(attr="accuracy", summary_groups=[])
work_dir = os.environ.get("AISBENCH_PREFIX_CACHE_WORK_DIR", "outputs/prefix_cache")

