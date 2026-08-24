from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_5_shot_str import ceval_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general

models = vllm_api_general
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8080

work_dir = "outputs/default/"

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLApiInferTask),
    ),
)
