from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream_session import models
    from ais_bench.benchmark.configs.datasets.custom.custom_agent_session_gen import agent_demo_datasets

datasets = [*agent_demo_datasets]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,                          # session 模式强制单进程，这里也置 1
        task=dict(type=OpenICLApiInferTask),
    ),
)

work_dir = 'outputs/api-vllm-stream-session/'
