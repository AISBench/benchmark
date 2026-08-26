from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

models = vllm_api_stream_chat
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
