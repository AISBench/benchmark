from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_str import gsm8k_datasets
    from ais_bench.benchmark.configs.datasets.aime2024.aime2024_gen_0_shot_chat_prompt import aime2024_datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

datasets = gsm8k_datasets + aime2024_datasets

models = vllm_api_general_chat + vllm_api_stream_chat
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8080
models[1]["host_ip"] = "localhost"
models[1]["host_port"] = 8081
models[0]["max_out_len"] = 512
models[0]["batch_size"] = 1
models[1]["max_out_len"] = 512
models[1]["batch_size"] = 1

work_dir = "outputs/default/"

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLApiInferTask),
    ),
)
