from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_str import gsm8k_datasets
    from ais_bench.benchmark.configs.datasets.aime2024.aime2024_gen_0_shot_chat_prompt import aime2024_datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

# Multi-Dataset Performance Evaluation:
# Specify multiple datasets and send them to the same service for performance evaluation.
datasets = gsm8k_datasets + aime2024_datasets

models = vllm_api_stream_chat
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8080
models[0]["max_out_len"] = 512
models[0]["batch_size"] = 1
models[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)

work_dir = "outputs/default/"

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLApiInferTask),
    ),
)
