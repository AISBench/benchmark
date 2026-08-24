from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import synthetic_datasets as base_synthetic_datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

# Custom Sequence Multi-Task Combinations:
# Build multiple synthetic sub-datasets with different input/output lengths, then use
# `model_dataset_combinations` to precisely pair models with partial datasets.
datasets = []
for input_len in [256, 512]:
    for output_len in [256, 512]:
        ds = dict(base_synthetic_datasets[0])
        ds["abbr"] = f"syn_in{input_len}_out{output_len}"
        ds["config"] = {
            "Type": "string",
            "RequestCount": 100,
            "TrustRemoteCode": False,
            "StringConfig": {
                "Input": {
                    "Method": "uniform",
                    "Params": {"MinValue": input_len, "MaxValue": input_len},
                },
                "Output": {
                    "Method": "uniform",
                    "Params": {"MinValue": output_len, "MaxValue": output_len},
                },
            },
        }
        datasets.append(ds)

models = vllm_api_stream_chat
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8080
models[0]["max_out_len"] = 512
models[0]["batch_size"] = 1
models[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)

# Key: Only specify partial models for partial datasets
model_dataset_combinations = [
    dict(models=[models[0]], datasets=[datasets[0], datasets[1]]),
    dict(models=[models[0]], datasets=[datasets[2]]),
]

work_dir = "outputs/default/"

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLApiInferTask),
    ),
)
