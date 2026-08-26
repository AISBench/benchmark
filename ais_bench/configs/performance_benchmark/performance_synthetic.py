from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import synthetic_datasets as base_synthetic_datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

# Synthetic Dataset Multi-Task Combinations:
# Define multiple synthetic sub-datasets with different input/output lengths via the
# `config` field of `SyntheticDataset`, and combine them with the same model.
datasets = []
for input_len in [256, 512, 1024]:
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
models[0]["request_rate"] = 2
models[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)

work_dir = "outputs/default/"
