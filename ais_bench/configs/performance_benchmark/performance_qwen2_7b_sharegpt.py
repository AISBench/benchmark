from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.sharegpt.sharegpt_gen import sharegpt_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

# Single-Task Performance Evaluation:
# Send the ShareGPT dataset to the service at request_rate=1 (QPS) for performance evaluation.
datasets = datasets

models = vllm_api_stream_chat
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8080
models[0]["max_out_len"] = 1024
models[0]["batch_size"] = 50
models[0]["request_rate"] = 1  # Request sending frequency: send 1 request to the server every 1/request_rate seconds; if less than 0.001, all requests are sent at once
models[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)  # When testing performance and needing to limit the output length, ignore_eos must be set to True

work_dir = "outputs/default/"
