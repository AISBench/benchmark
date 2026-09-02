from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.sharegpt.sharegpt_gen import sharegpt_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import models as vllm_api_general_stream

# Multi-Model Performance Evaluation:
# Evaluate multiple models on the same dataset simultaneously for performance comparison.
datasets = datasets

# Rename the abbr of each model so that the results are distinguishable
vllm_api_stream_chat[0]["abbr"] = "vllm-qwen2.5-7b"
vllm_api_general_stream[0]["abbr"] = "vllm-qwen2.5-14b"

vllm_api_stream_chat[0]["host_ip"] = "localhost"
vllm_api_stream_chat[0]["host_port"] = 8080
vllm_api_stream_chat[0]["max_out_len"] = 1024
vllm_api_stream_chat[0]["batch_size"] = 50
vllm_api_stream_chat[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)

vllm_api_general_stream[0]["host_ip"] = "localhost"
vllm_api_general_stream[0]["host_port"] = 8081
vllm_api_general_stream[0]["max_out_len"] = 1024
vllm_api_general_stream[0]["batch_size"] = 50
vllm_api_general_stream[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)

models = vllm_api_stream_chat + vllm_api_general_stream

work_dir = "outputs/default/"
