import copy
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.sharegpt.sharegpt_gen import sharegpt_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as base_vllm_api_stream_chat

# Multi-Rate Performance Evaluation:
# Send the ShareGPT dataset to the service at request_rate=1, 2, 4, 8 (QPS) respectively.
# In AISBench, `request_rate` is a field of the model configuration, so build one model
# configuration per rate via `copy.deepcopy` and combine them with a single dataset.
datasets = datasets

models = []
for rate in [1, 2, 4, 8]:
    model_cfg = copy.deepcopy(base_vllm_api_stream_chat[0])
    model_cfg["abbr"] = f"vllm-api-stream-chat-rate-{rate}"
    model_cfg["host_ip"] = "localhost"
    model_cfg["host_port"] = 8080
    model_cfg["max_out_len"] = 1024
    model_cfg["batch_size"] = 50
    model_cfg["request_rate"] = rate
    model_cfg["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)
    models.append(model_cfg)

work_dir = "outputs/default/"
