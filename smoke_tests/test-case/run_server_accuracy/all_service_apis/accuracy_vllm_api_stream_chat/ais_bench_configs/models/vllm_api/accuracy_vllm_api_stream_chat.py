from mmengine import read_base

with read_base():
    from .vllm_api_stream_chat import models

models[0]['model'] = "qwen"
models[0]['max_out_len'] = 20
models[0]['batch_size'] = 20