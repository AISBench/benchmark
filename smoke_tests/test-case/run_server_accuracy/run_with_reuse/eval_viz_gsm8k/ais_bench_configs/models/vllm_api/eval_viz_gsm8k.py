from mmengine import read_base

with read_base():
    from .vllm_api_general_chat import models

models[0]['model'] = "qwen"
models[0]['request_rate'] = 0
models[0]['max_out_len'] = 100
models[0]['batch_size'] = 100
models[0]['generation_kwargs'] = dict(ignore_eos=True)