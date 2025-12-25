from mmengine import read_base

with read_base():
    from .triton_stream_api_general import models

models[0]['model_name'] = "qwen"
models[0]['max_out_len'] = 20
models[0]['batch_size'] = 20