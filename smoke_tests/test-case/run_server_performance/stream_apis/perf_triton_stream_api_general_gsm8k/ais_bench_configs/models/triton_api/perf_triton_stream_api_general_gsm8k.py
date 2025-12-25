from mmengine import read_base

with read_base():
    from .triton_stream_api_general import models

models[0]['model_name'] = 'qwen'
models[0]['request_rate'] = 50
models[0]['max_out_len'] = 200
models[0]['batch_size'] = 200
models[0]['generation_kwargs'] = dict(ignore_eos=True)