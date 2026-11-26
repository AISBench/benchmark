from mmengine import read_base

with read_base():
    from .vllm_api_general_stream import models

models[0]['model'] = "qwen"
models[0]['request_rate'] = 50
models[0]['max_out_len'] = 200
models[0]['batch_size'] = 200
models[0]['generation_kwargs'] = dict(ignore_eos=True)

# Runtime import also works with mmengine's lazy config loader.
if __import__('os').environ.get('AISBENCH_SMOKE_FAST') == '1':
    models[0]['model'] = 'aisbench-smoke-functional/qwen'
    models[0]['max_out_len'] = min(models[0]['max_out_len'], 32)
