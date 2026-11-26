from mmengine import read_base

with read_base():
    from .gsm8k_gen_4_shot_cot_str import gsm8k_datasets

# Runtime import also works with mmengine's lazy config loader.
_smoke_fast = __import__('os').environ.get('AISBENCH_SMOKE_FAST') == '1'
gsm8k_datasets[0]['reader_cfg']['test_range'] = '[0:20]' if _smoke_fast else '[0:100]'