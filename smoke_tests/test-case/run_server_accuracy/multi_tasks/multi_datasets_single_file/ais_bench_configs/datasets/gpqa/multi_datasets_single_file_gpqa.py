from mmengine import read_base

with read_base():
    from .gpqa_gen_0_shot_str import gpqa_datasets

gpqa_datasets[0]['reader_cfg']['test_range'] = '[0:10]'