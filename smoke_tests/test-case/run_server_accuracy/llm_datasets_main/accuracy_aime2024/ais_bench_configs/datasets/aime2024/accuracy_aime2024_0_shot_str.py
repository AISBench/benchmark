from mmengine import read_base

with read_base():
    from .aime2024_gen_0_shot_str import aime2024_datasets

aime2024_datasets[0]['reader_cfg']['test_range'] = '[0:10]'