from mmengine import read_base

with read_base():
    from .aime2026_gen_0_shot_str import aime2026_datasets

aime2026_datasets[0]['abbr'] = 'aime2026_0_shot_str'
aime2026_datasets[0]['reader_cfg']['test_range'] = '[0:10]'

