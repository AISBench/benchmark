from mmengine import read_base

with read_base():
    from .math_prm800k_500_0shot_cot_gen import math_datasets

# 冒烟：小样本
math_datasets[0]['reader_cfg'] = dict(
    math_datasets[0]['reader_cfg'],
    test_range='[0:10]',
)
