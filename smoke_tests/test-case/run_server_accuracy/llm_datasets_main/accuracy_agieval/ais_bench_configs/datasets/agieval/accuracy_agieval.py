from mmengine import read_base

with read_base():
    from .agieval_gen_0_shot_chat_prompt import agieval_datasets

# 冒烟：仅取一个子集、小样本
agieval_datasets = agieval_datasets[:1]
agieval_datasets[0]['reader_cfg'] = dict(
    agieval_datasets[0]['reader_cfg'],
    test_range='[0:10]',
)
