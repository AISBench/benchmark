from mmengine import read_base

with read_base():
    from .ceval_gen_5_shot_str import ceval_datasets # noqa: F401, F403

ceval_datasets[0]['reader_cfg']['test_range'] = '[0:10]'