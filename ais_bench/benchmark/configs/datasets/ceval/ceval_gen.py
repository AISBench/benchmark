from mmengine.config import read_base

with read_base():
    from .ceval_gen_5_shot_str import ceval_datasets # noqa: F401, F403