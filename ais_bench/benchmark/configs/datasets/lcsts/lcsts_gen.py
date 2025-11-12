from mmengine.config import read_base

with read_base():
    from .lcsts_gen_0_shot_chat import lcsts_datasets  # noqa: F401, F403
