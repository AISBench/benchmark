from mmengine.config import read_base

with read_base():
    from .siqa_gen_0_shot_chat import siqa_datasets  # noqa: F401, F403
