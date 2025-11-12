from mmengine.config import read_base

with read_base():
    from .lambada_gen_0_shot_chat import lambada_datasets  # noqa: F401, F403