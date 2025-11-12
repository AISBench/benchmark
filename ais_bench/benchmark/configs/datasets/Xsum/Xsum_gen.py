from mmengine.config import read_base

with read_base():
    from .Xsum_gen_0_shot_chat import Xsum_datasets  # noqa: F401, F403
