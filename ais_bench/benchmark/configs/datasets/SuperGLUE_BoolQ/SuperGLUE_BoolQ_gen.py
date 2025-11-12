from mmengine.config import read_base

with read_base():
    from .SuperGLUE_BoolQ_gen_0_shot_str import BoolQ_datasets  # noqa: F401, F403
