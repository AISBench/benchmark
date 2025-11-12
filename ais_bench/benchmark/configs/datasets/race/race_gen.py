from mmengine.config import read_base

with read_base():
    from .race_high_5_shot_cot_chat import race_datasets  # noqa: F401, F403