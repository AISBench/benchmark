import random
from mmengine.config import read_base

with read_base():
    from .mmlu_gen_5_shot_str import mmlu_datasets  # noqa: F401, F403

for i in mmlu_datasets:
    i['reader_cfg']['test_range'] = '[0:10]'

mmlu_datasets = random.sample(mmlu_datasets, 3)