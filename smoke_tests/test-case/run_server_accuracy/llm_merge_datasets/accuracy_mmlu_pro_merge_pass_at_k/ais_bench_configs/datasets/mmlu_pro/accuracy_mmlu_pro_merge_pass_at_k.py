import random
from mmengine.config import read_base

with read_base():
    from .mmlu_pro_gen_5_shot_str import mmlu_pro_datasets  # noqa: F401, F403

for i in mmlu_pro_datasets:
    i['reader_cfg']['test_range'] = '[0:3]'

mmlu_pro_datasets = random.sample(mmlu_pro_datasets, 3)