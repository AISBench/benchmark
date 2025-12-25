import random
from mmengine.config import read_base

with read_base():
    from .ceval_gen_5_shot_str import ceval_datasets # noqa: F401, F403

for i in ceval_datasets:
    i['reader_cfg']['test_range'] = '[0:10]'
 
ceval_datasets = random.sample(ceval_datasets, 3)