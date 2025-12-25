import random
from mmengine import read_base

with read_base():
    from . ceval_gen_0_shot_cot_chat_prompt import ceval_datasets
ceval_datasets=ceval_datasets[:3]
for i in ceval_datasets:
    i['reader_cfg']['test_range'] = '[0:5]'