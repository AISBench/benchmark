import random
from mmengine import read_base

with read_base():
    from .mmlu_gen_0_shot_cot_chat_prompt import mmlu_datasets
mmlu_datasets=mmlu_datasets[:3]
for i in mmlu_datasets:
    i['reader_cfg']['test_range'] = '[0:5]'