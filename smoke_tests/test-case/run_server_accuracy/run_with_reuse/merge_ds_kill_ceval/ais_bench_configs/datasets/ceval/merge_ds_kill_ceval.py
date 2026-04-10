from mmengine import read_base

with read_base():
    from .ceval_gen_0_shot_noncot_chat_prompt import ceval_datasets
    
ceval_datasets = ceval_datasets[:3]

for ceval_dataset in ceval_datasets:
    ceval_dataset['reader_cfg']['test_range'] = '[0:10]'