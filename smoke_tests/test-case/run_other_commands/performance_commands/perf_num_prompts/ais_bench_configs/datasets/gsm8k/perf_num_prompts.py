from mmengine import read_base

with read_base():
    from .gsm8k_gen_0_shot_cot_chat_prompt import gsm8k_datasets

# gsm8k_datasets[0]['reader_cfg']['test_range'] = '[0:100]'