from mmengine import read_base

with read_base():
    from .aime2025_gen_0_shot_llmjudge import aime2025_datasets

aime2025_datasets[0]['abbr'] = 'aime2025_judge_1'
aime2025_datasets[0]['judge_infer_cfg']['judge_model']['abbr'] = "first"
aime2025_datasets[0]['judge_infer_cfg']['judge_model']['model'] = "not_empty"

