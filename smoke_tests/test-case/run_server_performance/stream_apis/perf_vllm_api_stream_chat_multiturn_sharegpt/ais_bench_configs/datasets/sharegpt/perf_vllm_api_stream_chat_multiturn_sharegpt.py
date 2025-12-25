from mmengine import read_base

with read_base():
    from .sharegpt_gen import sharegpt_datasets

sharegpt_datasets[0]['path'] = 'ais_bench/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json'
sharegpt_datasets[0]['reader_cfg']['test_range'] = '[0:3]'