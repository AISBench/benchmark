from mmengine import read_base

with read_base():
    from .newssumm.leval_news_summ_gen import LEval_newssumm_datasets

for dataset in [
    *LEval_newssumm_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
