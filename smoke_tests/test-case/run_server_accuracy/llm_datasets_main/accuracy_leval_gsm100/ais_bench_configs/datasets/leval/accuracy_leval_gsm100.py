from mmengine import read_base

with read_base():
    from .gsm100.leval_gsm100_gen import LEval_gsm100_datasets

for dataset in [
    *LEval_gsm100_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
