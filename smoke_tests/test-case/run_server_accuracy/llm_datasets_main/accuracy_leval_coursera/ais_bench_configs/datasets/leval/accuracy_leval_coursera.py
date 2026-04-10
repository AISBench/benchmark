from mmengine import read_base

with read_base():
    from .coursera.leval_coursera_gen import LEval_coursera_datasets

for dataset in [
    *LEval_coursera_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
