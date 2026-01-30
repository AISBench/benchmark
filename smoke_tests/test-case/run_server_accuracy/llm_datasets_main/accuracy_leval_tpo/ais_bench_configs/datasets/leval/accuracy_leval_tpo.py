from mmengine import read_base

with read_base():
    from .tpo.leval_tpo_gen import LEval_tpo_datasets

for dataset in [
    *LEval_tpo_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
