from mmengine import read_base

with read_base():
    from .quality.leval_quality_gen import LEval_quality_datasets

for dataset in [
    *LEval_quality_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
