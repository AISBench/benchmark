from mmengine import read_base

with read_base():
    from .scifi.leval_sci_fi_gen import LEval_sci_fi_datasets

for dataset in [
    *LEval_sci_fi_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
