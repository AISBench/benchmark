from mmengine import read_base

with read_base():
    from .codeu.leval_code_u_gen import LEval_code_u_datasets

for dataset in [
    *LEval_code_u_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
