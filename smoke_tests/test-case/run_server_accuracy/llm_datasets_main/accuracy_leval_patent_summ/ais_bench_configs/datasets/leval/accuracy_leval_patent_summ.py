from mmengine import read_base

with read_base():
    from .patentsumm.leval_patent_summ_gen import LEval_patent_summ_datasets

for dataset in [
    *LEval_patent_summ_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
