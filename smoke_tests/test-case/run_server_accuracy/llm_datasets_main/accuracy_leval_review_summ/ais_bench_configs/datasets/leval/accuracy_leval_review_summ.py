from mmengine import read_base

with read_base():
    from .reviewsumm.leval_review_summ_gen import LEval_review_summ_datasets

for dataset in [
    *LEval_review_summ_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
