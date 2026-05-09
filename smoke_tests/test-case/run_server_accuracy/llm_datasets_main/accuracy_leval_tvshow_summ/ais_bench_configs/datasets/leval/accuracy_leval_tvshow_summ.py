from mmengine import read_base

with read_base():
    from .tvshowsumm.leval_tv_show_summ_gen import LEval_tvshow_summ_datasets

for dataset in [
    *LEval_tvshow_summ_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
