from mmengine import read_base

with read_base():
    from .paperassistant.leval_paper_assistant_gen import LEval_ps_summ_datasets

for dataset in [
    *LEval_ps_summ_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
