from mmengine import read_base

with read_base():
    from .meetingsumm.leval_meeting_summ_gen import LEval_meetingsumm_datasets

for dataset in [
    *LEval_meetingsumm_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
