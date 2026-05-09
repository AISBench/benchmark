from mmengine import read_base

with read_base():
    from .govreportsumm.leval_gov_report_summ_gen import LEval_govreport_summ_datasets

for dataset in [
    *LEval_govreport_summ_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
