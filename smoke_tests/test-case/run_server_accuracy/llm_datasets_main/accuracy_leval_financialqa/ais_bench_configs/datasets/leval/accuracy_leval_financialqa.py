from mmengine import read_base

with read_base():
    from .financialqa.leval_financial_qa_gen import LEval_financialqa_datasets

for dataset in [
    *LEval_financialqa_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
