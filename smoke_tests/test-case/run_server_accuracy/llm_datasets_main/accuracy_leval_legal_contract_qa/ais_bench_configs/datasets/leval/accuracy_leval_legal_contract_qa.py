from mmengine import read_base

with read_base():
    from .legalcontractqa.leval_legal_contract_qa_gen import LEval_legalqa_datasets

for dataset in [
    *LEval_legalqa_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
