from mmengine import read_base

with read_base():
    from .multidocqa.leval_multidoc_qa_gen import LEval_multidocqa_datasets

for dataset in [
    *LEval_multidocqa_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
