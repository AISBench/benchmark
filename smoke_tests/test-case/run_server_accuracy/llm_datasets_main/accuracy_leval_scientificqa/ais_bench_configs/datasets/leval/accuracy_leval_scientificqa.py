from mmengine import read_base

with read_base():
    from .scientificqa.leval_scientific_qa_gen import LEval_scientificqa_datasets

for dataset in [
    *LEval_scientificqa_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
