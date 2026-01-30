from mmengine import read_base

with read_base():
    from .narrativeqa.leval_narrative_qa_gen import LEval_narrativeqa_datasets

for dataset in [
    *LEval_narrativeqa_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
