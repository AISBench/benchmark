from mmengine import read_base

with read_base():
    from .naturalquestion.leval_natural_question_gen import LEval_nq_datasets

for dataset in [
    *LEval_nq_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
