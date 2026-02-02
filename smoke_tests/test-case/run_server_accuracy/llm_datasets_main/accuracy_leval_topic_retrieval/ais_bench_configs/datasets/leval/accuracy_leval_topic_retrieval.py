from mmengine import read_base

with read_base():
    from .topicretrievallongchat.leval_topic_retrieval_gen import LEval_tr_datasets

for dataset in [
    *LEval_tr_datasets,
]:
    dataset['reader_cfg']['test_range'] = '[0:10]'
