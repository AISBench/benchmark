from mmengine.config import read_base

with read_base():
    from .docvqa_gen import docvqa_datasets  # noqa: F401, F403

docvqa_datasets[0]['reader_cfg']['test_range'] = '[0:10]'