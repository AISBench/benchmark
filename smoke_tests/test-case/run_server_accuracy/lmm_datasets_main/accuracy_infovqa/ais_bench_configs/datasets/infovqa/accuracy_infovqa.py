from mmengine.config import read_base

with read_base():
    from .infovqa_gen import infovqa_datasets  # noqa: F401, F403

infovqa_datasets[0]['reader_cfg']['test_range'] = '[0:10]'