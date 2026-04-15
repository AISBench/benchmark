from mmengine.config import read_base

with read_base():
    from .textvqa_gen import textvqa_datasets  # noqa: F401, F403

textvqa_datasets[0]['reader_cfg']['test_range'] = '[0:10]'