from mmengine.config import read_base

with read_base():
    from .vocalsound_gen import vocalsound_datasets  # noqa: F401, F403

vocalsound_datasets[0]['reader_cfg']['test_range'] = '[0:10]'