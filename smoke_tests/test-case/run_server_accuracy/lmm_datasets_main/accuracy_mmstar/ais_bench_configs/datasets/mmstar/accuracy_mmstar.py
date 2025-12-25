from mmengine.config import read_base

with read_base():
    from .mmstar_gen import mmstar_datasets  # noqa: F401, F403

mmstar_datasets[0]['reader_cfg']['test_range'] = '[0:10]'