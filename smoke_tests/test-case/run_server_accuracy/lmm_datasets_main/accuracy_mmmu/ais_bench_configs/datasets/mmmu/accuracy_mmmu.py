from mmengine.config import read_base

with read_base():
    from .mmmu_gen import mmmu_datasets  # noqa: F401, F403

mmmu_datasets[0]['reader_cfg']['test_range'] = '[0:10]'