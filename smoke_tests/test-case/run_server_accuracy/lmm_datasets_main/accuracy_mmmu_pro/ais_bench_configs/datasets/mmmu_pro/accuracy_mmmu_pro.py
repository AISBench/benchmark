from mmengine.config import read_base

with read_base():
    from .mmmu_pro_options10_gen import mmmu_pro_datasets  # noqa: F401, F403

mmmu_pro_datasets[0]['reader_cfg']['test_range'] = '[0:10]'