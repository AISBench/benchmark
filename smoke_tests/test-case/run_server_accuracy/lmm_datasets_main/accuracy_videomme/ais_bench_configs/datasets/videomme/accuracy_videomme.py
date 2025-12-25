from mmengine.config import read_base

with read_base():
    from .videomme_gen import videomme_datasets  # noqa: F401, F403

videomme_datasets[0]['reader_cfg']['test_range'] = '[0:10]'