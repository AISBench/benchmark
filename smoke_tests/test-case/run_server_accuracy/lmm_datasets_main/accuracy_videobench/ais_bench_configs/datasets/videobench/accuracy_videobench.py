from mmengine.config import read_base

with read_base():
    from .videobench_gen import videobench_datasets  # noqa: F401, F403

videobench_datasets[0]['reader_cfg']['test_range'] = '[0:10]'