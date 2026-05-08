from mmengine.config import read_base

with read_base():
    from .refcocog_gen import refcocog_datasets  # noqa: F401, F403

refcocog_datasets = [
    dataset for dataset in refcocog_datasets if dataset['abbr'] == 'RefCOCOg_val'
]
refcocog_datasets[0]['reader_cfg']['test_range'] = '[0:10]'