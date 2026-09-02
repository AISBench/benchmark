from mmengine.config import read_base

with read_base():
    from .refcoco_plus_gen import refcoco_plus_datasets  # noqa: F401, F403

refcoco_plus_datasets = [
    dataset for dataset in refcoco_plus_datasets if dataset['abbr'] == 'RefCOCOPlus_val'
]
refcoco_plus_datasets[0]['reader_cfg']['test_range'] = '[0:10]'