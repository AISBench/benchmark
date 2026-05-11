from mmengine.config import read_base

with read_base():
    from .refcoco_gen import refcoco_datasets  # noqa: F401, F403

refcoco_datasets = [
    dataset for dataset in refcoco_datasets if dataset['abbr'] == 'RefCOCO_val'
]
refcoco_datasets[0]['reader_cfg']['test_range'] = '[0:10]'