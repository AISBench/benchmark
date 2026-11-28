from mmengine import read_base

with read_base():
    from .ceval_gen_5_shot_str import ceval_datasets # noqa: F401, F403

ceval_datasets[0]['reader_cfg']['test_range'] = '[0:10]'

# Fast smoke: performance results are converted (and every chart regenerated)
# once per dataset, so the full 52-subject sweep costs minutes of plotting.
if __import__('os').environ.get('AISBENCH_SMOKE_FAST') == '1':
    ceval_datasets = ceval_datasets[:3]
    for _dataset in ceval_datasets:
        _dataset['reader_cfg']['test_range'] = '[0:20]'
    del _dataset
