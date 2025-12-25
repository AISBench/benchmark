from mmengine import read_base

with read_base():
    from .BFCL_gen_multi_turn_base import bfcl_datasets
bfcl_datasets[0]['reader_cfg']['test_range'] = '[0:40]'