from mmengine.config import read_base

with read_base():
    from .livecodebench_code_generate_lite_gen_0_shot_chat import LCB_datasets  # noqa: F401, F403

LCB_datasets[0]['reader_cfg']['test_range'] = '[0:100]'