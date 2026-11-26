import random
from mmengine.config import read_base

with read_base():
    from .ceval_gen_5_shot_str import ceval_datasets # noqa: F401, F403

# Runtime import also works with mmengine's lazy config loader.
_smoke_fast = __import__('os').environ.get('AISBENCH_SMOKE_FAST') == '1'
if _smoke_fast:
    ceval_datasets = ceval_datasets[:3]
for i in ceval_datasets:
    i['reader_cfg']['test_range'] = '[0:20]' if _smoke_fast else '[0:100]'
 