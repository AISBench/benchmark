from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_multi_turn_base import (
        bfcl_datasets as bfcl_multi_turn_base,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_multi_turn_miss_func import (
        bfcl_datasets as bfcl_multi_turn_miss_func,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_multi_turn_miss_param import (
        bfcl_datasets as bfcl_multi_turn_miss_param,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_multi_turn_long_context import (
        bfcl_datasets as bfcl_multi_turn_long_context,
    )


bfcl_datasets = (
    bfcl_multi_turn_base
    + bfcl_multi_turn_miss_func
    + bfcl_multi_turn_miss_param
    + bfcl_multi_turn_long_context
)
