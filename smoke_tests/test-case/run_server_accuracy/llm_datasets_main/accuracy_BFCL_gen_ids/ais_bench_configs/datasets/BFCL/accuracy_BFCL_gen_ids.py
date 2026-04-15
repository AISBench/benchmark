from mmengine import read_base

test_ids_to_generate = {
    "simple": ["simple_16"],
    "live_simple": ["live_simple_0-0-0"],
    "multi_turn_base": ["multi_turn_base_0"],
}

with read_base():
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_simple import (
        bfcl_datasets as bfcl_simple,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_irrelevance import (
        bfcl_datasets as bfcl_irrelevance,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_parallel import (
        bfcl_datasets as bfcl_parallel,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_multiple import (
        bfcl_datasets as bfcl_multiple,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_parallel_multiple import (
        bfcl_datasets as bfcl_parallel_multiple,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_java import (
        bfcl_datasets as bfcl_java,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_javascript import (
        bfcl_datasets as bfcl_javascript,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_live_simple import (
        bfcl_datasets as bfcl_live_simple,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_live_multiple import (
        bfcl_datasets as bfcl_live_multiple,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_live_parallel import (
        bfcl_datasets as bfcl_live_parallel,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_live_parallel_multiple import (
        bfcl_datasets as bfcl_live_parallel_multiple,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_live_irrelevance import (
        bfcl_datasets as bfcl_live_irrelevance,
    )
    from ais_bench.benchmark.configs.datasets.BFCL.BFCL_gen_live_relevance import (
        bfcl_datasets as bfcl_live_relevance,
    )
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


data_list = (
    bfcl_simple
    + bfcl_irrelevance
    + bfcl_parallel
    + bfcl_multiple
    + bfcl_parallel_multiple
    + bfcl_java
    + bfcl_javascript
    + bfcl_live_simple
    + bfcl_live_multiple
    + bfcl_live_parallel
    + bfcl_live_parallel_multiple
    + bfcl_live_irrelevance
    + bfcl_live_relevance
    + bfcl_multi_turn_base
    + bfcl_multi_turn_miss_func
    + bfcl_multi_turn_miss_param
    + bfcl_multi_turn_long_context
)

bfcl_datasets = []
for data in data_list:
    if data['category'] in test_ids_to_generate and test_ids_to_generate[data['category']]:
        data['test_ids'] = test_ids_to_generate[data['category']]
        bfcl_datasets.append(data)