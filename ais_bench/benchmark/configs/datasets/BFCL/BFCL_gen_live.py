from mmengine.config import read_base

with read_base():
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


bfcl_datasets = (
    bfcl_live_simple
    + bfcl_live_multiple
    + bfcl_live_parallel
    + bfcl_live_parallel_multiple
    + bfcl_live_irrelevance
    + bfcl_live_relevance
)
