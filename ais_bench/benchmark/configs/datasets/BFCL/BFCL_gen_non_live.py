from mmengine.config import read_base

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


bfcl_datasets = (
    bfcl_simple
    + bfcl_irrelevance
    + bfcl_parallel
    + bfcl_multiple
    + bfcl_parallel_multiple
    + bfcl_java
    + bfcl_javascript
)
