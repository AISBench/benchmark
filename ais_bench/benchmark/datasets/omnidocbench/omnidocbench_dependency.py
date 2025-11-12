OMNIDOCBENCH_INSTALLED = True
try:
    from ais_bench.benchmark.datasets.omnidocbench.end2end_dataset import End2EndDataset
    from ais_bench.benchmark.datasets.omnidocbench.metric import call_Edit_dist, call_CDM_plain, call_TEDS
    from ais_bench.benchmark.datasets.omnidocbench.registry import METRIC_REGISTRY
    from ais_bench.benchmark.datasets.omnidocbench.utils import get_full_labels_results, get_page_split
except ImportError as e:
    OMNIDOCBENCH_INSTALLED = False