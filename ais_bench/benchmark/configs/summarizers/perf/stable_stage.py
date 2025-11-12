from mmengine.config import read_base
from ais_bench.benchmark.summarizers import DefaultPerfSummarizer
from ais_bench.benchmark.calculators import StablePerfMetricCalculator

summarizer = dict(
    attr = "performance",
    type=DefaultPerfSummarizer,
    calculator=dict(
        type=StablePerfMetricCalculator,
        stats_list=["Average", "Min", "Max", "Median", "P75", "P90", "P99"],
    )
)