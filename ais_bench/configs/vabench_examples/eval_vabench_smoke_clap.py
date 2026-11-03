"""One pinned T2AV sample, one CLAP metric. Prepare with --subset smoke first.

Edit paths for the GPU server. This is a plumbing smoke test, not Mini10/778.
"""
from ais_bench.benchmark.datasets.vabench import VABenchDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import VABenchEvalTask
from ais_bench.benchmark.summarizers import VABenchSummarizer

PREPARED_DIR = '/workspace/prepared_kitten_gpu'
SOURCE_ROOT = '/workspace/code/VABench-main'
CACHE_DIR = '/workspace/vabench-cache'
DEVICE = 'cuda:0'
BACKEND_PYTHON = None

models = [dict(attr='local', type='VABenchEvalPlaceholder', abbr='doubao_kitten_smoke')]
datasets = [dict(
    type=VABenchDataset, abbr='vabench_smoke_clap', path=PREPARED_DIR,
    eval_cfg=dict(source_root=SOURCE_ROOT, cache_dir=CACHE_DIR,
                  dimensions=['second_clap'], device=DEVICE, python=BACKEND_PYTHON),
)]
eval = dict(
    partitioner=dict(type=NaivePartitioner, keep_keys=['eval.runner.task.num_gpus']),
    runner=dict(type=LocalRunner, max_num_workers=1,
                task=dict(type=VABenchEvalTask, num_gpus=1)),
)
summarizer = dict(attr='accuracy', type=VABenchSummarizer)
