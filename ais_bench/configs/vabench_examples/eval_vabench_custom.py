"""Eval-only: prepare MP4 once, then run this config with ais_bench -m eval.

Edit the constants below. This example consumes an explicitly prepared custom subset, not the 778 standard set.
"""

from ais_bench.benchmark.datasets.vabench import VABenchDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import VABenchEvalTask
from ais_bench.benchmark.summarizers import VABenchSummarizer

PREPARED_DIR = '/workspace/prepared_videos'
SOURCE_ROOT = '/workspace/code/VABench-main'
CACHE_DIR = '/workspace/vabench-cache'
DEVICE = 'cuda:0'
BACKEND_PYTHON = None  # e.g. '/root/miniconda3/envs/vabench311/bin/python3'
NUM_DEVICES = 2  # number reserved by LocalRunner; GPU Omni may require more

models = [dict(attr='local', type='VABenchEvalPlaceholder', abbr='vabench_eval')]
datasets = [dict(
    type=VABenchDataset,
    abbr='vabench_t2av',
    path=PREPARED_DIR,
    reader_cfg=dict(input_columns=['idx'], output_column='idx'),
    eval_cfg=dict(
        source_root=SOURCE_ROOT,
        cache_dir=CACHE_DIR,
        device=DEVICE,
        python=BACKEND_PYTHON,
        device_overrides={'first_dnsmos': 'cpu'},
        params={'device_map': 'auto'},
    ),
)]
eval = dict(
    partitioner=dict(type=NaivePartitioner,
                     keep_keys=['eval.runner.task.num_gpus']),
    runner=dict(type=LocalRunner, max_num_workers=1,
                task=dict(type=VABenchEvalTask,
                          num_gpus=NUM_DEVICES)),
)
summarizer = dict(attr='accuracy', type=VABenchSummarizer)
