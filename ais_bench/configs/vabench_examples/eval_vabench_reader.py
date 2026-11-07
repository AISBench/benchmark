"""AISBench VABench evaluation config.

VABench is evaluation-only. This complete runnable config includes a reader
configuration for --num-prompts and explicit test_range selection. Run it by
passing its path directly, rather than through --datasets:

    ais_bench \
      ais_bench/configs/vabench_examples/eval_vabench_reader.py \
      -m eval -w /workspace/vabench_runs/aisbench_vabench

``--num-prompts N`` is handled by AISBench through
``reader_cfg.test_range='[:N]'``.
"""
from ais_bench.benchmark.datasets.vabench import VABenchDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.summarizers import VABenchSummarizer
from ais_bench.benchmark.tasks import VABenchEvalTask

PREPARED_DIR = '/workspace/prepared_videos'
SOURCE_ROOT = '/workspace/code/VABench-main'
CACHE_DIR = '/workspace/vabench-cache'
DEVICE = 'cuda:0'
BACKEND_PYTHON = None  # Current Python; or an absolute scoring Conda python3 path.
NUM_DEVICES = 2

# The prepared manifest rows are already the VABench samples. This reader
# config exists for AISBench's --num-prompts/test_range handling; no text
# generation inference is performed.
vabench_reader_cfg = dict(
    input_columns=['idx'],
    output_column='idx',
)

vabench_eval_cfg = dict(
    source_root=SOURCE_ROOT,
    cache_dir=CACHE_DIR,
    device=DEVICE,
    python=BACKEND_PYTHON,
    # DNSMOS official ONNX scorer runs explicitly on CPU.
    device_overrides={'first_dnsmos': 'cpu'},
    params={'device_map': 'auto'},
    # Omit dimensions to run all 15; set e.g. ['second_clap'] for a smoke test.
)

vabench_datasets = [
    dict(
        abbr='vabench_t2av',
        type=VABenchDataset,
        path=PREPARED_DIR,
        reader_cfg=vabench_reader_cfg,
        eval_cfg=vabench_eval_cfg,
    )
]

models = [
    dict(
        attr='local',
        type='VABenchEvalPlaceholder',
        abbr='vabench_eval',
    )
]

datasets = vabench_datasets

eval = dict(
    partitioner=dict(
        type=NaivePartitioner,
        keep_keys=['eval.runner.task.num_gpus'],
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        task=dict(
            type=VABenchEvalTask,
            num_gpus=NUM_DEVICES,
        ),
    ),
)

summarizer = dict(
    attr='accuracy',
    type=VABenchSummarizer,
)
