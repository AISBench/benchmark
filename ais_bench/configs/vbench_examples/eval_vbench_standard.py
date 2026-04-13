# VBench 1.0 eval-only config
#
# Usage:
#   ais_bench ais_bench/configs/vbench_examples/eval_vbench_standard.py -m eval
#     → runs eval + summary (no -m eval needed)
#   ais_bench ais_bench/configs/vbench_examples/eval_vbench_standard.py -m viz
#     → runs summary only
#
from ais_bench.benchmark.datasets import VBenchDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import VBenchEvalTask
from ais_bench.benchmark.summarizers import VBenchSummarizer


DATA_PATH = ""

# Optional: VBench small-model cache root (same as env VBENCH_CACHE_DIR; overrides process env when set).
# VBENCH_CACHE_DIR = "/path/to/cache"

# Dimension list for VBench 1.0, total 16 dimensions
VBENCH_DEFAULT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "appearance_style",
]

models = [
    dict(
        attr="local",
        type="VBenchEvalPlaceholder",  # placeholder, not built in eval
        abbr="vbench_eval",
    )
]

vbench_eval_cfg = dict(
    load_ckpt_from_local=True,
    # full_json_dir: optional, default is third_party/vbench/VBench_full_info.json
    # prompt_file: optional; if set, custom_input mode is inferred automatically
    # category: optional; if set, vbench_category mode is inferred automatically
)

# Per-dimension VBench datasets: each dim is an independent eval task (abbr=vbench_<dim>).
datasets = [
    dict(
        abbr=f"vbench_{dim}",
        type=VBenchDataset,
        # path (or videos_path): required — set to your video directory; use --config with overrides or edit here
        path=DATA_PATH,
        eval_cfg=dict(
            **vbench_eval_cfg,
            dimension_list=[dim],
        ),
    )
    for dim in VBENCH_DEFAULT_DIMENSIONS
]


eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=VBenchEvalTask),
    ),
)


summarizer = dict(
    attr="accuracy",
    type=VBenchSummarizer,
)
