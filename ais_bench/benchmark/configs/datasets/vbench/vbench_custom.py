# VBench 1.0 custom input evaluation (prompt from file or filename).
# Usage: ais_bench --mode eval --models vbench_eval --datasets vbench/vbench_custom
# Set path to your video folder; set eval_cfg.prompt_file for prompt dict JSON.
from ais_bench.benchmark.datasets import VBenchDataset

vbench_reader_cfg = dict(
    input_columns=['dummy'],
    output_column='dummy',
)

vbench_infer_cfg = dict(
    inferencer='vbench_eval',
)

vbench_eval_cfg = dict(
    use_vbench_task=True,
    device='cuda',
    # prompt_file: path to JSON {"video_path": "prompt", ...}; if set, custom_input
    # mode is inferred automatically. If omitted, prompts are derived from filenames.
)

VBENCH_CUSTOM_DIMENSIONS = [
    'subject_consistency', 'background_consistency', 'aesthetic_quality',
    'imaging_quality', 'temporal_style', 'overall_consistency',
    'human_action', 'temporal_flickering', 'motion_smoothness', 'dynamic_degree',
]

# Base path to generated videos; override via CLI/config as needed.
_BASE_PATH = ''

# Config key must end with 'vbench_custom' when using --datasets vbench/vbench_custom
# Per-dimension custom-input datasets (abbr=vbench_custom_<dim>).
_vbench_custom_single_dim = [
    dict(
        abbr=f'vbench_custom_{dim}',
        type=VBenchDataset,
        path=_BASE_PATH,  # required: your video directory
        reader_cfg=vbench_reader_cfg,
        infer_cfg=vbench_infer_cfg,
        eval_cfg=dict(
            **vbench_eval_cfg,
            dimension_list=[dim],
        ),
    )
    for dim in VBENCH_CUSTOM_DIMENSIONS
]

# Aggregated config that evaluates all custom-input dimensions in one run.
_vbench_custom_all_dims = [
    dict(
        abbr='vbench_custom_all',
        type=VBenchDataset,
        path=_BASE_PATH,
        reader_cfg=vbench_reader_cfg,
        infer_cfg=vbench_infer_cfg,
        eval_cfg=dict(
            **vbench_eval_cfg,
            dimension_list=VBENCH_CUSTOM_DIMENSIONS,
        ),
    )
]

# Exported entry used by `--datasets vbench/vbench_custom`.
vbench_custom = _vbench_custom_single_dim + _vbench_custom_all_dims
