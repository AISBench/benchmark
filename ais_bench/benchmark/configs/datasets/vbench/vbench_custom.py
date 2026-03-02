# VBench 1.0 custom evaluation dataset config.
# Usage: ais_bench --mode eval --models vbench_eval --datasets vbench_custom
# Set path (or videos_path) to your folder of generated videos; optionally set full_json_dir.
from ais_bench.benchmark.datasets import VBenchDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

# Minimal reader_cfg/infer_cfg for framework compatibility (eval uses VBenchEvalTask only).
vbench_reader_cfg = dict(
    input_columns=['dummy'],
    output_column='dummy',
)

vbench_infer_cfg = dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{question}'),
                        dict(role='BOT', prompt=''),
                    ]
                )
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer)
        )

VBENCH_CUSTOM_DIMENSIONS = [
    'subject_consistency', 'background_consistency', 'aesthetic_quality',
    'imaging_quality', 'temporal_style', 'overall_consistency',
    'human_action', 'temporal_flickering', 'motion_smoothness', 'dynamic_degree',
]

vbench_eval_cfg = dict(
    use_vbench_task=True,
    load_ckpt_from_local=True,
    mode='custom_input',
    # full_json_dir: optional, default is third_party/vbench/VBench_full_info.json
    # prompt_file: optional; if set, custom_input mode is inferred automatically
    # category: optional; if set, vbench_category mode is inferred automatically
)

_BASE_PATH = '/data/zhanggaohua/datasets/vbench/lavie/animal/'

# Per-dimension VBench datasets: each dim is an independent eval task (abbr=vbench_<dim>).
vbench_custom_datasets = [
    dict(
        abbr=f'vbench_custom_{dim}',
        type=VBenchDataset,
        # path (or videos_path): required — set to your video directory; use --config with overrides or edit here
        path=_BASE_PATH,
        reader_cfg=vbench_reader_cfg,
        infer_cfg=vbench_infer_cfg,
        eval_cfg=dict(
            **vbench_eval_cfg,
            dimension_list=[dim],
        ),
    )
    for dim in VBENCH_CUSTOM_DIMENSIONS
]
