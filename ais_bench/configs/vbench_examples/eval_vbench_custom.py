# VBench 1.0 eval-only config. No infer section — defaults to eval mode when run without -m.
#
# Usage:
#   ais_bench ais_bench/configs/vbench_examples/eval_vbench_custom.py
#     → runs eval + summary (no -m eval needed)
#   ais_bench ais_bench/configs/vbench_examples/eval_vbench_custom.py -m viz
#     → runs summary only
#
from ais_bench.benchmark.datasets import VBenchDataset
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import VBenchEvalTask
from ais_bench.benchmark.summarizers import DefaultSummarizer


DATA_PATH = "/data/zhanggaohua/datasets/vbench/CogVideoX-5B-mini/subject_consistency"

# Dimension list for VBench 1.0, total 10 dimensions
VBENCH_CUSTOM_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_style",
    "overall_consistency",
    "human_action",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
]

models = [
    dict(
        attr="local",
        type="VBenchEvalPlaceholder",  # placeholder, not built in eval
        abbr="vbench_eval",
    )
]


# Minimal reader_cfg/infer_cfg for framework compatibility (eval uses VBenchEvalTask only).
vbench_reader_cfg = dict(
    input_columns=["dummy"],
    output_column="dummy",
)

vbench_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template="{question}"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


vbench_eval_cfg = dict(
    load_ckpt_from_local=True,
    mode="custom_input",
    # full_json_dir: optional, default is third_party/vbench/VBench_full_info.json
    # prompt_file: optional; if set, custom_input mode is inferred automatically
)

# Per-dimension VBench datasets: each dim is an independent eval task (abbr=vbench_<dim>).
datasets = [
    dict(
        abbr=f"vbench_custom_{dim}",
        type=VBenchDataset,
        # path (or videos_path): required — set to your video directory; use --config with overrides or edit here
        path=DATA_PATH,
        reader_cfg=vbench_reader_cfg,
        infer_cfg=vbench_infer_cfg,
        eval_cfg=dict(
            **vbench_eval_cfg,
            dimension_list=[dim],
        ),
    )
    for dim in VBENCH_CUSTOM_DIMENSIONS
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
    type=DefaultSummarizer,
)
