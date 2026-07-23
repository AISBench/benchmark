from ais_bench.benchmark.datasets import SWEBenchProDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import SWEBenchProInferTask, SWEBenchProEvalTask
from ais_bench.benchmark.summarizers import SWEBenchProSummarizer

# 数据集路径：用 mmengine 的 {{$VAR:default}} 占位符从环境变量读取。
# mmengine 在执行本文件前，会先用 os.environ['VAR'] 的值替换整个 {{$VAR:default}}
# 占位符；如果环境变量未设置，则替换为 default 部分（此处为空 → 沿用 HF 在线下载）。
#
# 这里 AISBENCH_AGENT_DATASET_PATH 的值就是 bootstrap.sh --datasets 传入的完整路径，
# 由 bootstrap.sh 自动注入到容器环境变量中，无需 import sys / import os 任何库。
#
# 使用流程：
#   1. 物理机: bash bootstrap.sh --datasets /data/datasets/swebench_pro/<...>
#   2. 容器内: ais_bench <此配置> --debug     # path 自动从 env var 读
#
# 不使用 agent_runtime 容器方案时：直接 export AISBENCH_AGENT_DATASET_PATH=<path> 即可；
# 不 export 时回落到 HF 在线下载（默认行为）。
DEFAULT_DATASET_PATH = '{{$AISBENCH_AGENT_DATASET_PATH:}}'

STEP_LIMIT = 250

models = [
    dict(
        attr="local",
        abbr="swebench_pro_full_model",
        type="LiteLLMChat",
        model="",
        api_key="EMPTY",
        url="http://127.0.0.1:8000/v1",
        batch_size=1,
        generation_kwargs=dict(
            # Supports arbitrary generation parameters, consistent with regular model tasks.
            # Common parameters include temperature, top_p, top_k, timeout, etc.
            # temperature=0.0,   # Set 0 for deterministic output; omit or set >0 for diversity
            # top_p=1.0,
            # top_k=-1,
            # timeout=200,       # Inference timeout in seconds
        ),
    )
]

SWEBP_SCRIPT_PATH_ABS = ""
SWEBP_DOCKER_PATH_ABS = ""

datasets = [
    dict(
        type=SWEBenchProDataset,
        abbr="swebench_pro_full_data",
        # 本字段通过环境变量 AISBENCH_AGENT_DATASET_PATH 注入；不设置时为空，沿用 HF 在线下载
        path=DEFAULT_DATASET_PATH,
        name="full",
        split="test",
        step_limit=STEP_LIMIT,
        filter_spec="",
        shuffle=False,
        swebp_scripts_dir=SWEBP_SCRIPT_PATH_ABS,
        swebp_docker_dir=SWEBP_DOCKER_PATH_ABS,
    ),
]

summarizer = dict(
    attr="accuracy",
    type=SWEBenchProSummarizer,
)

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=SWEBenchProInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=SWEBenchProEvalTask),
    ),
)
