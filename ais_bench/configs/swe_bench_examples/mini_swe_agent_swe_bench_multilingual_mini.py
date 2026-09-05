from ais_bench.benchmark.datasets import SWEBenchDataset
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import SWEBenchInferTask, SWEBenchEvalTask
from ais_bench.benchmark.summarizers import SWEBenchSummarizer

# 数据集路径：用 mmengine 的 {{$VAR:default}} 占位符从环境变量读取。
# mmengine 在执行本文件前，会先用 os.environ['VAR'] 的值替换整个 {{$VAR:default}}
# 占位符；如果环境变量未设置，则替换为 default 部分（此处为空 → HF 在线下载会失败）。
#
# 这里 AISBENCH_AGENT_DATASET_PATH 的值就是 bootstrap.sh --datasets 传入的完整路径，
# 由 bootstrap.sh 自动注入到容器环境变量中，无需 import sys / import os 任何库。
#
# 使用流程：
#   1. 物理机: 下载 mini 数据集并准备目录（HF 无该数据集）
#      https://modelers.cn/datasets/AISBench/SWE-Bench_Multilingual_mini
#   2. 物理机: bash bootstrap.sh --datasets /data/datasets/swebench/<mini路径>
#   3. 容器内: ais_bench <此配置> --debug     # path 自动从 env var 读
#
# 不使用 agent_runtime 容器方案时：直接 export AISBENCH_AGENT_DATASET_PATH=<path> 即可。
DEFAULT_DATASET_PATH = '{{$AISBENCH_AGENT_DATASET_PATH:}}'

STEP_LIMIT = 200

models = [
    dict(
        attr="local",
        abbr="swebench",
        type="LiteLLMChat",
        model="",
        api_key="EMPTY",
        url="http://127.0.0.1:8080/v1",  # API base, e.g. http://127.0.0.1:8000/v1
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

datasets = [
    dict(
        type=SWEBenchDataset,
        abbr="swebench_multilingual_mini",
        # Relative to AIS_BENCH_DATASETS_CACHE (default: project root); missing -> HF download
        # 本字段通过环境变量 AISBENCH_AGENT_DATASET_PATH 注入；mini 数据集 HF 无，必须 env var 注入本地路径
        path=DEFAULT_DATASET_PATH,
        name="multilingual",
        split="test",
        step_limit=STEP_LIMIT,
        filter_spec="",
        shuffle=False,
    ),
]

summarizer = dict(
    attr="accuracy",
    type=SWEBenchSummarizer,
)


infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=SWEBenchInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=SWEBenchEvalTask),
    ),
)
