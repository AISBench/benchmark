"""SWE-bench DinD 3×3 example config for ais_bench.

Demonstrates how to invoke the din_integration/ SWE-bench DinD pipeline
via AISBench's task system. Run after pip-installing both ais_bench and
this swebench_dind package:

    pip install ais_bench
    cd din_integration && pip install -e .
    ais_bench din_integration/configs/swebench_dind_3x3.py

Note: This config relies on the swebench-dind CLI being on PATH and the
DinD orchestrator container already running. See din_integration/README.md
for setup.
"""
from mmengine.config import read_base

# Use a no-op task for the standard Infer stage; the SWE-bench DinD pipeline
# doesn't fit AISBench's standard model-inference shape.
with read_base():
    infer = dict(runner=dict(task=dict(type="EmptyTask")))

# Eval stage uses our custom SwebenchDindTask.
eval = dict(
    runner=dict(
        task=dict(
            type="SwebenchDindTask",
        ),
    ),
)

# Summarizer: HarborSummarizer reads <work_dir>/results/<model>/<dataset>.json
summarizer = dict(attr="accuracy", type="HarborSummarizer")

work_dir = "./outputs/swebench_dind_3x3/"

# 3 cases × 3 agents = 9 trials
models = [
    dict(
        abbr="qwen3-coder-30b",
        type="LiteLLMModel",
        model_names=["openai/Qwen/Qwen3-Coder-30B-A3B-Instruct"],
        agent_name="aider",
        agent_kwargs={},
        agent_env={},
    ),
    dict(
        abbr="qwen3-coder-30b",
        type="LiteLLMModel",
        model_names=["openai/Qwen/Qwen3-Coder-30B-A3B-Instruct"],
        agent_name="mini-swe-agent",
        agent_kwargs={},
        agent_env={},
    ),
    dict(
        abbr="qwen3-coder-30b",
        type="LiteLLMModel",
        model_names=["openai/Qwen/Qwen3-Coder-30B-A3B-Instruct"],
        agent_name="qwen-coder",
        agent_kwargs={},
        agent_env={},
    ),
]

datasets = [
    dict(
        abbr="django-11099",
        type="SwebenchDindDataset",
        args=dict(
            path="/opt/swebench/data/tasks/django__django-11099-aider",
        ),
    ),
    dict(
        abbr="django-12308",
        type="SwebenchDindDataset",
        args=dict(
            path="/opt/swebench/data/tasks/django__django-12308-msa",
        ),
    ),
    dict(
        abbr="django-13741",
        type="SwebenchDindDataset",
        args=dict(
            path="/opt/swebench/data/tasks/django__django-13741-aider",
        ),
    ),
]