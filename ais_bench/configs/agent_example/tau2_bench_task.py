from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.tasks.custom_tasks.tau2_bench_task import TAU2BenchTask
from ais_bench.benchmark.tasks.base import EmptyTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer

models = [
    dict(
        abbr="openai-v1-chat",
        api_key=None, # API KEY 默认是个无效字符串 ,内部会声明OPENAI_API_KEY
        agent = None,                 # 使用的 agent 实现，默认为 DEFAULT_AGENT_IMPLEMENTATION
        llm_agent = "openai/qwen3",               # 必填，agent 使用的 LLM，填写"openai/{推理服务的模型名称}"
        llm_args_agent = { # agent LLM 的参数，支持传其他兼容openai接口格式的参数
            "api_base": "http://localhost:2498/v1", # 必填，推理服务的base_url
            "temperature": 0.5
        },
    )
]

datasets = []

task_count_map = {
    "airline": None,
    "retail": None,
    "telecom": None,
}

sub_tasks = ["airline", "retail", "telecom"]
for task in sub_tasks:
    datasets.append(
        dict(
            abbr=f'tau2_bench_{task}',
            args = dict(
                domain = task,                      # -d, 要运行的模拟域，可选值为 "airline", "retail", "telecom"
                num_trials = 1,                     # 每个任务运行的次数，默认为 1
                user = None,                  # 使用的 user 实现，默认为 DEFAULT_USER_IMPLEMENTATION
                llm_user = "openai/qwen3",                # 必填，user 使用的 LLM，填写"openai/{推理服务的模型名称}"
                llm_args_user = { # user LLM 的参数，支持传其他兼容openai接口格式的参数
                    "api_base": "http://localhost:2498/v1", # 必填，推理服务的base_url
                    "temperature": 0.0
                },
                task_set_name = None,               # 要运行的任务集，如未提供则加载域的默认任务集
                task_split_name = None,           # 要运行的任务分割，默认为 'base'，可选 'base','train', 'test'
                task_ids = None,                    # 可选，只运行指定 ID 的任务
                num_tasks = task_count_map[task],                   # 要运行的任务数量
                max_steps = None,                    # 模拟运行的最大步数，默认为 DEFAULT_MAX_STEPS=200
                max_errors = None,                     # 模拟中连续允许的最大工具错误数，默认为 DEFAULT_MAX_ERRORS=10
                max_concurrency = 5,               # 并发运行的最大模拟数，默认为 DEFAULT_MAX_CONCURRENCY=5
                seed = None,                       # 模拟使用的随机种子，默认为 DEFAULT_SEED=300
                log_level = "INFO",                 # 模拟的日志级别，默认为 DEFAULT_LOG_LEVEL="INFO"
                enforce_communication_protocol = False,  # 是否强制执行通信协议规则，默认为 False
            ),
        )
    )

infer = dict(
    runner=dict(
        task=dict(type=EmptyTask)
    ),
)

eval = dict(
    runner=dict(
        task=dict(type=TAU2BenchTask)
    ),
)


"""
default tasks count reference
### Airline
- train : 30
- test : 20
- base : 50 (train + test)
### Retail
- train : 74
- test : 40
- base : 114 (train + test)
### Telecom
- small : 20
- train : 74
- test : 40
- base : 114 (train + test)
- full : 2285
"""

default_task_count = { # default
    "airline": 50,
    "retail": 114,
    "telecom": 114,
}

tau2_task_weights = {}
for ds_config in datasets:
    task = ds_config["args"]["domain"]
    if not ds_config["args"]["num_tasks"]:
        tau2_task_weights[ds_config["abbr"]] = default_task_count[task]
    else:
        tau2_task_weights[ds_config["abbr"]] = ds_config["args"]["num_tasks"]

_all_sub_set = [(ds_config["abbr"], f'pass@{datasets[0]["args"]["num_trials"]}') for ds_config in datasets]

tau2_summary_groups = [
    {'name': 'tau2_bench_avg', 'subsets': _all_sub_set},
    {'name': 'tau2_bench_avg-weighted', 'subsets': _all_sub_set, 'weights': tau2_task_weights},
]

summarizer = dict(
    attr = "accuracy",
    summary_groups=tau2_summary_groups,
)
