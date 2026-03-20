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
        llm_agent = "openai/qwen3",               # agent 使用的 LLM，默认为 DEFAULT_LLM_AGENT
        llm_args_agent = {"api_base": "http://localhost:2498/v1", "temperature": 0.5},  # agent LLM 的参数，默认为 {"temperature": DEFAULT_LLM_TEMPERATURE_AGENT}
    )
]

work_dir = 'outputs/default/'

datasets = []
sub_tasks = ["airline", "retail", "telecom"]
for task in sub_tasks:
    datasets.append(
        dict(
            abbr=f'tau2_bench_{task}',
            args = dict(
                domain = "airline",                      # -d, 要运行的模拟域，可选值为 get_options().domains ["airline", "retail", "telecom"]
                num_trials = 1,                     # 每个任务运行的次数，默认为 1
                # agent = "baseline",                 # 使用的 agent 实现，默认为 DEFAULT_AGENT_IMPLEMENTATION
                # agent_llm = "openai/gpt-4o",               # agent 使用的 LLM，默认为 DEFAULT_LLM_AGENT
                # agent_llm_args = {"api_base": "http://localhost:2998/v1", "temperature": 0.0},  # agent LLM 的参数，默认为 {"temperature": DEFAULT_LLM_TEMPERATURE_AGENT}
                user = None,                  # 使用的 user 实现，默认为 DEFAULT_USER_IMPLEMENTATION
                llm_user = "openai/qwen3",                # user 使用的 LLM，默认为 DEFAULT_LLM_USER
                llm_args_user = {"api_base": "http://localhost:2498/v1", "temperature": 1.0},   # user LLM 的参数，默认为 {"temperature": DEFAULT_LLM_TEMPERATURE_USER}
                task_set_name = None,               # 要运行的任务集，如未提供则加载域的默认任务集
                task_split_name = None,           # 要运行的任务分割，默认为 'base'
                task_ids = None,                    # 可选，只运行指定 ID 的任务
                num_tasks = 5,                   # 要运行的任务数量
                max_steps = None,                    # 模拟运行的最大步数，默认为 DEFAULT_MAX_STEPS
                max_errors = None,                     # 模拟中连续允许的最大工具错误数，默认为 DEFAULT_MAX_ERRORS
                # save_to = None,                     # 模拟结果的保存路径，保存到 data/simulations/<save_to>.json
                max_concurrency = 5,               # 并发运行的最大模拟数，默认为 DEFAULT_MAX_CONCURRENCY
                seed = None,                       # 模拟使用的随机种子，默认为 DEFAULT_SEED
                log_level = "INFO",                 # 模拟的日志级别，默认为 DEFAULT_LOG_LEVEL
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

sub_task_count = { # default
    "airline": 50,
    "retail": 114,
    "telecom": 114,
}

tau2_task_weights = {}
for ds_config in datasets:
    task = ds_config["args"]["domain"]
    if not ds_config["args"]["num_tasks"]:
        tau2_task_weights[ds_config["abbr"]] = sub_task_count[task]
    else:
        tau2_task_weights[ds_config["abbr"]] = ds_config["args"]["num_tasks"]

tau2_summary_groups = [
    {'name': 'tau2_bench_avg', 'subsets': [ds_config["abbr"] for ds_config in datasets], 'weights': tau2_task_weights},
]

summarizer = dict(
    attr = "accuracy",
    summary_groups=tau2_summary_groups,
)
