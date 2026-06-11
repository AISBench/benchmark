from mmengine.config import read_base
from ais_bench.benchmark.tasks.custom_tasks.harbor_task import HarborTask
from ais_bench.benchmark.tasks.base import EmptyTask
from ais_bench.benchmark.summarizers.harbor import HarborSummarizer

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer

models = [
    dict(
        abbr="terminus-2",
        agent_name="terminus-2",  # -a/--agent: Agent名称 (terminus-2, claude-code, openhands等)
        model_names=["hosted_vllm/qwen3"],  # -m/--model: 模型名称, hosted_vllm/{模型名称}
        agent_kwargs={  # --ak/--agent-kwarg: Agent额外参数
            "api_base": "http://0.0.0.0:8080/v1",  # terminus-2需要api_base连接推理服务，例如填"http://0.0.0.0:8080/v1"会访问"http://0.0.0.0:8080/v1/chat/completions"
            "model_info": {  # 模型token限制和成本信息
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
        },
        agent_env=None,  # --ae/--agent-env: 传递给agent的环境变量
    )
]

datasets = []

sub_tasks = ["terminal-bench-2"]
for task in sub_tasks:
    datasets.append(
        dict(
            abbr=f'harbor_{task}',
            args=dict(
                n_attempts=1,  # -k/--n-attempts: 每个trial的尝试次数
                timeout_multiplier=1.0,  # --timeout-multiplier: 超时倍数（所有超时乘以此系数）
                agent_timeout_multiplier=None,  # --agent-timeout-multiplier: Agent执行超时倍数（覆盖timeout-multiplier）
                verifier_timeout_multiplier=None,  # --verifier-timeout-multiplier: 验证器超时倍数
                agent_setup_timeout_multiplier=None,  # --agent-setup-timeout-multiplier: Agent设置超时倍数
                environment_build_timeout_multiplier=None,  # --environment-build-timeout-multiplier: 环境构建超时倍数
                debug=False,  # --debug: 启用调试日志
                n_concurrent_trials=5,  # -n/--n-concurrent: 并发运行的trial数量
                quiet=False,  # -q/--quiet: 静默模式
                max_retries=0,  # -r/--max-retries: 最大重试次数
                retry_include_exceptions=None,  # --retry-include: 需要重试的异常类型列表
                retry_exclude_exceptions=[  # --retry-exclude: 不需要重试的异常类型列表
                    # "AgentTimeoutError",
                    # "VerifierTimeoutError",
                    # "RewardFileNotFoundError",
                    "RewardFileEmptyError",
                    "VerifierOutputParseError",
                ],
                environment_type="docker",  # -e/--env: 环境类型 (docker, daytona, e2b, modal)
                environment_force_build=False,  # --force-build/--no-force-build: 是否强制重建环境
                environment_delete=False,  # --delete/--no-delete: 完成后是否删除环境
                path="/path/to/terminal-bench-2/",  # -p/--path: 本地数据集路径
                dataset_name_version=None,  # -d/--dataset: 远程数据集名称@版本
                task_names=None,  # --include-task-name: 包含的任务名称（支持glob模式）例如 ["task_name1", "task_name2"]
                exclude_task_names=None,  # --exclude-task-name: 排除的任务名称
                n_tasks=None,  # --n-tasks: 最大任务数量
                disable_verification=False,  # --disable-verification: 禁用验证器
                verifier_env=None,  # --ve/--verifier-env: 验证器环境变量
                yes=True,  # -y/--yes: 自动确认环境变量提示
                env_file=None,  # --env-file: .env文件路径
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
        task=dict(type=HarborTask)
    ),
)

summarizer = dict(
    attr="accuracy",
    type=HarborSummarizer,
)