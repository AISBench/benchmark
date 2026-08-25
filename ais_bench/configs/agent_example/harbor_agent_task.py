from mmengine.config import read_base
from ais_bench.benchmark.tasks.custom_tasks.harbor_agent_task import HarborAgentTask
from ais_bench.benchmark.runners.harbor_runner import HarborRunner
from ais_bench.benchmark.tasks.base import EmptyTask
from ais_bench.benchmark.summarizers.harbor import HarborSummarizer

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer

# models：承载模型服务与 agent 本身相关的参数。
# 统一语义参数（api_base / api_key / llm_kwargs / model_info）由
# AgentParamAdapter 自动转换为各 agent 私有参数（如 terminus-2 → kwargs.api_base，
# claude-code → ANTHROPIC_BASE_URL 环境变量）；agent_kwargs / agent_env
# 优先级最高，配置后原样直通 harbor。
models = [
    dict(
        abbr="terminus-2",
        agent_name="terminus-2",  # -a/--agent: harbor AgentName 或 module.path:ClassName
        # agent_import_path=None,  # 自定义 agent（module.path:ClassName）
        model_names=["hosted_vllm/qwen3"],  # --model: 模型名称
        api_base="http://0.0.0.0:8080/v1",  # --api-base: 模型服务 base url（统一语义）
        # api_key=None,            # --api-key: 模型服务 API key
        llm_kwargs={  # LLM 调用参数，合并进 agent kwargs
            "max_tokens": 4096,
            # "temperature": 0.7,
        },
        model_info={  # 模型 token 限制与成本信息
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        # 原始 agent 私有参数/环境变量（优先级高于统一参数转换结果）
        # agent_kwargs={},
        # agent_env=None,          # --ae/--agent-env: 传递给 agent 的环境变量
        # ---- 以下为可选的高级 agent 参数（均有对应 CLI 或配置项）----
        # deps_path=None,          # --agent-deps: 离线 agent 依赖包路径
        # n_concurrent=None,       # --n-concurrent-agents: 每个 agent 并发上限
        # skills=None,             # skill 目录或 git 源，可多值
        # mcp_servers=None,        # MCP 服务器配置列表
        # resume_trajectory=False, # 多步任务跨步恢复 agent 会话
        # load_trajectory=None,    # 预加载轨迹文件路径
        # extra_allowed_hosts=None,# 额外允许的 host/IP
        # include_logs=None,       # 需要保留的 agent 日志 glob
        # exclude_logs=None,       # 需要排除的 agent 日志 glob
        # override_timeout_sec=None,
        # override_setup_timeout_sec=None,
        # max_timeout_sec=None,
    )
]

# datasets：agent 测评任务本身的参数。
datasets = []

sub_tasks = ["terminal-bench-2"]
for task in sub_tasks:
    datasets.append(
        dict(
            abbr=f'harbor_{task}',
            args=dict(
                n_attempts=1,  # -k/--n-attempts: 每个trial的尝试次数
                timeout_multiplier=1.0,  # --timeout-multiplier: 超时倍数
                agent_timeout_multiplier=None,  # --agent-timeout-multiplier
                verifier_timeout_multiplier=None,  # --verifier-timeout-multiplier
                agent_setup_timeout_multiplier=None,  # --agent-setup-timeout-multiplier
                environment_build_timeout_multiplier=None,  # --environment-build-timeout-multiplier
                debug=False,  # --debug: 启用调试日志
                n_concurrent_trials=5,  # -n/--n-concurrent: 并发运行的trial数量
                quiet=False,  # -q/--quiet: 静默模式
                max_retries=0,  # --max-retries: 最大重试次数
                retry_include_exceptions=None,  # --retry-include
                retry_exclude_exceptions=[  # --retry-exclude
                    "RewardFileEmptyError",
                    "VerifierOutputParseError",
                ],
                environment_type="docker",  # -e/--environment: 环境类型
                environment_force_build=False,  # --force-build/--no-force-build
                environment_delete=False,  # --delete/--no-delete
                # environment_kwargs=None,  # 环境附加参数（--host-network 会写入 {"host_network": True}）
                path="/path/to/terminal-bench-2/",  # -p/--path: 本地数据集路径（也支持单 task 目录）
                dataset_name_version=None,  # -d/--dataset: 远程数据集 name@version / org/name@ref
                task_names=None,  # --include-task-name
                exclude_task_names=None,  # --exclude-task-name
                n_tasks=None,  # --n-tasks: 最大任务数量
                disable_verification=False,  # --disable-verification
                verifier_env=None,  # --ve/--verifier-env
                yes=True,  # -y/--yes: 自动确认环境变量提示
                env_file=None,  # --env-file: .env文件路径
            ),
        )
    )

# agent 模式无需原生 inference 阶段
infer = dict(
    runner=dict(
        task=dict(type=EmptyTask)
    ),
)

eval = dict(
    runner=dict(
        type=HarborRunner,
        monitor_port=0,  # --monitor-port: 监控服务端口（0=关闭）
        task=dict(type=HarborAgentTask),
    ),
)

summarizer = dict(
    attr="accuracy",
    type=HarborSummarizer,
)
