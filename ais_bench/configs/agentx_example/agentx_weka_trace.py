"""AgentX（SemiAnalysis InferenceX）Weka Trace 性能测评示例配置。

运行方式::

    # 1) 直接命令行执行，不需要改本文件
    ais_bench --mode agentx \
        --url http://127.0.0.1:8000 \
        --model-name qwen3.8 \
        --path /mnt/weight/Qwen3.8-27B-w8a8 \
        --batch-size 8 \
        --agentx-python ~/miniconda3/envs/agentx113/bin/python

    # 2) 或者用配置模板
    ais_bench ais_bench/configs/agentx_example/agentx_weka_trace.py --mode agentx

    # 3) 预集成镜像（Linux，无需安装 Harness 或传 AgentX 环境参数）
    docker run --rm --network host \
        -v /mnt/weight:/mnt/weight -v "$PWD"/outputs:/benchmark/outputs \
        ghcr.io/aisbench/aisbench_benchmark:<TAG>-ubuntu24.04-py312-agentx-x86_64 \
        ais_bench --mode agentx --url http://127.0.0.1:8000 \
        --model-name qwen3.8 --path /mnt/weight/Qwen3.8-27B-w8a8 --batch-size 8

命令行模式下常用的补充参数：

- ``--agentx-python``：显式指定 Conda/venv 中的 Harness 解释器，也可使用 ``AISBENCH_AGENTX_PYTHON``；
- ``--agentx-env HF_ENDPOINT=https://hf-mirror.com``：给子进程注入环境变量（可重复）；
  也支持 ``--agentx-env AGENTX_PROCESS_TIMEOUT=7200`` 为 Harness 设置看门狗超时（秒，默认不启用）；
- ``--agentx-extra-arg '{ --log-level INFO }'``：直接写出 agentx 原始参数，追加到
  组合好的 ``agentx profile`` 命令末尾；与已有参数重名时以这里的值为准。
  shell 侧必须用**单引号**包住整段，否则空格会把它拆成多个 AISBench 参数并报
  ``unrecognized arguments``；段内可用双引号，如
  ``--agentx-extra-arg '{ --concurrency 2 --goodput "time_to_first_token:10000" }'``；
  值里若含单引号，用 ``'"'"'`` 拼接；不要用中文弯引号；
- ``--reuse latest``：沿用 ``--work-dir`` 下最近的时间戳目录重跑，产物覆盖原结果。

本配置只驱动上游 ``agentx-harness`` 的 ``agentx`` 预设，不实现 Trace 回放：

- 并发由 ``--batch-size`` 指定（缺省 1），可传多个档位，逐档串行执行；
- 数据集、warmup、时长都走榜单口径（1h profile + 每 lane 10 次 warmup + seed 42）；
- 多档并发共用同一服务，每档使用独立 run ID、日志和输出目录；档间默认等待 5 秒。

跑之前推荐使用独立 Conda 环境安装依赖（预集成 ``-agentx`` 镜像可跳过）::

    conda create -n agentx113 python=3.11 -y
    conda run -n agentx113 python -m pip install agentx-harness==1.0.13

也可使用独立 venv。Conda 和 venv 都通过 ``agentx_python`` 显式提供对应环境的
Python 路径；上述 Conda 示例通常为 ``~/miniconda3/envs/agentx113/bin/python``，
实际路径可用 ``conda run -n agentx113 which python`` 查询。

不要装进 AISBench 环境：AISBench 锁 ``Pillow==11.2.1``，而 harness 要求
``pillow~=12.3.0``，pip 会直接解析失败；另外 harness 内置的 fork 版 ``aiperf``
也会和上游 aiperf 争用同名模块。

另外必须满足：

1. ``agentx-harness==1.0.13`` 已安装（Python 3.11~3.13）；
2. 语料变体必须匹配服务端 ``--max-model-len``：``full`` 变体单请求峰值输入约 98.9 万 token，
   262144 窗口的服务必须改用 ``256k`` 变体，否则 warmup 首条请求直接 400；
3. 服务端开启 prefix caching（AgentX 有 96% 前缀复用，关掉结果不可比）；
4. 多档并发由 AISBench 逐档串行并为每档创建全新输出目录，档间默认等待 5 秒（``--cooldown`` 可调）。
"""

from ais_bench.benchmark.summarizers.agentx import AgentXSummarizer

models = [
    dict(
        abbr="qwen3.8",
        # 被测服务暴露的模型名（不是权重目录）
        model="qwen3.8",
        # 服务 base URL，不要带 Chat Completions 路径
        url="http://127.0.0.1:8000",
        # 追加到 base URL 的请求路径，对应 `agentx --endpoint`
        endpoint="/v1/chat/completions",
        # 模型/tokenizer 目录或 HF 名称；同时作为 harness 的 --tokenizer
        path="/mnt/weight/Qwen3.8-27B-w8a8",
        # 对应 agentx 的 --tokenizer-trust-remote-code（官方 quickstart 默认带上）
        trust_remote_code=True,
        # api_key="EMPTY",
    )
]

datasets = [
    dict(
        abbr="weka-trace-062126",
        args=dict(
            # 并发档位；可写多个值（如 [1, 8, 32]）逐档串行执行，
            # 每档一个独立 Harness 进程和全新输出目录
            concurrency=8,
            # benchmark 留空沿用官方 3600 秒，也可显式指定（不得低于 900 秒）；
            # smoke 留空即默认 60 秒 / 4 条，也可显式覆盖
            run_mode="benchmark",
            duration=None,
            trace_limit=None,
            # 必填：full / 256k；不按模型名自动推断
            variant="full",
            # 本地语料目录（内含 traces.jsonl）：配置后走离线加载，
            # 留空则用 --public-dataset 由上游自动下载；
            # 本地内容必须与注册的官方 hash 一致才能产出正式结果，
            # smoke 自动允许不匹配并记录 actual
            dataset_path=None,
            # 显式 Harness 解释器；留空再检查 AISBENCH_AGENTX_PYTHON、
            # 镜像 /opt/venvs/agentx113 和当前解释器
            agentx_python="~/miniconda3/envs/agentx113/bin/python",
            # agentx 可执行文件；留空按 agentx_python 同目录自动推断
            agentx_bin=None,
            # 直接写出 agentx 原始参数（整段字符串），分词后追加到命令末尾；例如
            # '{ --log-level INFO --concurrency 1 }'
            # 命令行传入时必须用单引号包住整段，避免被 shell 按空格拆开
            extra_args=None,
            env={},
        ),
    )
]

work_dir = "outputs/agentx/"

summarizer = dict(
    attr="agentx",
    type=AgentXSummarizer,
)
