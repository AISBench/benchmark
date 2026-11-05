from ais_bench.benchmark.models import VLLMCustomAPI

models = [
    dict(
        attr="service",
        type=VLLMCustomAPI,
        abbr="vllm-api-general-stream-session",
        path="/mnt/weight/Qwen3-8B",
        model="qwen",
        stream=True,
        use_chat_api=True,            # 发送到 v1/chat/completions
        session_aware=True,           # 启用 session 模式
        session_concurrency=4,        # 同时活跃的 session 数上限（跨 session 在途总数）
        # pass_agent_hint：False 时 agent_hint 仅用于内部（提取 session_id/stage），不透传到 API;
        # True 时把 agent_hint 作为额外字段合并进 vLLM 请求 body。
        pass_agent_hint=True,
        # batch_size 在 session 模式下退化为 janus 队列容量（producer -> dispatcher 的缓冲）
        # 推荐设为 session_concurrency，dispatcher 实时路由，每个 session 按需取下一条
        batch_size=4,
        request_rate=0,               # 0=不限速；设 >0 可在 session 并发基础上叠加全局 RPS 限速
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=31036,
        url="",
        max_out_len=256,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=False,
        ),
    )
]
