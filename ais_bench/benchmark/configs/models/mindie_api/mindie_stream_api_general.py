from ais_bench.benchmark.models import MindieStreamApi

models = [
    dict(
        attr="service",
        type=MindieStreamApi,
        abbr="mindie-stream-api",
        path="",
        request_rate=0,
        retry=2,
        host_ip="localhost",
        host_port=8080,
        url="",
        max_out_len=512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False,
        ),
    )
]
