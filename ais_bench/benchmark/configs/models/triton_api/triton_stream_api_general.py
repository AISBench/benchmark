from ais_bench.benchmark.models import TritonCustomAPI

models = [
    dict(
        attr="service",
        type=TritonCustomAPI,
        abbr="triton-stream-api-general",
        path="",
        stream=True,
        model_name="",
        request_rate=0,
        retry=2,
        headers={"Content-Type": "application/json"},
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
