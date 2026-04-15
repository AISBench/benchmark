from ais_bench.benchmark.models.api_models.vita_generate_api import VITAGenerateAPI

models = [
    dict(
        attr="service",
        type=VITAGenerateAPI,
        abbr="vita-generate-chat",
        path="/mnt/nfs/weight/VITA-MLLM/VITA-1___5",
        model="vita",
        stream=False,
        request_rate=0,
        retry=2,
        api_key="",
        host_ip="127.0.0.1",
        host_port=1025,
        url="",
        max_out_len=512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(
            do_sample=False,
            temperature=0.01,
        ),
    )
]
