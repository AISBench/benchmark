from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPI

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import (
        synthetic_datasets,
    )

datasets = synthetic_datasets

models = [
    dict(
        attr="service",
        type=VLLMCustomAPI,
        abbr='vllm-api-rps-distribution',
        path="",
        model="",
        stream=True,
        request_rate=100,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8080,
        url="",
        max_out_len=512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(temperature=0.01, ignore_eos=False),
        traffic_cfg=dict(
            burstiness=0.5,
            ramp_up_strategy="linear",
            ramp_up_start_rps=10,
            ramp_up_end_rps=200,
        ),
    )
]

work_dir = 'outputs/rps_distribution_perf/'
