from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPI

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import (
        synthetic_datasets,
    )

datasets = synthetic_datasets

models = []
for rate in [0, 5, 10, 20]:
    model_cfg = dict(
        attr="service",
        type=VLLMCustomAPI,
        abbr=f'vllm-api-steady-rate-{rate}',
        path="",
        model="",
        stream=True,
        request_rate=rate,
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
    )
    models.append(model_cfg)

work_dir = 'outputs/steady_state_perf/'
