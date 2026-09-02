from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.perf.default_perf import summarizer
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

# Fixed Request Count Evaluation:
# Method 1 (basic): pass `--num-prompts N` on the command line to read only the first N samples.
# Method 2 (advanced): control the sampling range flexibly via `reader_cfg.test_range`,
# for example '[0:8]' reads the first 8 samples; '[10:20]' reads samples from index 10 to 20.
datasets[0]['reader_cfg']['test_range'] = '[0:8]'

models = vllm_api_stream_chat
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8080
models[0]["max_out_len"] = 512
models[0]["batch_size"] = 1
# Fixed Request Count Performance Evaluation:
# Set request_rate to -1 to send requests concurrently without rate limiting (max throughput).
models[0]["request_rate"] = -1
models[0]["generation_kwargs"] = dict(temperature=0.01, ignore_eos=True)

work_dir = "outputs/default/"

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLApiInferTask),
    ),
)
