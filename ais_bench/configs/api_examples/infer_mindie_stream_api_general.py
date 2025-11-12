from mmengine.config import read_base
from ais_bench.benchmark.models import MindieStreamApi
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local_api import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str

datasets = [
    *gsm8k_0_shot_cot_str,
]


models = [
    dict(
        attr="service",
        type=MindieStreamApi,
        path='',
        abbr='mindie-stream-api-general',
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",
        host_port = 8080,
        max_out_len = 512,
        batch_size=1,
        generation_kwargs = dict(
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            do_sample = True,
            seed = None,
            repetition_penalty = 1.03,
            details = True,
            typical_p = 0.5,
            watermark = False,
            priority = 5,
            timeout = None,
        )
    )
]


infer = dict(partitioner=dict(type=NaivePartitioner),
             runner=dict(
                 type=LocalAPIRunner,
                 max_num_workers=2,
                 task=dict(type=OpenICLInferTask)), )

work_dir = 'outputs/api-mindie-stream/' # 工作路径