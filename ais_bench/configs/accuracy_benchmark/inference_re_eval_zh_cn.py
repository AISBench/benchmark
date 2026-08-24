from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local import LocalRunner
from ais_bench.benchmark.tasks import OpenICLApiInferTask
from ais_bench.benchmark.datasets import gsm8k_postprocess, gsm8k_dataset_postprocess

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.demo.demo_gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat

models = vllm_api_general_chat
models[0]["host_ip"] = "localhost"
models[0]["host_port"] = 8080

# 替换或修改答案的提取函数实现
datasets[0]['eval_cfg']['pred_postprocessor'] = dict(type=gsm8k_postprocess)
datasets[0]['eval_cfg']['dataset_postprocessor'] = dict(type=gsm8k_dataset_postprocess)

work_dir = "outputs/default/"

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLApiInferTask),
    ),
)
